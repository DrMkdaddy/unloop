// Teacher-forced DRY vs DRY-C measurement.
//
// Build against a llama.cpp tree with the DRY-C patch applied:
//
//   g++ -O2 -std=c++17 bench/teacher_forced.cpp \
//       -I/path/to/llama.cpp/include -I/path/to/llama.cpp/ggml/include \
//       -L/path/to/build/bin -lllama -Wl,-rpath,/path/to/build/bin \
//       -o teacher_forced
//
// The program tokenizes a unit repeated N times, decodes it, feeds the prompt
// tokens to the sampler as if they had been generated, and prints the top
// tokens after the sampler. That is the paper's teacher-forced setup: a fixed
// context, one forward pass, no sampling luck.
//
// Usage:
//   teacher_forced MODEL "k: v\n" 16

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string token_text(const llama_vocab * vocab, llama_token id) {
    char buf[256];
    int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    if (n < 0) {
        return "?";
    }
    return std::string(buf, n);
}

static std::string json_escape(const std::string & s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;     break;
        }
    }
    return out;
}

static void softmax(std::vector<float> & x) {
    float m = x[0];
    for (float v : x) m = std::max(m, v);
    double z = 0.0;
    for (float & v : x) { v = std::exp(v - m); z += v; }
    for (float & v : x) v = (float) (v / z);
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL UNIT REPS [top-k] [count-allowed]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const std::string unit = argv[2];
    const int reps = std::atoi(argv[3]);
    const int top_k = argc > 4 ? std::atoi(argv[4]) : 10;
    const int count_allowed = argc > 5 ? std::atoi(argv[5]) : 2;

    std::string prompt;
    for (int i = 0; i < reps; ++i) prompt += unit;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens(prompt.size() + 8);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                                  tokens.data(), (int32_t) tokens.size(), true, true);
    if (n_tokens < 0) {
        fprintf(stderr, "tokenization failed\n");
        return 1;
    }
    tokens.resize(n_tokens);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = (uint32_t) std::max(512, n_tokens);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    const float * logits = llama_get_logits_ith(ctx, -1);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const char * breakers[] = {"\n", ":", "\"", "*"};

    struct Config {
        const char * name;
        int count_min;   // 0 = plain DRY
        int allowed;     // length-term allowed (count-only sets this high)
        float count_base;
    } configs[] = {
        {"top-p",      0,      2, 3.0f},
        {"dry",        0,      2, 3.0f},
        {"dry-c",     24,      2, 3.0f},
        {"count-only", 24, 100000, 3.0f},
    };

    printf("{\"reps\": %d, \"unit\": \"%s\", \"tokens\": %d, \"configs\": [\n", reps, json_escape(unit).c_str(), n_tokens);
    for (size_t ci = 0; ci < sizeof(configs) / sizeof(configs[0]); ++ci) {
        const Config & c = configs[ci];

        llama_sampler * smpl = nullptr;
        if (c.count_min > 0) {
            smpl = llama_sampler_init_dry_count(vocab, 0.8f, 1.75f, c.allowed, 4096,
                                                breakers, 4, c.count_min, count_allowed, c.count_base);
        } else if (c.name[0] == 'd') {
            smpl = llama_sampler_init_dry(vocab, 0.8f, 1.75f, c.allowed, 4096, breakers, 4);
        }
        // top-p: no sampler at all

        std::vector<llama_token_data> cur;
        cur.reserve(n_vocab);
        for (llama_token i = 0; i < n_vocab; ++i) {
            cur.push_back({i, logits[i], 0.0f});
        }
        llama_token_data_array cur_p = {cur.data(), cur.size(), -1, false};

        if (smpl) {
            for (llama_token t : tokens) {
                llama_sampler_accept(smpl, t);
            }
            llama_sampler_apply(smpl, &cur_p);
        }

        std::vector<float> probs(n_vocab, -INFINITY);
        for (size_t i = 0; i < cur_p.size; ++i) probs[cur_p.data[i].id] = cur_p.data[i].logit;
        softmax(probs);

        std::vector<int> order(n_vocab);
        for (int i = 0; i < n_vocab; ++i) order[i] = i;
        std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                          [&](int a, int b) { return probs[a] > probs[b]; });

        printf("  {\"name\": \"%s\", \"top\": [", c.name);
        for (int i = 0; i < top_k; ++i) {
            int id = order[i];
            printf("%s{\"id\": %d, \"token\": \"%s\", \"prob\": %.8f}",
                   i ? ", " : "", id, json_escape(token_text(vocab, id)).c_str(), probs[id]);
        }
        printf("]}%s\n", ci + 1 < sizeof(configs) / sizeof(configs[0]) ? "," : "");

        if (smpl) llama_sampler_free(smpl);
    }
    printf("]}\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
