import math
import random
import unittest

from dryc import DRYParams, apply, count_penalties, find_loop, find_text_loop, length_penalties, sample_dist
from dryc.sim import dry_cap, run_trial, summarize, unlock_recurrence


def uniform_logits(n, p):
    return [math.log(p)] * n


class TestCountTerm(unittest.TestCase):
    def params(self, **kw):
        base = dict(multiplier=1.0, base=1.1, allowed_length=100, penalty_last_n=8,
                    count_min_length=3, count_allowed=1, count_base=2.0)
        base.update(kw)
        return DRYParams(**base)

    def test_one_recurrence(self):
        # history [0,1,2] [0,1,2]: the earlier 0,1,2 is followed by 0
        got = sample_dist(apply(uniform_logits(4, 0.25), [0, 1, 2, 0, 1, 2], self.params()))
        self.assertAlmostEqual(got[0], 0.109232, places=6)
        self.assertAlmostEqual(got[1], 0.296923, places=6)
        self.assertAlmostEqual(got[2], 0.296923, places=6)
        self.assertAlmostEqual(got[3], 0.296923, places=6)

    def test_two_recurrences_grow(self):
        got = sample_dist(apply(uniform_logits(4, 0.25), [0, 1, 2] * 3, self.params(penalty_last_n=10)))
        self.assertAlmostEqual(got[0], 0.043163, places=5)
        self.assertAlmostEqual(got[1], 0.318946, places=5)

    def test_applies_to_breaker_tokens(self):
        # token 3 is a single-token sequence breaker, but the count term still
        # penalizes it: breakers cap the length term, not the count term.
        p = self.params(sequence_breakers=((3,),), penalty_last_n=8)
        got = sample_dist(apply(uniform_logits(4, 0.25), [0, 1, 2, 3, 0, 1, 2], p))
        self.assertAlmostEqual(got[3], 0.109232, places=6)

    def test_no_recurrence_changes_nothing(self):
        got = sample_dist(apply(uniform_logits(4, 0.25), [0, 1, 2, 3, 1, 0], self.params()))
        for x in got:
            self.assertAlmostEqual(x, 0.25, places=9)

    def test_disabled(self):
        got = count_penalties([0, 1, 2] * 3, self.params(count_min_length=0))
        self.assertEqual(got, {})


class TestLengthTerm(unittest.TestCase):
    def test_matches_llama_cpp(self):
        # the same case as tests/test-sampling.cpp: test_dry(... 2.0, 1.1, 2, 5)
        p = DRYParams(multiplier=2.0, base=1.1, allowed_length=2, penalty_last_n=5,
                      count_min_length=0)
        got = sample_dist(apply(uniform_logits(5, 0.2), [0, 1, 2, 0, 1], p))
        expected = [0.241818, 0.241818, 0.032727, 0.241818, 0.241818]
        for g, e in zip(got, expected):
            self.assertAlmostEqual(g, e, places=6)

    def test_breaker_caps_length_term(self):
        # [0,1,2,3] repeats, but token 3 is a breaker: rep_limit becomes 0, so
        # the length term cannot see the match that a plain DRY run would see.
        hist = [0, 1, 2, 3, 0, 1, 2, 3]
        plain = DRYParams(multiplier=2.0, base=1.1, allowed_length=2, penalty_last_n=8,
                          count_min_length=0)
        blocked = DRYParams(multiplier=2.0, base=1.1, allowed_length=2, penalty_last_n=8,
                            sequence_breakers=((3,),), count_min_length=0)
        self.assertAlmostEqual(length_penalties(hist, plain)[0], 2.42, places=6)
        self.assertEqual(length_penalties(hist, blocked), {})

    def test_count_survives_the_breaker(self):
        # ... while the count term still sees the [0,1] recurrence
        p = DRYParams(multiplier=1.0, base=1.1, allowed_length=2, penalty_last_n=6,
                      sequence_breakers=((3,),), count_min_length=2,
                      count_allowed=1, count_base=2.0)
        self.assertEqual(count_penalties([0, 1, 2, 3, 0, 1], p), {2: 1.0})


class TestLoopDetector(unittest.TestCase):
    def test_token_loop(self):
        self.assertEqual(find_loop([0, 1] * 9, min_unit=1, max_unit=8), (2, 9))

    def test_text_loop(self):
        line = "the same line repeats here\n"
        self.assertEqual(find_text_loop(line * 5, min_unit=10, max_unit=40), (len(line), 5))

    def test_no_loop(self):
        self.assertIsNone(find_loop(list(range(100)), min_unit=4, max_unit=20))


class TestLockSimulation(unittest.TestCase):
    def test_dry_stays_locked_but_dry_c_escapes(self):
        p = DRYParams(multiplier=0.8, base=1.75, allowed_length=2, penalty_last_n=4096,
                      count_min_length=24, count_allowed=2, count_base=3.0)
        q, r, pi, g = 0.999, 0.0009, 0.9, 6
        delta = math.log(q / r) - math.log(pi / (1 - pi))
        self.assertGreater(delta, dry_cap(p, g))       # plain DRY cannot escape
        self.assertIsNotNone(unlock_recurrence(delta, p))

        dry = summarize([run_trial(q, r, pi, p, "dry", g, 200, random.Random(s)) for s in range(50)])
        cnt = summarize([run_trial(q, r, pi, p, "count", g, 200, random.Random(s)) for s in range(50)])
        self.assertEqual(dry["locked_fraction"], 1.0)
        self.assertEqual(cnt["locked_fraction"], 0.0)
        self.assertLessEqual(cnt["median_escape_step"], 10)


if __name__ == "__main__":
    unittest.main()
