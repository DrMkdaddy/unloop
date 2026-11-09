import test from "node:test"
import assert from "node:assert/strict"

import plugin from "../src/index.ts"

const LINE = "The quick brown fox jumps over the lazy dog.  \n"

function fakeContext(options: Record<string, any> = {}) {
  const tools: any[] = []
  const commands: any[] = []
  const hooks: Array<[string, (event: any) => Promise<void>]> = []
  const synthetics: string[] = []
  const storage = new Map<string, any>()
  let assistant = ""

  const ctx = {
    options,
    tool: {
      transform: async (cb: (editor: any) => void) => {
        cb({ namespace: () => {}, add: (tool: any) => tools.push(tool) })
      },
    },
    command: {
      transform: async (cb: (editor: any) => void) => {
        cb({ add: (command: any) => commands.push(command) })
      },
    },
    session: {
      hook: async (name: string, cb: (event: any) => Promise<void>) => {
        hooks.push([name, cb])
      },
      context: async () =>
        assistant ? [{ type: "assistant", content: [{ type: "text", text: assistant }] }] : [],
      synthetic: async ({ text }: { text: string }) => {
        synthetics.push(text)
      },
    },
    storage: {
      get: async (key: string) => storage.get(key),
      set: async (key: string, value: any) => {
        storage.set(key, value)
      },
    },
  }

  return {
    ctx,
    tools,
    commands,
    hooks,
    synthetics,
    setAssistant: (text: string) => {
      assistant = text
    },
  }
}

test("registers a namespaced scan tool and a /unloop command", async () => {
  const fake = fakeContext()
  await plugin.setup(fake.ctx as any)

  assert.equal(fake.tools.length, 1)
  assert.equal(fake.tools[0].name, "scan")
  assert.equal(fake.commands.length, 1)
  assert.equal(fake.commands[0].name, "unloop")
  assert.equal(fake.hooks.length, 0) // guard is opt-in
})

test("the scan tool reports the loop and a trimmed text", async () => {
  const fake = fakeContext()
  await plugin.setup(fake.ctx as any)

  const result = await fake.tools[0].execute({ text: LINE.repeat(5) }, {})
  const data = JSON.parse(result.content)
  assert.equal(data.loop, true)
  assert.equal(data.repetitions, 5)
  assert.equal(data.trimmed, LINE)
})

test("/unloop scans the last assistant message", async () => {
  const fake = fakeContext()
  await plugin.setup(fake.ctx as any)
  fake.setAssistant(LINE.repeat(4))

  await fake.commands[0].execute({
    sessionID: "ses_test",
    prompt: { text: "" },
    delivery: "steer",
  })

  assert.match(fake.synthetics.at(-1) ?? "", /loop found/)
  assert.match(fake.synthetics.at(-1) ?? "", /4×/)
})

test("the idle guard reports once per message and respects coverage", async () => {
  const fake = fakeContext({ guard: true })
  await plugin.setup(fake.ctx as any)

  assert.equal(fake.hooks.length, 1)
  assert.equal(fake.hooks[0][0], "experimental.idle")

  fake.setAssistant(LINE.repeat(5))
  await fake.hooks[0][1]({ sessionID: "ses_test" })
  assert.equal(fake.synthetics.length, 1)

  await fake.hooks[0][1]({ sessionID: "ses_test" })
  assert.equal(fake.synthetics.length, 1) // same text: deduped

  fake.setAssistant(`${LINE}done`)
  await fake.hooks[0][1]({ sessionID: "ses_test" })
  assert.equal(fake.synthetics.length, 1) // coverage below 25%
})
