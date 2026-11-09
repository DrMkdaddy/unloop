import { Plugin } from "@opencode/plugin"

import { scanText, type ScanOptions, type ScanResult } from "./detector.ts"

type SessionReader = {
  session: {
    context(input: { sessionID: string }): Promise<readonly any[]>
    synthetic(input: { sessionID: string; text: string }): Promise<unknown>
  }
}

const DEFAULTS: Required<ScanOptions> = {
  minUnit: 40,
  maxUnit: 1600,
  minRepeats: 3,
  maxChars: 50_000,
}

function numberOption(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback
}

function scanOptionsFrom(options: Record<string, any>): Required<ScanOptions> {
  return {
    minUnit: numberOption(options.minUnit, DEFAULTS.minUnit),
    maxUnit: numberOption(options.maxUnit, DEFAULTS.maxUnit),
    minRepeats: numberOption(options.minRepeats, DEFAULTS.minRepeats),
    maxChars: numberOption(options.maxChars, DEFAULTS.maxChars),
  }
}

async function lastAssistantText(ctx: SessionReader, sessionID: string): Promise<string> {
  const messages = await ctx.session.context({ sessionID })
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i] as any
    if (message?.type !== "assistant") continue
    const parts = Array.isArray(message.content) ? message.content : []
    return parts
      .filter((part: any) => part?.type === "text" && typeof part.text === "string")
      .map((part: any) => part.text)
      .join("\n")
  }
  return ""
}

function report(result: ScanResult, text: string): string {
  if (!result.loop) {
    return `unloop: no verbatim loop found (${result.scanned} chars scanned).`
  }
  const removed = text.length - (result.trimmed?.length ?? text.length)
  const percent = Math.round(result.coverage * 100)
  return [
    `unloop: loop found — ${result.unitLength} chars repeated ${result.repetitions}× ` +
      `(${result.end - result.start} chars, ${percent}% of ${result.scanned}).`,
    `start=${result.start} end=${result.end}`,
    `unit: ${result.unit}`,
    `trimmed: ${result.trimmed?.length ?? 0} chars (${removed} removed)`,
  ].join("\n")
}

function fnv1a(text: string): string {
  let hash = 0x811c9dc5
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(16)
}

export default Plugin.define({
  id: "unloop",
  async setup(ctx) {
    const options = ctx.options as Record<string, any>
    const defaults = scanOptionsFrom(options)

    await ctx.tool.transform((editor) => {
      editor.namespace({
        name: "unloop",
        description: "Detect verbatim repetition loops in text",
      })
      editor.add({
        name: "scan",
        description:
          "Scan text for a verbatim repetition loop (the same unit repeated back to back). " +
          "Returns the unit length, repetition count, character offsets, and the text with " +
          "the redundant copies removed. Useful before sending a long structured answer or " +
          "after reading a long tool output. In Code Mode it is exposed as `tools.scan`.",
        input: {
          type: "object",
          properties: {
            text: { type: "string", description: "Text to scan." },
            minUnit: { type: "number", description: "Smallest repeated unit in characters." },
            maxUnit: { type: "number", description: "Largest repeated unit in characters." },
            minRepeats: { type: "number", description: "Consecutive copies required." },
            maxChars: { type: "number", description: "Only scan the first N characters." },
          },
          required: ["text"],
          additionalProperties: false,
        },
        execute: async (input) => {
          const value = input as {
            text: string
            minUnit?: number
            maxUnit?: number
            minRepeats?: number
            maxChars?: number
          }
          const result = scanText(value.text, {
            minUnit: numberOption(value.minUnit, defaults.minUnit),
            maxUnit: numberOption(value.maxUnit, defaults.maxUnit),
            minRepeats: numberOption(value.minRepeats, defaults.minRepeats),
            maxChars: numberOption(value.maxChars, defaults.maxChars),
          })
          return {
            content: JSON.stringify(result, null, 2),
            metadata: { loop: result.loop, repetitions: result.repetitions },
          }
        },
      })
    })

    await ctx.command.transform((editor) => {
      editor.add({
        name: "unloop",
        description:
          "Scan the last assistant message, or the text after the command, for a repetition loop",
        execute: async ({ sessionID, prompt }) => {
          const explicit = prompt.text.trim()
          const text = explicit || (await lastAssistantText(ctx, sessionID))
          if (!text) {
            await ctx.session.synthetic({ sessionID, text: "unloop: nothing to scan yet." })
            return
          }
          const result = scanText(text, defaults)
          await ctx.session.synthetic({ sessionID, text: report(result, text) })
        },
      })
    })

    // Opt-in: after a turn goes idle, look for a loop that took over the message and
    // leave a synthetic notice. It does not ask the model to continue on its own.
    if (options.guard === true) {
      await ctx.session.hook("experimental.idle", async ({ sessionID }) => {
        const text = await lastAssistantText(ctx, sessionID)
        if (!text) return

        const result = scanText(text, defaults)
        if (!result.loop || result.coverage < 0.25) return

        const key = `guard/${sessionID}`
        const hash = fnv1a(text)
        if ((await ctx.storage.get(key)) === hash) return
        await ctx.storage.set(key, hash)

        await ctx.session.synthetic({ sessionID, text: report(result, text) })
      })
    }
  },
})
