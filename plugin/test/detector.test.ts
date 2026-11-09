import test from "node:test"
import assert from "node:assert/strict"

import { scanText } from "../src/detector.ts"

const LINE = "The quick brown fox jumps over the lazy dog.  \n"

test("detects a repeated line", () => {
  const result = scanText(LINE.repeat(25))
  assert.equal(result.loop, true)
  assert.equal(result.unitLength, LINE.length)
  assert.equal(result.repetitions, 25)
  assert.equal(result.start, 0)
  assert.equal(result.end, LINE.length * 25)
  assert.equal(result.trimmed, LINE)
})

test("keeps the prefix and suffix when the loop is in the middle", () => {
  const unit = "0123456789"
  const text = `PREFIX|${unit.repeat(5)}|SUFFIX`
  const result = scanText(text, { minUnit: 10, maxUnit: 10 })
  assert.equal(result.loop, true)
  assert.equal(result.start, "PREFIX|".length)
  assert.equal(result.trimmed, `PREFIX|${unit}|SUFFIX`)
})

test("no loop for varied text", () => {
  const text = Array.from({ length: 100 }, (_, i) => `${i}: value ${i * 7}`).join("\n")
  const result = scanText(text)
  assert.equal(result.loop, false)
  assert.equal(result.trimmed, null)
})

test("respects minRepeats", () => {
  assert.equal(scanText(LINE.repeat(2)).loop, false)
  assert.equal(scanText(LINE.repeat(2), { minRepeats: 2 }).loop, true)
})

test("respects minUnit", () => {
  const short = "ab\n"
  assert.equal(scanText(short.repeat(20), { minUnit: 40 }).loop, false)
  assert.equal(scanText(short.repeat(20), { minUnit: 3 }).loop, true)
})

test("reports coverage and a preview", () => {
  const result = scanText(LINE.repeat(10))
  assert.equal(result.coverage, 1)
  assert.ok(result.unit?.includes("quick brown fox"))
})

test("caps the scanned text", () => {
  const result = scanText(LINE.repeat(100), { maxChars: 200 })
  assert.equal(result.scanned, 200)
})
