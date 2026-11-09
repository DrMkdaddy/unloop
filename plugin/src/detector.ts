export interface ScanOptions {
  /** Smallest repeated unit, in characters. */
  minUnit?: number
  /** Largest repeated unit, in characters. */
  maxUnit?: number
  /** Consecutive copies required to call it a loop. */
  minRepeats?: number
  /** Only scan the first N characters. */
  maxChars?: number
}

export interface ScanResult {
  loop: boolean
  unitLength: number | null
  repetitions: number
  /** Character offsets of the repeated run in the scanned text. */
  start: number
  end: number
  /** Fraction of the scanned text occupied by the loop. */
  coverage: number
  /** The repeated unit, escaped and truncated for display. */
  unit: string | null
  /** The text with the redundant copies removed, or null when there is no loop. */
  trimmed: string | null
  scanned: number
}

const DEFAULTS = {
  minUnit: 40,
  maxUnit: 1600,
  minRepeats: 3,
  maxChars: 50_000,
}

/**
 * Find the largest verbatim repetition in `text`.
 *
 * This is the TypeScript port of `dryc.find_text_loop` from the unloop repo:
 * a unit of `minUnit`..`maxUnit` characters repeated at least `minRepeats`
 * times, back to back. Exact characters only; no whitespace folding.
 */
export function scanText(text: string, options: ScanOptions = {}): ScanResult {
  const minUnit = Math.max(1, Math.floor(options.minUnit ?? DEFAULTS.minUnit))
  const maxChars = Math.max(0, Math.floor(options.maxChars ?? DEFAULTS.maxChars))
  const minRepeats = Math.max(2, Math.floor(options.minRepeats ?? DEFAULTS.minRepeats))

  const scanned = text.slice(0, maxChars)
  const n = scanned.length
  const maxUnit = Math.min(
    Math.floor(options.maxUnit ?? DEFAULTS.maxUnit),
    Math.floor(n / minRepeats),
  )

  let best: { unit: number; reps: number; start: number } | null = null

  for (let unit = minUnit; unit <= maxUnit; unit++) {
    const lastStart = n - unit * minRepeats
    for (let start = 0; start <= lastStart; start++) {
      // Cheap reject before allocating substrings.
      if (scanned.charCodeAt(start) !== scanned.charCodeAt(start + unit)) continue

      let reps = 1
      let pos = start + unit
      while (pos + unit <= n) {
        // The first and last character check rejects most non-matches without a slice.
        if (
          scanned.charCodeAt(pos) !== scanned.charCodeAt(start) ||
          scanned.charCodeAt(pos + unit - 1) !== scanned.charCodeAt(start + unit - 1)
        ) {
          break
        }
        if (scanned.slice(start, start + unit) !== scanned.slice(pos, pos + unit)) break
        reps++
        pos += unit
      }

      if (reps >= minRepeats) {
        const total = unit * reps
        if (!best || total > best.unit * best.reps) {
          best = { unit, reps, start }
        }
      }
    }
  }

  if (!best) {
    return {
      loop: false,
      unitLength: null,
      repetitions: 0,
      start: 0,
      end: 0,
      coverage: 0,
      unit: null,
      trimmed: null,
      scanned: n,
    }
  }

  const { unit, reps, start } = best
  const end = start + unit * reps
  const repeated = text.slice(start, start + unit)
  const trimmed = text.slice(0, start + unit) + text.slice(end)

  return {
    loop: true,
    unitLength: unit,
    repetitions: reps,
    start,
    end,
    coverage: n === 0 ? 0 : (end - start) / n,
    unit: preview(repeated),
    trimmed,
    scanned: n,
  }
}

function preview(unit: string): string {
  const escaped = JSON.stringify(unit)
  return escaped.length > 160 ? `${escaped.slice(0, 157)}..."` : escaped
}
