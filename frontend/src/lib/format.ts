import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs))

const utc = { timeZone: "UTC" } as const
// Data dates are calendar dates (YYYY-MM-DD); parse them as UTC so no time zone shifts the day.
const day = (iso: string) => new Date(`${iso.slice(0, 10)}T00:00:00Z`)

export const fmtDate = (iso: string | null | undefined) =>
  iso ? day(iso).toLocaleDateString("en-US", { ...utc, day: "numeric", month: "short", year: "numeric" }) : "None"

export const fmtMonth = (iso: string) =>
  day(iso.length === 7 ? `${iso}-01` : iso).toLocaleDateString("en-US", { ...utc, month: "long", year: "numeric" })

export const fmtTimestamp = (iso: string) => new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })

export const fmtP = (p: number) => p.toFixed(3)
export const fmtPct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`
export const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US")
export const fmtMoney = (n: number | null) =>
  n === null ? "None" : n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 })

/** "Riskiest tenth", "6th tenth"... decile 1 is the riskiest tenth of the month. */
export function decileLabel(d: number) {
  if (d === 1) return "Riskiest tenth"
  if (d === 10) return "Least risky tenth"
  const suffix = d === 2 ? "nd" : d === 3 ? "rd" : "th"
  return `${d}${suffix} tenth`
}

/** CMS scope and severity letters G to L are harm or immediate jeopardy: the outcome the model predicts. */
export const isHarm = (sev: string) => sev >= "G" && sev <= "L"

export const titleCase = (s: string | null | undefined) =>
  s ? s.toLowerCase().replace(/\b([a-z])/g, (m) => m.toUpperCase()) : ""

const SMALL = new Set(["a", "an", "and", "at", "for", "in", "of", "on", "the", "to"])

/**
 * A facility or chain name for display. CMS stores some names in capitals and some in mixed case; capitals become
 * title case, and mixed case is kept as written.
 * shortcut: a vowel-less word (HCC, NSG, LLC) is treated as an abbreviation; a list of real ones would be exact.
 */
export function displayName(name: string | null | undefined) {
  if (!name) return ""
  if (name !== name.toUpperCase()) return name
  const words = name.toLowerCase().split(/(\s+)/)
  return words
    .map((w, i) => {
      if (!/[a-z]/.test(w)) return w
      if (!/[aeiouy]/.test(w) && /^[a-z]{2,}$/.test(w)) return w.toUpperCase()
      const afterComma = i >= 2 && /,$/.test(words[i - 2] ?? "")
      if (i > 0 && !afterComma && SMALL.has(w)) return w
      return w.replace(/(^|[-/(])([a-z])/g, (_, p: string, c: string) => p + c.toUpperCase())
    })
    .join("")
}

const CANDIDATES: Record<string, string> = { lgbm: "LightGBM", lr: "logistic regression", ensemble: "LightGBM and logistic regression ensemble" }
export const modelName = (candidate: string) => CANDIDATES[candidate] ?? candidate

// The monthly decision (monitoring/decide.py DecisionName), as a sentence ending.
const DECISIONS: Record<string, string> = {
  SCORE: "scored with the current model, with no retraining",
  PROMOTE: "a new model was promoted",
  RETRAIN: "the model was retrained",
  RETUNE: "the model was re-tuned",
  SKIP: "nothing new to score, so the run was skipped",
  REFUSE: "the run was refused because a check failed",
}
export const decisionText = (d: string | null | undefined) => (d ? (DECISIONS[d] ?? d) : "no decision recorded")
