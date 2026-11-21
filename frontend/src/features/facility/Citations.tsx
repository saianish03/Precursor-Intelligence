import { useState } from "react"
import type { Schemas } from "@/api/client"
import { EvidenceId, useEvidenceMark, useRecord } from "@/components/evidence"
import { Severity, td, th } from "@/components/ui"
import { cn, fmtDate, fmtMoney, isHarm } from "@/lib/format"

type Citation = Schemas["Citation"]
const MONTHS = 36 // the evidence window (configs/agents.yaml evidence.citation_months), fixed on every facility

const monthIndex = (iso: string) => Number(iso.slice(0, 4)) * 12 + Number(iso.slice(5, 7)) - 1

/** Surveys on one fixed 36-month axis, so two facilities read side by side. One mark per survey date. */
export function CitationStrip({ citations, scoreMonth }: { citations: Citation[]; scoreMonth: string }) {
  const { mark } = useEvidenceMark()
  const end = monthIndex(scoreMonth) // the cutoff: the first day of the score month
  const start = end - MONTHS
  const byDate = new Map<string, Citation[]>()
  for (const c of citations) byDate.set(c.survey_date, [...(byDate.get(c.survey_date) ?? []), c])
  const surveys = [...byDate.entries()].map(([date, cs]) => {
    const worst = cs.map((c) => c.scope_severity).sort().at(-1) ?? "A"
    const dayFrac = (Number(date.slice(8, 10)) - 1) / 31
    return { date, cs, worst, x: ((monthIndex(date) + dayFrac - start) / MONTHS) * 100 }
  })
  const years = Array.from({ length: 4 }, (_, i) => Math.floor(start / 12) + i + 1)
    .map((y) => ({ y, x: ((y * 12 - start) / MONTHS) * 100 }))
    .filter((t) => t.x > 0 && t.x < 100)

  return (
    <figure className="mb-5">
      <div className="relative h-14 border-b border-ink">
        {years.map((t) => (
          <span key={t.y} aria-hidden className="absolute top-0 bottom-0 border-l border-rule-soft" style={{ left: `${t.x}%` }} />
        ))}
        {surveys.map((s) => {
          const harm = isHarm(s.worst)
          return (
            <button
              key={s.date}
              type="button"
              onClick={() => mark(s.cs[0]!.evidence_id)}
              aria-label={`${fmtDate(s.date)}: ${s.cs.length} citation${s.cs.length > 1 ? "s" : ""}, worst severity ${s.worst}`}
              title={`${fmtDate(s.date)}: ${s.cs.length} citation${s.cs.length > 1 ? "s" : ""}, worst ${s.worst}`}
              className={cn(
                "absolute -translate-x-1/2 rounded-full border-2 border-paper hover:scale-125 focus-visible:scale-125",
                harm ? "bottom-5 size-3.5 bg-risk" : "bottom-1.5 size-3 bg-survey",
              )}
              style={{ left: `${s.x}%` }}
            />
          )
        })}
      </div>
      <div className="relative mt-1 h-4 text-sm text-muted" aria-hidden>
        {years.map((t) => (
          <span key={t.y} className="absolute -translate-x-1/2" style={{ left: `${t.x}%` }}>
            {t.y}
          </span>
        ))}
      </div>
      <figcaption className="mt-2 flex flex-wrap gap-x-5 text-sm text-muted">
        <span>
          <span aria-hidden className="mr-1.5 inline-block size-2.5 rounded-full bg-risk align-baseline" />
          Survey with a G to L citation (upper row)
        </span>
        <span>
          <span aria-hidden className="mr-1.5 inline-block size-2.5 rounded-full bg-survey align-baseline" />
          Other surveys with citations
        </span>
        <span>Select a mark to find its citations in the table.</span>
      </figcaption>
    </figure>
  )
}

const GRID = [
  { level: "Immediate jeopardy", letters: ["J", "K", "L"] },
  { level: "Actual harm", letters: ["G", "H", "I"] },
  { level: "No actual harm, more than minimal", letters: ["D", "E", "F"] },
  { level: "Potential for minimal harm", letters: ["A", "B", "C"] },
]

/** The CMS scope and severity grid with the facility's 36-month counts in each cell. */
export function SeverityGrid({ citations }: { citations: Citation[] }) {
  const count = (l: string) => citations.filter((c) => c.scope_severity === l).length
  return (
    <table className="mb-5 border-collapse text-sm">
      <caption className="mb-1.5 text-left font-semibold">Citations by scope and severity</caption>
      <thead>
        <tr>
          <th className="pr-3 text-left font-semibold text-muted">Severity</th>
          {["Isolated", "Pattern", "Widespread"].map((s) => (
            <th key={s} className="w-24 px-1 pb-1 text-left font-semibold text-muted">
              {s}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {GRID.map((row) => (
          <tr key={row.level}>
            <th className="py-0.5 pr-3 text-left font-normal">{row.level}</th>
            {row.letters.map((l) => {
              const n = count(l)
              const harm = isHarm(l)
              return (
                <td key={l} className="p-0.5">
                  <span
                    className={cn(
                      "flex justify-between rounded-sm px-2 py-1",
                      n === 0 ? "bg-ground text-muted" : harm ? "bg-risk font-bold text-paper" : "bg-sev-low font-semibold",
                    )}
                  >
                    <span>{l}</span>
                    <span>{n}</span>
                  </span>
                </td>
              )
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export function CitationTable({ citations }: { citations: Citation[] }) {
  const { marked } = useEvidenceMark()
  const [harmOnly, setHarmOnly] = useState(false)
  const [all, setAll] = useState(false)
  const rows = citations.filter((c) => !harmOnly || isHarm(c.scope_severity))
  // A marked citation is always rendered, so marking from the strip never lands on a hidden row.
  const markedAt = rows.findIndex((c) => c.evidence_id === marked)
  const shown = all || markedAt >= 15 ? rows : rows.slice(0, 15)
  return (
    <>
      <label className="mb-2 inline-flex items-center gap-2 text-sm">
        <input type="checkbox" checked={harmOnly} onChange={(e) => setHarmOnly(e.target.checked)} />
        Only G to L citations
      </label>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <caption className="sr-only">Citations in the last 36 months, newest first</caption>
          <thead>
            <tr>
              <th className={th}>Survey date</th>
              <th className={th}>F-tag</th>
              <th className={`${th} hidden md:table-cell`}>Category</th>
              <th className={th}>Severity</th>
              <th className={`${th} hidden lg:table-cell`}>Survey</th>
              <th className={`${th} hidden xl:table-cell`}>Corrected</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((c) => (
              <CitationRow key={c.evidence_id} c={c} />
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > shown.length && (
        <button type="button" onClick={() => setAll(true)} className="mt-2 font-semibold text-link hover:underline">
          Show all {rows.length} citations
        </button>
      )}
      {rows.length === 0 && <p className="py-3 text-muted">No {harmOnly ? "G to L " : ""}citations in the last 36 months.</p>}
    </>
  )
}

function CitationRow({ c }: { c: Citation }) {
  const rec = useRecord(c.evidence_id)
  return (
    <tr {...rec}>
      <td className={`${td} whitespace-nowrap`}>{fmtDate(c.survey_date)}</td>
      <td className={td}>
        <code>{c.ftag}</code>
        <span className="block font-mono text-[0.75rem] text-muted md:hidden"><EvidenceId id={c.evidence_id} /></span>
      </td>
      <td className={`${td} hidden md:table-cell`}>
        {c.deficiency_category.replace(/ Deficiencies$/, "")}
        <span className="block font-mono text-[0.75rem] whitespace-nowrap text-muted">{c.evidence_id}</span>
      </td>
      <td className={td}>
        <Severity letter={c.scope_severity} />
      </td>
      <td className={`${td} hidden lg:table-cell`}>
        {[c.is_standard && "Standard", c.is_complaint && "Complaint"].filter(Boolean).join(" and ") || "Other"}
      </td>
      <td className={`${td} hidden whitespace-nowrap xl:table-cell`}>{fmtDate(c.correction_date)}</td>
    </tr>
  )
}

export function PenaltyTable({ penalties }: { penalties: Schemas["Penalty"][] }) {
  if (penalties.length === 0) return <p className="py-2 text-muted">No fines or payment denials in the last 36 months.</p>
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <caption className="sr-only">Penalties in the last 36 months, newest first</caption>
        <thead>
          <tr>
            <th className={th}>Date</th>
            <th className={th}>Type</th>
            <th className={`${th} text-right`}>Amount or length</th>
            <th className={`${th} hidden xl:table-cell`}>Record</th>
          </tr>
        </thead>
        <tbody>
          {penalties.map((p) => (
            <PenaltyRow key={p.evidence_id} p={p} />
          ))}
        </tbody>
      </table>
    </div>
  )
}

function PenaltyRow({ p }: { p: Schemas["Penalty"] }) {
  const rec = useRecord(p.evidence_id)
  const days = p.payment_denial_length_days
  return (
    <tr {...rec}>
      <td className={`${td} whitespace-nowrap`}>{fmtDate(p.penalty_date)}</td>
      <td className={td}>
        {p.penalty_type}
        <span className="block font-mono text-[0.75rem] text-muted xl:hidden"><EvidenceId id={p.evidence_id} /></span>
      </td>
      <td className={`${td} text-right`}>
        {p.fine_amount !== null ? fmtMoney(p.fine_amount) : days !== null ? `${days} days from ${fmtDate(p.payment_denial_start_date)}` : "Not reported"}
      </td>
      <td className={`${td} hidden font-mono text-[0.75rem] whitespace-nowrap text-muted xl:table-cell`}>{p.evidence_id}</td>
    </tr>
  )
}
