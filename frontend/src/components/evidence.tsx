import { X } from "lucide-react"
import { createContext, Fragment, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react"
import { cn } from "@/lib/format"

/**
 * The citation rail: any evidence id on the page can be marked. Marking scrolls its record into view and keeps the
 * highlight until the reader dismisses it (Escape or the Clear button), so a mark is never lost to a stray hover.
 */
type Mark = { marked: string | null; mark: (id: string) => void; clear: () => void }
const MarkContext = createContext<Mark>({ marked: null, mark: () => {}, clear: () => {} })

export const recordDomId = (evidenceId: string) => `rec-${evidenceId.replace(/[^A-Za-z0-9_-]/g, "_")}`

export function EvidenceProvider({ children }: { children: ReactNode }) {
  const [marked, setMarked] = useState<string | null>(null)
  const mark = useCallback((id: string) => {
    setMarked(id)
    const el = document.getElementById(recordDomId(id))
    el?.scrollIntoView({ block: "center", behavior: "smooth" })
    el?.focus({ preventScroll: true })
  }, [])
  const clear = useCallback(() => setMarked(null), [])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setMarked(null)
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])
  const value = useMemo(() => ({ marked, mark, clear }), [marked, mark, clear])
  return <MarkContext.Provider value={value}>{children}</MarkContext.Provider>
}

export const useEvidenceMark = () => useContext(MarkContext)

/** Props for the element that holds one record, so a mark can find, focus and highlight it. */
export function useRecord(evidenceId: string) {
  const { marked } = useEvidenceMark()
  const on = marked === evidenceId
  return {
    id: recordDomId(evidenceId),
    tabIndex: -1,
    "data-marked": on || undefined,
    className: cn("scroll-mt-24 outline-none transition-colors duration-200", on && "bg-mark outline-2 outline-mark-edge"),
  }
}

/** An evidence id as a citation chip; activating it marks its record on the page. */
export function EvidenceChip({ id, label }: { id: string; label?: string }) {
  const { marked, mark } = useEvidenceMark()
  const on = marked === id
  return (
    <button
      type="button"
      onClick={() => mark(id)}
      aria-pressed={on}
      title={`Show the record ${id}`}
      className={cn(
        "rounded-sm px-1.5 py-px font-mono text-[0.75rem] whitespace-nowrap",
        on ? "bg-mark text-ink outline-1 outline-mark-edge" : "bg-link-tint text-link hover:underline",
      )}
    >
      {label ?? id}
    </button>
  )
}

export function MarkBar() {
  const { marked, clear } = useEvidenceMark()
  if (!marked) return null
  return (
    <div className="fixed right-6 bottom-6 z-20 flex items-center gap-3 rounded border border-mark-edge bg-mark px-3 py-2 text-sm shadow-overlay">
      <span>
        Marked <code>{marked}</code>
      </span>
      <button type="button" onClick={clear} className="inline-flex items-center gap-1 font-semibold hover:underline">
        <X aria-hidden className="size-3.5" /> Clear
      </button>
    </div>
  )
}

/** An evidence id that wraps only after its colons, never inside a date or a code. */
export function EvidenceId({ id }: { id: string }) {
  return id.split(":").map((part, i) => (
    <Fragment key={i}>
      {i > 0 && ":"}
      {i > 0 && <wbr />}
      {part}
    </Fragment>
  ))
}
