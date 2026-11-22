import type { ReactNode } from "react"
import { cn, decileLabel, isHarm } from "@/lib/format"

export function PageTitle({ title, children }: { title: ReactNode; children?: ReactNode }) {
  return (
    <header className="mb-6 max-w-4xl">
      <h1 className="text-xl font-bold tracking-[-0.01em]">{title}</h1>
      {children && <p className="mt-1.5 text-muted">{children}</p>}
    </header>
  )
}

export function Section({
  title,
  note,
  id,
  children,
  className,
}: {
  title: string
  note?: ReactNode
  id?: string
  children: ReactNode
  className?: string
}) {
  return (
    <section id={id} aria-label={title} className={cn("min-w-0 scroll-mt-6", className)}>
      <h2 className="mb-2.5 border-b border-ink pb-1.5 text-base font-bold">
        {title}
        {note && <span className="ml-2 font-normal text-muted">{note}</span>}
      </h2>
      {children}
    </section>
  )
}

/** A fixed decile scale, the same on every screen: oxblood for the riskiest tenth, fading to paper. */
export const decileColor = (d: number) =>
  `color-mix(in oklch, var(--color-risk) ${Math.max(8, (11 - d) * 10)}%, var(--color-paper))`

export function DecileMeter({ decile, showLabel = true }: { decile: number; showLabel?: boolean }) {
  return (
    <span className="inline-flex items-center gap-2" title={decileLabel(decile)}>
      {/* least risky tenth on the left, riskiest on the right: risk grows rightward, as every bar does */}
      <span aria-hidden className="inline-grid grid-cols-10 gap-px">
        {Array.from({ length: 10 }, (_, i) => (
          <span
            key={i}
            className="h-3 w-1.5"
            style={{ background: i === 10 - decile ? decileColor(decile) : "var(--color-rule-soft)" }}
          />
        ))}
      </span>
      <span className={cn("whitespace-nowrap", showLabel ? "sr-only sm:not-sr-only" : "sr-only")}>{decileLabel(decile)}</span>
    </span>
  )
}

export function Severity({ letter }: { letter: string }) {
  const harm = isHarm(letter)
  return (
    <span
      title={harm ? "Harm or immediate jeopardy (G to L)" : "No actual harm (A to F)"}
      className={cn(
        "inline-block min-w-6 rounded-sm px-1 text-center font-bold",
        harm ? "bg-risk text-paper" : "bg-sev-low text-ink",
      )}
    >
      {letter}
    </span>
  )
}

/** A horizontal bar on a fixed 0 to max scale, with its value as text so the bar is never the only signal. */
export function Bar({ value, max, color = "var(--color-risk)" }: { value: number; max: number; color?: string }) {
  return (
    <span aria-hidden className="block h-2 bg-rule-soft">
      <span className="block h-full" style={{ width: `${Math.min(100, (Math.abs(value) / max) * 100)}%`, background: color }} />
    </span>
  )
}

export const th = "border-b border-ink px-2 py-1.5 text-left text-sm font-semibold text-muted"
export const td = "border-b border-rule-soft px-2 py-1.5 align-top"
