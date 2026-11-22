import { AlertTriangle, RotateCw } from "lucide-react"
import type { ReactNode } from "react"
import { ApiError } from "@/api/client"
import { cn } from "@/lib/format"

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cn("animate-pulse rounded-sm bg-rule-soft", className)} />
}

/** Skeleton rows that keep the layout of the table they stand in for. */
export function TableSkeleton({ rows = 8 }: { rows?: number }) {
  return (
    <div role="status" aria-label="Loading" className="grid gap-2 py-2">
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className="h-7" />
      ))}
    </div>
  )
}

export function ErrorPanel({ error, onRetry, what }: { error: unknown; onRetry?: () => void; what: string }) {
  const e = error instanceof ApiError ? error : null
  const message = e?.message ?? "Something went wrong while loading this."
  return (
    <div role="alert" className="flex items-start gap-3 rounded border border-rule bg-paper p-4">
      <AlertTriangle aria-hidden className="mt-0.5 size-4 shrink-0 text-attn" />
      <div className="grid gap-1">
        <p className="font-semibold">Couldn't load {what}.</p>
        <p className="text-muted">{message}</p>
        {e?.requestId && (
          <p className="text-sm text-muted">
            Request <code>{e.requestId}</code>
          </p>
        )}
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="mt-1 inline-flex w-fit items-center gap-1.5 rounded border border-rule bg-paper px-3 py-1.5 text-sm font-semibold hover:bg-ground"
          >
            <RotateCw aria-hidden className="size-3.5" /> Try again
          </button>
        )}
      </div>
    </div>
  )
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded border border-dashed border-rule bg-paper px-4 py-8 text-center">
      <p className="font-semibold">{title}</p>
      {children && <div className="mt-1 text-muted">{children}</div>}
    </div>
  )
}
