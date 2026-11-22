import { Menu, X } from "lucide-react"
import { useState } from "react"
import { Link, NavLink, Outlet, useLocation } from "react-router"
import { useStatus } from "@/api/queries"
import { Skeleton } from "@/components/states"
import { cn, fmtDate, fmtMonth, modelName } from "@/lib/format"
import { useSession } from "./auth"
import { Wordmark } from "./Wordmark"

const NAV = [
  { to: "/", label: "This month", end: true },
  { to: "/facilities", label: "Facilities", end: false },
  { to: "/model", label: "Model health", end: false },
]

export function Shell() {
  const [open, setOpen] = useState(false)
  const { pathname } = useLocation()
  const [lastPath, setLastPath] = useState(pathname)
  if (pathname !== lastPath) {
    setLastPath(pathname)
    setOpen(false)
  }
  return (
    <div className="min-h-dvh md:grid md:grid-cols-[232px_minmax(0,1fr)]">
      <a href="#main" className="sr-only z-30 bg-paper p-2 focus:not-sr-only focus:fixed focus:top-2 focus:left-2">
        Skip to content
      </a>
      <aside className="bg-navy text-rail-text md:sticky md:top-0 md:h-dvh">
        <div className="flex items-center justify-between px-5 py-4 md:block md:pt-5 md:pb-6">
          <Link to="/" className="rounded-sm">
            <Wordmark />
          </Link>
          <button
            type="button"
            className="rounded p-2 text-paper md:hidden"
            aria-expanded={open}
            aria-controls="primary-nav"
            onClick={() => setOpen((o) => !o)}
          >
            {open ? <X aria-hidden className="size-5" /> : <Menu aria-hidden className="size-5" />}
            <span className="sr-only">Menu</span>
          </button>
        </div>
        <nav id="primary-nav" aria-label="Primary" className={cn("pb-3 md:block", !open && "hidden")}>
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) =>
                cn(
                  "block px-5 py-2.5 font-medium text-rail-text hover:bg-navy-2 hover:text-paper",
                  isActive && "bg-navy-2 font-semibold text-paper",
                )
              }
            >
              {n.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <div className="min-w-0">
        <Freshness />
        <main id="main" className="px-4 py-6 md:px-7">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

/** The month and model behind every number on the page, and anything a reader should know before trusting them. */
function Freshness() {
  const status = useStatus()
  const session = useSession()
  const s = status.data
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-1.5 border-b border-rule bg-paper px-4 py-2.5 text-sm text-muted md:px-7">
      {s ? (
        <>
          <span>
            Scores for <b className="font-semibold text-ink">{fmtMonth(s.score_month)}</b>
          </span>
          <span>
            Model <b className="font-semibold text-ink">v{s.model_version}, {modelName(s.candidate)}</b>
          </span>
          <span>
            Scored <b className="font-semibold text-ink">{fmtDate(s.scored_at)}</b>
          </span>
          {!s.gold_verified && (
            <Link to="/model#data" className="rounded-sm bg-attn-tint px-2 py-0.5 font-semibold text-attn">
              Gold data not verified
            </Link>
          )}
          {s.stale_refresh && (
            <Link to="/model#data" className="rounded-sm bg-attn-tint px-2 py-0.5 font-semibold text-attn">
              Labels from an older month
            </Link>
          )}
          {s.flags.length > 0 && (
            <Link to="/model#drift" className="text-link hover:underline">
              {s.flags.length} data flag{s.flags.length > 1 ? "s" : ""}
            </Link>
          )}
        </>
      ) : status.isPending ? (
        <Skeleton className="h-4 w-96" />
      ) : (
        <span className="text-attn">The month being served couldn't be loaded.</span>
      )}
      <span className="ml-auto flex items-center gap-3">
        {session.fixtures ? (
          <span className="rounded-sm bg-ink px-2 py-0.5 font-semibold text-paper">Recorded data, no API</span>
        ) : (
          <>
            {session.email && <span>{session.email}</span>}
            <button type="button" onClick={session.signOut} className="font-semibold text-link hover:underline">
              Sign out
            </button>
          </>
        )}
      </span>
    </div>
  )
}
