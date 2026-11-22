import { Search } from "lucide-react"
import { useSearchParams } from "react-router"
import { useFacilities, useOverview, type FacilityQuery } from "@/api/queries"
import { useSession } from "@/app/auth"
import { Empty, ErrorPanel, TableSkeleton } from "@/components/states"
import { PageTitle } from "@/components/ui"
import { decileLabel, fmtInt } from "@/lib/format"
import { FacilityTable } from "./FacilityTable"

const LIMIT = 50
const SORTS = { p_desc: "Highest probability", p_asc: "Lowest probability", name_asc: "Name, A to Z" } as const
const control = "rounded border border-rule bg-paper px-2.5 py-1.5"

export function FacilitiesPage() {
  const [params, setParams] = useSearchParams()
  const { fixtures } = useSession()
  const states = useOverview().data?.states.flatMap((s) => (s.state ? [s.state] : [])).sort() ?? []
  const query: FacilityQuery = {
    q: params.get("q") || undefined,
    state: params.get("state") || undefined,
    decile: params.get("decile") ? Number(params.get("decile")) : undefined,
    sort: (params.get("sort") as FacilityQuery["sort"]) || "p_desc",
    offset: Number(params.get("offset") ?? 0),
    limit: LIMIT,
  }
  const result = useFacilities(query)
  const set = (key: string, value: string) =>
    setParams((p) => {
      if (value) p.set(key, value)
      else p.delete(key)
      p.delete("offset")
      return p
    })
  const page = result.data
  const offset = query.offset ?? 0

  return (
    <>
      <PageTitle title="Facilities">
        Search this month's scored facilities by name, city or CCN, and narrow by state or tenth.
        {fixtures && " Recorded data holds the 500 riskiest facilities, so searches only reach those."}
      </PageTitle>
      <form
        role="search"
        className="mb-4 flex flex-wrap items-end gap-3"
        onSubmit={(e) => {
          e.preventDefault()
          set("q", String(new FormData(e.currentTarget).get("q") ?? "").trim())
        }}
      >
        <label className="grid gap-1 text-sm font-semibold">
          Name, city or CCN
          <span className="flex">
            <input
              name="q"
              defaultValue={query.q}
              key={query.q}
              maxLength={80}
              className={`${control} w-64 rounded-r-none font-normal`}
              placeholder="e.g. Mattoon or 145584"
            />
            <button type="submit" className="rounded-r border border-l-0 border-navy bg-navy px-3 text-paper">
              <Search aria-hidden className="size-4" />
              <span className="sr-only">Search</span>
            </button>
          </span>
        </label>
        <label className="grid gap-1 text-sm font-semibold">
          State
          <select value={query.state ?? ""} onChange={(e) => set("state", e.target.value)} className={`${control} font-normal`}>
            <option value="">All states</option>
            {states.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
        <label className="grid gap-1 text-sm font-semibold">
          Tenth of the month
          <select value={query.decile ?? ""} onChange={(e) => set("decile", e.target.value)} className={`${control} font-normal`}>
            <option value="">All tenths</option>
            {Array.from({ length: 10 }, (_, i) => (
              <option key={i + 1} value={i + 1}>
                {decileLabel(i + 1)}
              </option>
            ))}
          </select>
        </label>
        <label className="grid gap-1 text-sm font-semibold">
          Sort
          <select value={query.sort} onChange={(e) => set("sort", e.target.value)} className={`${control} font-normal`}>
            {Object.entries(SORTS).map(([v, label]) => (
              <option key={v} value={v}>
                {label}
              </option>
            ))}
          </select>
        </label>
      </form>

      {result.isError ? (
        <ErrorPanel error={result.error} what="facilities" onRetry={() => result.refetch()} />
      ) : !page ? (
        <TableSkeleton rows={12} />
      ) : page.total === 0 ? (
        <Empty title="No facilities match">Try a shorter name, another state, or all tenths.</Empty>
      ) : (
        <div aria-busy={result.isFetching}>
          <p className="mb-2 text-sm text-muted" aria-live="polite">
            {fmtInt(offset + 1)} to {fmtInt(Math.min(offset + LIMIT, page.total))} of {fmtInt(page.total)}
          </p>
          <FacilityTable rows={page.items} caption="Facilities matching the search" />
          <nav aria-label="Pages" className="mt-4 flex gap-2">
            <PageButton disabled={offset === 0} onClick={() => setParams((p) => (p.set("offset", String(Math.max(0, offset - LIMIT))), p))}>
              Previous
            </PageButton>
            <PageButton
              disabled={offset + LIMIT >= page.total}
              onClick={() => setParams((p) => (p.set("offset", String(offset + LIMIT)), p))}
            >
              Next
            </PageButton>
          </nav>
        </div>
      )}
    </>
  )
}

function PageButton(props: { disabled: boolean; onClick: () => void; children: string }) {
  return (
    <button
      type="button"
      {...props}
      className="rounded border border-rule bg-paper px-3 py-1.5 font-semibold hover:bg-ground disabled:cursor-not-allowed disabled:text-muted disabled:hover:bg-paper"
    />
  )
}
