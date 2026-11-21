import { Link } from "react-router"
import { useOverview, useStatus } from "@/api/queries"
import { ErrorPanel, TableSkeleton } from "@/components/states"
import { Bar, decileColor, PageTitle, Section, td, th } from "@/components/ui"
import { FacilityTable } from "@/features/facilities/FacilityTable"
import { decileLabel, displayName, fmtInt, fmtMonth, fmtP, fmtPct } from "@/lib/format"

export function OverviewPage() {
  const overview = useOverview()
  const status = useStatus()
  const o = overview.data
  const s = status.data
  return (
    <>
      <PageTitle title={o ? fmtMonth(o.data_as_of.score_month) : "This month"}>
        {s
          ? `${fmtInt(s.n_facilities)} facilities scored by model v${s.model_version}. Each probability is the model's estimate of a G to L citation in the next 12 months; on the held-out test month, ${fmtPct(s.base_rate)} of facilities had one.`
          : "The month's scores, the riskiest facilities, and where risk concentrates."}
      </PageTitle>
      {overview.isError ? (
        <ErrorPanel error={overview.error} what="this month's overview" onRetry={() => overview.refetch()} />
      ) : (
        <div className="grid gap-8 lg:grid-cols-12">
          <Section title="Riskiest facilities" note="highest probability first" className="lg:col-span-8">
            {o ? <FacilityTable rows={o.riskiest} caption="The 25 riskiest facilities this month" rank decile={false} /> : <TableSkeleton rows={10} />}
            <Link to="/facilities" className="mt-3 inline-block font-semibold text-link hover:underline">
              Search all facilities
            </Link>
          </Section>
          <Section title="Average probability by tenth" note="fixed 0 to 1 scale" className="lg:col-span-4">
            {o ? (
              <table className="w-full border-collapse text-sm">
                <caption className="sr-only">Average model probability in each tenth of the month's facilities</caption>
                <thead>
                  <tr>
                    <th className={th}>Tenth</th>
                    <th className={`${th} w-1/2`}>
                      <span className="sr-only">Bar</span>
                    </th>
                    <th className={`${th} text-right`}>Average</th>
                  </tr>
                </thead>
                <tbody>
                  {o.deciles.map((d) => (
                    <tr key={d.risk_decile}>
                      <td className={td}>{decileLabel(d.risk_decile)}</td>
                      <td className={`${td} align-middle`}>
                        <Bar value={d.mean_p} max={1} color={decileColor(d.risk_decile)} />
                      </td>
                      <td className={`${td} text-right`}>{fmtP(d.mean_p)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <TableSkeleton rows={10} />
            )}
          </Section>
          <Section title="States with the most facilities in the riskiest tenth" className="lg:col-span-6">
            {o ? (
              <table className="w-full border-collapse text-sm">
                <caption className="sr-only">States ranked by facilities in the riskiest tenth</caption>
                <thead>
                  <tr>
                    <th className={th}>State</th>
                    <th className={`${th} text-right`}>Scored</th>
                    <th className={`${th} text-right`}>In riskiest tenth</th>
                    <th className={`${th} text-right`}>Share</th>
                  </tr>
                </thead>
                <tbody>
                  {o.states.slice(0, 10).map((r) => (
                    <tr key={r.state ?? "unknown"}>
                      <td className={td}>
                        {r.state ? (
                          <Link to={`/facilities?state=${r.state}&decile=1`} className="text-link hover:underline">
                            {r.state}
                          </Link>
                        ) : (
                          "Unknown"
                        )}
                      </td>
                      <td className={`${td} text-right`}>{fmtInt(r.n)}</td>
                      <td className={`${td} text-right`}>{fmtInt(r.n_top_decile)}</td>
                      <td className={`${td} text-right`}>{fmtPct(r.n_top_decile / r.n)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <TableSkeleton />
            )}
          </Section>
          <Section title="Chains with the most expected citations" className="lg:col-span-6">
            {o ? (
              <table className="w-full border-collapse text-sm">
                <caption className="sr-only">Chains ranked by the expected number of facilities with a G to L citation</caption>
                <thead>
                  <tr>
                    <th className={th}>Chain</th>
                    <th className={`${th} text-right`}>Facilities</th>
                    <th className={`${th} text-right`} title="The sum of the facilities' probabilities">
                      Expected with a G to L
                    </th>
                    <th className={`${th} text-right`}>In riskiest tenth</th>
                  </tr>
                </thead>
                <tbody>
                  {o.chains.map((c) => (
                    <tr key={c.chain_id}>
                      <td className={td}>{c.chain_name ? displayName(c.chain_name) : `Chain ${c.chain_id}`}</td>
                      <td className={`${td} text-right`}>{fmtInt(c.n_facilities)}</td>
                      <td className={`${td} text-right`}>{c.expected_facilities_with_event.toFixed(1)}</td>
                      <td className={`${td} text-right`}>{fmtPct(c.share_top_decile, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <TableSkeleton />
            )}
          </Section>
        </div>
      )}
    </>
  )
}
