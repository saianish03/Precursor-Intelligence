import { Link } from "react-router"
import type { Schemas } from "@/api/client"
import { DecileMeter, td, th } from "@/components/ui"
import { displayName, fmtP, titleCase } from "@/lib/format"

/**
 * Facility rows. Columns drop by priority on narrow screens: chain first, then place. `decile` can be turned off
 * where every row shares one, as in the riskiest list.
 */
export function FacilityTable({
  rows,
  caption,
  rank = false,
  decile = true,
}: {
  rows: Schemas["FacilityRow"][]
  caption: string
  rank?: boolean
  decile?: boolean
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr>
            {rank && <th className={th}>Rank</th>}
            <th className={th}>Facility</th>
            <th className={`${th} hidden sm:table-cell`}>Place</th>
            <th className={`${th} hidden lg:table-cell`}>Chain</th>
            {decile && <th className={th}>Tenth of the month</th>}
            <th className={`${th} text-right`}>Probability</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.ccn} className="hover:bg-ground">
              {rank && <td className={`${td} text-muted`}>{i + 1}</td>}
              <td className={td}>
                <Link to={`/facilities/${r.ccn}`} className="font-semibold text-link hover:underline">
                  {displayName(r.name) || `CCN ${r.ccn}`}
                </Link>
                <div className="text-muted">
                  <code>{r.ccn}</code>
                  <span className="sm:hidden">
                    {" "}
                    {titleCase(r.city)}, {r.state}
                  </span>
                </div>
              </td>
              <td className={`${td} hidden sm:table-cell`}>
                {titleCase(r.city)}, {r.state}
              </td>
              <td className={`${td} hidden lg:table-cell`}>{r.chain_name ? displayName(r.chain_name) : <span className="text-muted">Independent</span>}</td>
              {decile && (
                <td className={td}>
                  <DecileMeter decile={r.risk_decile} />
                </td>
              )}
              <td className={`${td} text-right font-semibold`}>{fmtP(r.p)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
