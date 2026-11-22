import { Link, useParams } from "react-router"
import { ApiError, type Schemas } from "@/api/client"
import { useFacility, useStatus } from "@/api/queries"
import { EvidenceChip, EvidenceProvider, MarkBar, useRecord } from "@/components/evidence"
import { Empty, ErrorPanel, Skeleton, TableSkeleton } from "@/components/states"
import { Bar, DecileMeter, Section, td } from "@/components/ui"
import { decileLabel, displayName, fmtInt, fmtMonth, fmtP, fmtPct, titleCase } from "@/lib/format"
import { CitationStrip, CitationTable, PenaltyTable, SeverityGrid } from "./Citations"

export function FacilityPage() {
  const { ccn = "" } = useParams()
  const valid = /^[0-9A-Z]{6}$/.test(ccn)
  const profile = useFacility(ccn, valid)
  if (!valid) return <Empty title={`"${ccn}" isn't a CCN`}>A CCN is 6 characters, digits and capital letters.</Empty>
  if (profile.error instanceof ApiError && profile.error.status === 404)
    return (
      <Empty title={`No scored facility with CCN ${ccn} this month`}>
        It may have closed or not been scored. <Link to="/facilities" className="text-link underline">Search facilities</Link>
      </Empty>
    )
  if (profile.isError) return <ErrorPanel error={profile.error} what="this facility" onRetry={() => profile.refetch()} />
  if (!profile.data) return <ProfileSkeleton />
  return (
    <EvidenceProvider>
      <Profile p={profile.data} />
      <MarkBar />
    </EvidenceProvider>
  )
}

function Profile({ p }: { p: Schemas["FacilityProfile"] }) {
  const f = p.facility
  const base = useStatus().data?.base_rate
  const month = fmtMonth(p.data_as_of.score_month)
  const scoreRec = useRecord(p.score.evidence_id)
  return (
    <>
      <header className="mb-6 grid gap-5 border-b border-rule pb-5 md:grid-cols-[1fr_auto]">
        <div className="min-w-0">
          <nav aria-label="Breadcrumb" className="mb-1 text-sm">
            <Link to="/facilities" className="text-link hover:underline">
              Facilities
            </Link>
          </nav>
          <h1 className="text-xl font-bold tracking-[-0.01em]">{displayName(f.name) || `CCN ${f.ccn}`}</h1>
          <p className="mt-1.5 flex flex-wrap gap-x-5 gap-y-1 text-muted">
            <span>
              CCN <code className="text-ink">{f.ccn}</code>
            </span>
            <span>
              {titleCase(f.address)}, {titleCase(f.city)}, {f.state} {f.zip}
            </span>
            {f.beds !== null && <span>{fmtInt(f.beds)} certified beds</span>}
            <span>{f.chain_name ? `Chain: ${displayName(f.chain_name)}` : f.chain_id ? `Chain ${f.chain_id}` : "Independent"}</span>
          </p>
        </div>
        <div {...scoreRec} className={`${scoreRec.className} md:max-w-xs md:text-right`}>
          <div className="text-display font-bold text-risk">{fmtP(p.score.p)}</div>
          <p className="mt-1.5 text-sm text-muted">
            Model probability of a G to L citation in the next 12 months.{base !== undefined && ` Base rate ${fmtPct(base)}.`}
            {/* below lg the "About this score" caveat stacks far below, so it also sits here */}
            <span className="mt-1 block lg:hidden">
              An estimate from CMS data as of {month}, not a prediction that a citation will occur.
            </span>
          </p>
          <div className="mt-2 text-sm md:flex md:justify-end">
            <DecileMeter decile={p.score.risk_decile} />
          </div>
        </div>
      </header>

      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_400px]">
        <div className="grid min-w-0 content-start gap-8">
          <Section title="What moves the score" note="contribution to the model's log-odds">
            <Drivers reasons={p.reasons} />
          </Section>
          <Section title="Citations, last 36 months" note={`to ${month}`}>
            <CitationStrip citations={p.citations} scoreMonth={p.data_as_of.score_month} />
            <SeverityGrid citations={p.citations} />
            <CitationTable citations={p.citations} />
          </Section>
          <Section title="Penalties, last 36 months">
            <PenaltyTable penalties={p.penalties} />
          </Section>
        </div>
        <aside className="grid content-start gap-8 lg:sticky lg:top-4 lg:self-start">
          <Section title="About this score">
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
              <dt className="text-muted">Month</dt>
              <dd>{month}</dd>
              <dt className="text-muted">Standing</dt>
              <dd>
                {decileLabel(p.score.risk_decile)}, percentile {p.score.risk_percentile.toFixed(1)}
              </dd>
              <dt className="text-muted">Model</dt>
              <dd>Version {p.data_as_of.model_version}</dd>
              <dt className="text-muted">Gold run</dt>
              <dd>
                <code>{p.data_as_of.gold_run_id}</code>
              </dd>
              <dt className="text-muted">Record</dt>
              <dd>
                <EvidenceChip id={p.score.evidence_id} />
              </dd>
            </dl>
            <p className="mt-3 text-sm text-muted">
              A model estimate from CMS data published by {month}, not a prediction that a citation will occur.
            </p>
          </Section>
          {p.chain && <ChainPanel chain={p.chain} ccn={f.ccn} name={f.chain_name} />}
        </aside>
      </div>
    </>
  )
}

function Drivers({ reasons }: { reasons: Schemas["Reason"][] }) {
  const max = Math.max(...reasons.map((r) => Math.abs(r.contribution_logodds)), 1e-9)
  const raises = reasons.filter((r) => r.direction === "raises")
  const lowers = reasons.filter((r) => r.direction === "lowers")
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <DriverList title="Raises risk" rows={raises} max={max} color="var(--color-risk)" />
      <DriverList title="Lowers risk" rows={lowers} max={max} color="var(--color-lowers)" />
    </div>
  )
}

function DriverList({ title, rows, max, color }: { title: string; rows: Schemas["Reason"][]; max: number; color: string }) {
  return (
    <div>
      <h3 className="mb-1 text-sm font-semibold text-muted">{title}</h3>
      {rows.length === 0 ? (
        <p className="text-sm text-muted">None among the top reasons.</p>
      ) : (
        <ol>
          {rows.map((r) => (
            <DriverRow key={r.evidence_id} r={r} max={max} color={color} />
          ))}
        </ol>
      )}
    </div>
  )
}

function DriverRow({ r, max, color }: { r: Schemas["Reason"]; max: number; color: string }) {
  const rec = useRecord(r.evidence_id)
  return (
    <li {...rec} className={`${rec.className} grid grid-cols-[1fr_6rem] items-center gap-3 border-b border-rule-soft py-1.5`}>
      <span className="text-sm">
        {r.label} <span className="font-semibold">{r.feature_value ?? "not reported"}</span>
      </span>
      <span title={`${r.contribution_logodds >= 0 ? "+" : ""}${r.contribution_logodds.toFixed(3)} log-odds`}>
        <Bar value={r.contribution_logodds} max={max} color={color} />
      </span>
    </li>
  )
}

function ChainPanel({ chain, ccn, name }: { chain: Schemas["Chain"]; ccn: string; name: string | null }) {
  const rec = useRecord(chain.evidence_id ?? `chain:${chain.chain_id}`)
  const rows: [string, string][] = [
    ["Facilities", fmtInt(chain.n_facilities)],
    ["Certified beds", fmtInt(chain.beds)],
    ["Expected with a G to L citation", chain.expected_facilities_with_event.toFixed(1)],
    ["Chance of at least one, if facilities were independent", chain.p_any_event > 0.999 ? "Over 99.9%" : fmtPct(chain.p_any_event)],
    ["In the riskiest tenth", fmtPct(chain.share_top_decile, 0)],
  ]
  return (
    <Section title={name ? `Chain: ${displayName(name)}` : `Chain ${chain.chain_id}`}>
      <table {...rec} className={`${rec.className} w-full border-collapse text-sm`}>
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}>
              <th scope="row" className={`${td} text-left font-normal text-muted`}>
                {k}
              </th>
              <td className={`${td} text-right font-semibold`}>{v}</td>
            </tr>
          ))}
          <tr>
            <th scope="row" className={`${td} text-left font-normal text-muted`}>
              Riskiest facility
            </th>
            <td className={`${td} text-right`}>
              {chain.worst_ccn === ccn ? (
                "This facility"
              ) : (
                <Link to={`/facilities/${chain.worst_ccn}`} className="font-semibold text-link hover:underline">
                  <code>{chain.worst_ccn}</code>
                </Link>
              )}
            </td>
          </tr>
        </tbody>
      </table>
      {chain.evidence_id && (
        <p className="mt-2 text-sm">
          <EvidenceChip id={chain.evidence_id} />
        </p>
      )}
    </Section>
  )
}

function ProfileSkeleton() {
  return (
    <div role="status" aria-label="Loading the facility">
      <Skeleton className="mb-2 h-7 w-80" />
      <Skeleton className="mb-8 h-4 w-[32rem] max-w-full" />
      <TableSkeleton rows={10} />
    </div>
  )
}
