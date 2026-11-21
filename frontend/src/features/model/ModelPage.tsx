import type { Schemas } from "@/api/client"
import { useOps, useStatus } from "@/api/queries"
import { Empty, ErrorPanel, TableSkeleton } from "@/components/states"
import { Bar, PageTitle, Section, td, th } from "@/components/ui"
import { decisionText, fmtInt, fmtMonth, fmtP, fmtPct, fmtTimestamp, modelName } from "@/lib/format"

const BASELINES: Record<string, string> = {
  prior_gplus_rule: "Prior G to L citation (rule)",
  severity_24m: "Citation severity, last 24 months",
  star_rating: "CMS star rating",
}

/** What a data flag means, from monitoring/status.py and training/train.py. */
function explainFlag(flag: string) {
  const [kind, feature] = flag.split(":")
  const what =
    kind === "skew" || kind === "psi"
      ? "differs between the training window and the scored month (population stability index over 0.25)"
      : kind === "drift"
        ? "moved since the previous month (population stability index over 0.25)"
        : kind === "regime"
          ? "is reported under a different CMS method in part of the training window"
          : "was flagged by the pipeline"
  return { feature: feature ?? flag, what }
}

export function ModelPage() {
  const ops = useOps()
  const status = useStatus().data
  return (
    <>
      <PageTitle title="Model health">
        {status
          ? `Model v${status.model_version} (${modelName(status.candidate)}) scored ${fmtMonth(status.score_month)}: ${decisionText(status.decision)}.`
          : "How the scoring model was checked, and anything to know about this month's data."}
      </PageTitle>
      {ops.isError ? (
        <ErrorPanel error={ops.error} what="the model's checks" onRetry={() => ops.refetch()} />
      ) : !ops.data ? (
        <TableSkeleton rows={12} />
      ) : (
        <Health ops={ops.data} status={status} />
      )}
    </>
  )
}

function Health({ ops, status }: { ops: Schemas["Ops"]; status: Schemas["Status"] | undefined }) {
  const e = ops.evaluation
  const r = e.ranking
  const iv = e.intervals
  const ci = (k: string) => (iv[k] ? ` (95% CI ${fmtP(iv[k].ci_low)} to ${fmtP(iv[k].ci_high)})` : "")
  const metrics: [string, string, string][] = [
    ["PR-AUC", `${fmtP(r.pr_auc ?? 0)}${ci("pr_auc")}`, "How well the ranking finds the facilities that were cited. Chance equals the base rate."],
    ["Lift over the base rate", `${(r.pr_auc_lift ?? 0).toFixed(2)} times`, "PR-AUC divided by the base rate."],
    ["ROC-AUC", `${fmtP(r.roc_auc ?? 0)}${ci("roc_auc")}`, "The chance a cited facility ranks above an uncited one."],
    ["Precision in the riskiest 5%", fmtPct(r.precision_at_5 ?? 0), "Share of the top 5% that were cited within 12 months."],
    ["Precision in the riskiest 10%", fmtPct(r.precision_at_10 ?? 0), "Share of the top 10% that were cited."],
    ["Brier score", `${fmtP(e.probability.brier ?? 0)}${ci("brier")}`, "Mean squared error of the probabilities; lower is better."],
    ["Calibration error (ECE)", fmtP(e.probability.ece ?? 0), "Average gap between predicted and observed rates; lower is better."],
  ]
  // monitor's skew: and training's psi: say the same thing about a feature, so each feature and meaning shows once.
  const flags = [...new Map([...(status?.flags ?? []), ...e.flags].map((f) => [JSON.stringify(explainFlag(f)), f])).values()]
  return (
    <div className="grid gap-8 lg:grid-cols-12">
      <div className="grid content-start gap-8 lg:col-span-7">
        <Section
          title="How well it ranks and calibrates"
          note={`held-out month ${e.test_months.join(", ")}: ${fmtInt(e.rows)} facilities, ${fmtInt(e.positives)} cited`}
        >
          <table className="w-full border-collapse text-sm">
            <caption className="sr-only">Test-month metrics of the scoring model</caption>
            <tbody>
              {metrics.map(([k, v, why]) => (
                <tr key={k}>
                  <th scope="row" className={`${td} w-56 text-left font-normal`}>
                    {k}
                  </th>
                  <td className={td}>
                    <span className="font-semibold">{v}</span>
                    <span className="block text-muted">{why}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>

        <Section title="Calibration" note="predicted against observed, ten bins of predicted probability, lowest first">
          <Reliability bins={e.reliability} />
        </Section>
      </div>

      <div className="grid content-start gap-8 lg:col-span-5">
        <Section title="Against simple rules" note="PR-AUC, fixed 0 to 1 scale">
          <table className="w-full border-collapse text-sm">
            <caption className="sr-only">PR-AUC of the model and of the baselines</caption>
            <tbody>
              {[
                ["This model", r.pr_auc ?? 0, "var(--color-navy)"] as const,
                ...Object.entries(e.baselines).map(([k, v]) => [BASELINES[k] ?? k, v, "var(--color-muted)"] as const),
                ["Chance (the base rate)", r.base_rate ?? 0, "var(--color-rule)"] as const,
              ].map(([label, v, color]) => (
                <tr key={label}>
                  <th scope="row" className={`${td} text-left font-normal`}>
                    {label}
                  </th>
                  <td className={`${td} w-2/5 align-middle`}>
                    <Bar value={v} max={1} color={color} />
                  </td>
                  <td className={`${td} text-right font-semibold`}>{fmtP(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>

        <Section title="Data behind this month" id="data">
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
            <dt className="text-muted">Gold data verified</dt>
            <dd className={status?.gold_verified ? "" : "font-semibold text-attn"}>
              {status?.gold_verified ? "Yes" : "No: the gold build behind these scores hasn't passed verification"}
            </dd>
            <dt className="text-muted">Labels</dt>
            <dd>{status?.stale_refresh ? "From an older month than the features" : "From the same month as the features"}</dd>
            <dt className="text-muted">Scored</dt>
            <dd>{status ? fmtTimestamp(status.scored_at) : "Unknown"}</dd>
            <dt className="text-muted">Gold run</dt>
            <dd>
              <code>{status?.gold_run_id}</code>
            </dd>
          </dl>
        </Section>
        <Section title="Data flags" id="drift">
          {flags.length === 0 ? (
            <p className="text-sm text-muted">No feature drift or skew flagged this month.</p>
          ) : (
            <ul className="grid gap-2 text-sm">
              {flags.map((f) => {
                const x = explainFlag(f)
                return (
                  <li key={f} className="border-b border-rule-soft pb-2">
                    <code className="font-semibold">{x.feature}</code> {x.what}.
                  </li>
                )
              })}
            </ul>
          )}
        </Section>
        <Section title="Sanity checks">
          <p className="text-sm">
            A model trained on shuffled labels scores ROC-AUC{" "}
            <b>{fmtP(Number(e.sanity.shuffled_label_roc ?? 0))}</b>, chance level, as expected when nothing leaks the label.{" "}
            {Array.isArray(e.sanity.uncleared_leakage_flags) && e.sanity.uncleared_leakage_flags.length === 0
              ? "No leakage flags are open."
              : "Leakage flags are open: see the evaluation run."}
          </p>
        </Section>
        <Section title="Promotion gates">
          {ops.gates ? (
            <p className="text-sm">Outcome: {ops.gates.gate_outcome ?? "not recorded"}</p>
          ) : (
            <Empty title="Gate results aren't published yet">
              They arrive with the agents' monthly evidence build, from the run that registered this model.
            </Empty>
          )}
        </Section>
      </div>
    </div>
  )
}

function Reliability({ bins }: { bins: Schemas["Ops"]["evaluation"]["reliability"] }) {
  const size = 240
  const max = Math.min(1, Math.ceil(Math.max(...bins.flatMap((b) => [b.mean_predicted, b.observed_rate])) * 10) / 10)
  const at = (v: number) => (v / max) * size
  return (
    <div className="grid gap-5 sm:grid-cols-[auto_1fr]">
      <svg viewBox={`-30 -10 ${size + 40} ${size + 40}`} className="w-64" role="img" aria-label="Reliability plot: points near the diagonal mean the probabilities match observed rates">
        <rect x={0} y={0} width={size} height={size} fill="var(--color-paper)" stroke="var(--color-rule)" />
        <line x1={0} y1={size} x2={size} y2={0} stroke="var(--color-rule)" strokeDasharray="4 3" />
        {[0, max / 2, max].map((t) => (
          <g key={t} fontSize="11" fill="var(--color-muted)">
            <text x={at(t)} y={size + 16} textAnchor="middle">
              {t.toFixed(1)}
            </text>
            <text x={-8} y={size - at(t) + 4} textAnchor="end">
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        {bins.map((b) => (
          <circle key={b.bin} cx={at(b.mean_predicted)} cy={size - at(b.observed_rate)} r={4.5} fill="var(--color-navy)" />
        ))}
      </svg>
      <table className="w-full border-collapse self-start text-sm">
        <caption className="sr-only">Predicted and observed rates by bin of predicted probability</caption>
        <thead>
          <tr>
            <th className={th}>Bin</th>
            <th className={`${th} text-right`}>Predicted</th>
            <th className={`${th} text-right`}>Observed</th>
          </tr>
        </thead>
        <tbody>
          {bins.map((b) => (
            <tr key={b.bin}>
              <td className={td}>{b.bin}</td>
              <td className={`${td} text-right`}>{fmtPct(b.mean_predicted)}</td>
              <td className={`${td} text-right`}>{fmtPct(b.observed_rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
