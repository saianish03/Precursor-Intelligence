# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Delegated, then fixed by the team's agents plan (`docs/AGENTS_PLAN.md` part M): React, Vite, TypeScript, Tailwind CSS and
shadcn/ui in `frontend/`. The build (`dist/`) is served by the FastAPI backend from one Cloud Run service
(`precursor-api`), same origin, behind Google sign-in checked against an email allowlist. Node is build tooling only.

## Users

- **Primary: underwriters at long-term-care liability insurers**, quoting or renewing a nursing-home account or a
  book of facilities. They decide which facilities need attention, and what to ask the operator.
- **Secondary: risk-engineering and loss-control leads** choosing which facilities to visit and what to look at there.
- **Third: the Precursor platform team**, checking that this month's scores can be trusted (model health).

## Product Purpose

Every month, Precursor Intelligence gives each U.S. CMS-certified nursing home (about 14,700) a calibrated probability
of a serious health citation (scope and severity G to L) in the next 12 months, explains it with the features that
raise and lower it, and backs it with the facility's own citations and penalties. Four AI agents turn that evidence
into briefs: Facility Investigation, Regulatory Evidence Retriever, Model Ops Analyst and Portfolio Briefing.

Success: an underwriter can go from a book of facilities to the few that need attention, and say why, with every claim
traceable to a CMS record, in minutes.

## Positioning

Every claim points at its record. Scores, drivers, agent briefs and regulation quotes carry evidence IDs that resolve
to the actual CMS citation, penalty, score or regulation passage, and agent output is checked by validators (evidence
IDs, numbers, driver direction, quote fidelity, structure) before anyone sees it. It is not an analytics dashboard and
not a chatbot.

## Operating Context

- Monthly cadence: CMS publishes a release, the pipeline scores it, and the evidence files for the month are rebuilt.
  Data is monthly, never real time.
- Vocabulary users already know: CCN (6-character CMS id), F-tags, the scope and severity grid (A to L), standard and
  complaint surveys, civil money penalties, payment denials, chains, Special Focus Facilities.
- Agent inputs are fixed: Facility Investigation takes a CCN and a question type from a fixed list, never free text;
  Portfolio Briefing takes a list of CCNs; only the Regulatory Evidence Retriever takes a free-text question.
- An agent brief takes about 40-60 s live; the briefs for the riskiest facilities are pre-built and return at once.

## Capabilities and Constraints

- A probability is a model estimate, never a forecast that a citation will happen. Banned wording: "will be cited",
  "decline", "approve", "guarantee". Star ratings are never presented as model inputs.
- Every score and brief shows its data month (`data_as_of`), the model version and the caveat.
- Only one month is scored so far (2026-10), so there is no score history; history means the facility's last 36 months
  of citations and penalties.
- Risk bands are undecided: no band rule table exists yet, so the UI shows decile and percentile until one does.
- Agent answers can be a fallback (a fixed write-up from the tool data alone, marked as such); the daily call cap
  returns 429. Both are normal states, not errors.
- No agent conversations, sessions or memory exist on the server.

## Brand Commitments

The name "Precursor Intelligence" only. No logo, colour or typeface is committed.

## Evidence on Hand

- Real scores for 2026-10: `gs://precursor-intel-mlops-data/ml-dev/scores/v=2026-10/` (facility scores, reasons, SHAP,
  chain roll-ups; 14,686 facilities, model v1, `gold_verified: false`).
- Facility names and addresses: `gold/dim_facility_latest/`. Citations and penalties: silver, vintage 2026-10.
- Model results: README section 8 (champion PR-AUC 0.470, 95% CI 0.454-0.486; base rate 25.6%; ECE 0.028).
- No customers, testimonials, pricing or insurer logos exist; never invent them. UI fixtures are recorded from this
  real data, and agent fixtures come from the deterministic fallback templates.

## Product Principles

1. Evidence before interpretation: show the record, then the AI's reading of it, and keep the two visibly distinct.
2. Honest about time and certainty: always say which month and model, and never imply a trend or a certainty the data
   doesn't have.
3. Fixed questions beat open chat: the product asks the questions an underwriter needs answered, so answers are
   cacheable, checkable and consistent.
4. Degrade plainly: a fallback, a missing brief or a hit daily cap is shown as what it is, with what the user can do.

## Accessibility & Inclusion

WCAG 2.2 AA: 4.5:1 text contrast, full keyboard use with visible focus, screen-reader labels, a table alternative for
every chart, colour never the only signal (severity and risk also carry text), and reduced motion respected.
