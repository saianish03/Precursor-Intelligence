---
version: 1
slug: "frontend-src"
primary_target: "frontend/src"
related_targets: []
---

# Workbench (the whole React app)

Scope: every authenticated screen of `frontend/`: This month, Facilities, Facility profile, Portfolio, Regulations,
Model health, plus sign-in. Visitor mode: Operate.

Audience and job: underwriters at liability insurers (primary), risk engineers, the platform team. Task: find the
facilities in a book that need attention, understand why, and check the evidence and the model behind it.

Constraints: monthly data only (one scored month so far); probabilities are model estimates; every score and brief
shows its data month, model version and caveat; agent answers can be fallbacks or hit the daily cap (429); fixtures are
real-data recordings.

## Direction contract

THESIS: Every claim on screen ends in a citation that resolves to its CMS record, as on a Federal Register page. It
refuses the KPI-card dashboard and the chatbot pane.

OWN-WORLD: GPO navy rail (#14284B) with white type; cool off-white ground (#F6F7F5); white work surfaces; ink #16191D;
rules #C9CED6; citation blue #1A5DAB on a pale blue chip; oxblood #8A2B1F for risk and G to L severity; amber only for
states that need attention. Public Sans for UI and data with tabular figures, Merriweather for agent prose and quoted
regulation text, Roboto Mono only for identifiers. Hairline rules, 4px radius, no cards-as-structure.

STORY: The underwriter sees the month and model first, finds a facility or a book, reads the probability and what moves
it, checks the citations and penalties, then reads the brief, whose every sentence cites a record they can open.

FIRST VIEWPORT: Left navy rail (232px) with five sections. A white freshness line across the top: month, model, scored
date, and any data flag. Facility header: name at 24px bold, CCN, place, beds and chain on one line; the probability at
34px oxblood on the right with its plain-language label. Below, a 1fr evidence column (drivers in two columns, the
36-month citation strip, the citations table) and a 440px brief column with the question select and the primary action.

FORM: The Register, candidate 5 of 7 (the Federal Register and CFR page). seed:36dfce322e44b1d1

SIGNATURE: The citation rail. Citation chips in the brief resolve to their records: hovering or focusing one highlights
the record in the evidence column; activating it scrolls there and keeps the mark until dismissed. Raises: one 12-column
4px grid; rank by weight, case and rule with three type sizes; one identifier notation; fixed 36-month and decile scales
on every facility; a changed state holds its mark until seen; tabular figures everywhere.

## Unresolved

- Risk bands: no rule table yet, so show decile and percentile only.
- The question type list for Agent 1 (F3) isn't defined; the select shows what the API publishes.
