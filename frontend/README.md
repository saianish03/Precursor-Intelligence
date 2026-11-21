# frontend/: the Precursor Intelligence workbench (React)

The app underwriters use: this month's scores, facility search, each facility's score with its drivers, citations
and penalties, and the model's health. Agent briefs join next. It is served by the API's Cloud Run service from the
built `dist/` (`docs/adr/0001-ui-hosting-and-auth.md`).

**Stack:** React 19, Vite, TypeScript, Tailwind CSS 4, TanStack Query, React Router, a typed client generated from the
API contract (`openapi-fetch`), Vitest and Testing Library. Fonts are self-hosted (Public Sans, Merriweather, Roboto
Mono).

## Run it
```bash
cd frontend
npm ci
npm run dev:fixtures          # recorded real data, no API and no sign-in: http://localhost:5173
API_ORIGIN=http://localhost:8000 VITE_GOOGLE_CLIENT_ID=<client id> npm run dev   # against a local API
```

| Command | What it does |
|---|---|
| `npm run lint` / `npm run typecheck` | ESLint; TypeScript with no emit |
| `npm test` | Component and unit tests, on the recorded fixtures |
| `npm run build` | Typecheck, then the production build in `dist/` |
| `npm run gen:api` | Regenerate `src/api/schema.d.ts` from `src/api/openapi.yaml` |

## Layout
```
src/
  app/          shell (rail, freshness line), routes, Google sign-in
  api/          the contract (openapi.yaml), its generated types, the client and the query hooks
  components/   states (loading, empty, error), the evidence mark, shared table and scale pieces
  features/     overview, facilities, facility (profile), model (health)
  lib/          formatting: dates as calendar days, probabilities, names
  styles/       design tokens (Tailwind theme) and base styles
fixtures/       recorded API responses and the fixture server (dev and tests only)
```

## Fixtures
Real data, recorded by `scripts/record_ui_fixtures.py` from one scored month in GCS; never edit them by hand:
```bash
uv run --extra pipeline python scripts/record_ui_fixtures.py \
  --root precursor-intel-mlops-data/ml-dev --v 2026-10 --eval-run phase1-retrain-1
```
The list holds the 500 riskiest facilities, and five facilities have full profiles (the three riskiest, one in the
5th tenth and one in the least risky tenth). Gate results are `null` until `build-evidence` publishes them.

## Design
The product brief is `PRODUCT.md` at the repo root; the visual direction ("The Register": every claim ends in a
citation that resolves to its CMS record) is in `.impeccable/surfaces/frontend-src.md`. Colours, type and sizes are
tokens in `src/styles/index.css`; use them rather than raw values.
- **Probabilities are model estimates.** Never write that a facility "will be cited".
- **Every score shows its month and model.**
- **Colour is never the only signal:** severity and the tenth also appear as text.
