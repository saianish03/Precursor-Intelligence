# ADR 0001: UI hosting, sign-in and the API path

**Status:** proposed, 2026-11-21 (Nishanth, frontend owner). Part M1 of `docs/AGENTS_PLAN.md`.

## Context
The React app (`frontend/`) and the API (`backend/`, FastAPI with the agents) both run on GCP. The agents plan
recommends one Cloud Run service and Google sign-in (§1 items 7 and 8, decision 4). Three things had to be settled
before the app could be built: where it is served, how users sign in, and where the API's paths live.

## Decision
1. **One Cloud Run service, `precursor-api`.** The image's Node stage runs `npm ci` and `npm run build`; FastAPI
   serves `frontend/dist/`. Same origin, so no CORS, one deploy and one rollback for the app and the API together.
   - FastAPI answers every path outside `/api/` with `index.html`, so the app's own routes (`/facilities/145584`)
     load on a refresh or a shared link.
   - Hashed files under `/assets/` are cached for a year (`immutable`); `index.html` is `no-cache`, so a deploy
     reaches users on their next load.
2. **The API lives under `/api/v1/`.** The plan's paths (`/facilities/{ccn}`, `/status`) would collide with the
   app's routes on one origin. `/api/v1/health` stays open; every other path needs a signed-in user.
3. **Google sign-in, checked in FastAPI.** The app loads Google Identity Services, gets an ID token, keeps it in
   memory only (never in `localStorage`), and sends it as `Authorization: Bearer` on every call. FastAPI verifies it
   (`google.oauth2.id_token`) against the OAuth client ID and an email allowlist in Secret Manager. A 401 shows the
   sign-in screen again; a 403 shows "this account doesn't have access". The client ID is not a secret; it is a
   build variable, `VITE_GOOGLE_CLIENT_ID`.
   - The content security policy has to allow `https://accounts.google.com/gsi/client` (script and frame).
4. **The contract comes first.** `frontend/src/api/openapi.yaml` proposes the read endpoints the app needs and the
   error body every endpoint returns. The app's types are generated from it (`npm run gen:api`); once FastAPI exists,
   the same command reads FastAPI's `/openapi.json` instead, and a contract test keeps the two equal. The proposed
   endpoints beyond the plan's K6 are:
   - `GET /api/v1/overview`: counts for the month, from the evidence files (gap G2);
   - `GET /api/v1/facilities`: search and paging, with facility names (G1, G2);
   - `GET /api/v1/ops`: the ops evidence, with no LLM (G5).
   They need a `facilities` table in `build-evidence` from `gold/dim_facility_latest` (names, addresses, beds;
   display only, never a feature).
5. **Fixtures are recorded, never written by hand.** `scripts/record_ui_fixtures.py` records the API responses
   from one real scored month; agent responses will come from the agents' fallback templates (B9). The fixture
   server loads only with `VITE_FIXTURES=true` (`npm run dev:fixtures` and the tests), so it never ships.

## Alternatives not taken
- **A separate `precursor-ui` service (nginx) with CORS:** two deploys that must move together, and a CORS policy
  to maintain, for no gain at this scale.
- **IAP in front of Cloud Run:** no auth code, but it needs an OAuth brand on the project, and we haven't confirmed
  it works on a project with no organization. Google sign-in in FastAPI works either way.

## Consequences
- The API owners (K1, K6) build to `/api/v1/` and to the contract's error body: `{error: {code, message,
  request_id, retry_after_s}}`, where `message` is safe to show and no stack trace ever leaves the API.
- Agent endpoints (K3–K5) join the contract once their output models (A5, F1, G3, H1, I2) exist.
- The image needs the Node stage (L7); CI runs the frontend job (lint, typecheck, tests, build) on every PR that
  touches `frontend/`.
