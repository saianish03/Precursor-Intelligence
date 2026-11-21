/**
 * Recorded API responses for development and tests, answered in place of the network (npm run dev:fixtures).
 * The JSON files are real data recorded by scripts/record_ui_fixtures.py; this file only routes requests to them,
 * and filters and pages the list the way the API contract (src/api/openapi.yaml) describes.
 */
import facilities from "./facilities.json"
import ops from "./ops.json"
import overview from "./overview.json"
import status from "./status.json"

const profiles = import.meta.glob<{ default: unknown }>("./facility/*.json", { eager: true })

const dataAsOf = overview.data_as_of
type Row = (typeof facilities.items)[number]

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

function error(status: number, code: string, message: string): Response {
  return json({ error: { code, message, request_id: "fixture", retry_after_s: null } }, status)
}

export function listFacilities(params: URLSearchParams) {
  const q = (params.get("q") ?? "").trim().toLowerCase()
  const state = params.get("state")
  const decile = params.get("decile")
  const offset = Number(params.get("offset") ?? 0)
  const limit = Math.min(Number(params.get("limit") ?? 50), 100)
  const sort = params.get("sort") ?? "p_desc"
  let rows: Row[] = facilities.items.filter(
    (r) =>
      (!q || r.ccn.toLowerCase() === q || [r.name, r.city].some((s) => s?.toLowerCase().includes(q))) &&
      (!state || r.state === state) &&
      (!decile || r.risk_decile === Number(decile)),
  )
  if (sort === "p_asc") rows = [...rows].reverse()
  if (sort === "name_asc") rows = [...rows].sort((a, b) => (a.name ?? "").localeCompare(b.name ?? ""))
  return { items: rows.slice(offset, offset + limit), total: rows.length, offset, limit, data_as_of: dataAsOf }
}

export async function fixtureFetch(input: RequestInfo | URL): Promise<Response> {
  const url = new URL(input instanceof Request ? input.url : String(input), "http://fixtures")
  const path = url.pathname
  if (path === "/api/v1/status") return json(status)
  if (path === "/api/v1/overview") return json(overview)
  if (path === "/api/v1/ops") return json(ops)
  if (path === "/api/v1/facilities") return json(listFacilities(url.searchParams))
  const ccn = path.match(/^\/api\/v1\/facilities\/([0-9A-Z]{6})$/)?.[1]
  if (ccn) {
    const profile = profiles[`./facility/${ccn}.json`]
    return profile ? json(profile.default) : error(404, "not_found", `No facility profile for CCN ${ccn} in the fixtures.`)
  }
  return error(404, "not_found", `No fixture for ${path}.`)
}
