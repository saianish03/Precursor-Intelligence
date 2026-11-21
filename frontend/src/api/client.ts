import createClient from "openapi-fetch"
import type { components, paths } from "./schema"

export type Schemas = components["schemas"]

/** An API failure with the backend's safe message; `status` 0 means the request never got an answer. */
export class ApiError extends Error {
  readonly status: number
  readonly code: string
  readonly requestId: string | null
  readonly retryAfterS: number | null
  constructor(status: number, body: Partial<Schemas["ErrorBody"]> | undefined, fallback: string) {
    super(body?.error?.message ?? fallback)
    this.status = status
    this.code = body?.error?.code ?? "unknown"
    this.requestId = body?.error?.request_id ?? null
    this.retryAfterS = body?.error?.retry_after_s ?? null
  }
}

let idToken: string | null = null
let onUnauthorized: () => void = () => {}

export function setSession(token: string | null, unauthorized: () => void) {
  idToken = token
  onUnauthorized = unauthorized
}

// The fixture server is imported only when VITE_FIXTURES is set, so it never ships in a production bundle.
let fetchImpl: Promise<typeof fetch> | null = null
function getFetch(): Promise<typeof fetch> {
  fetchImpl ??=
    import.meta.env.VITE_FIXTURES === "true"
      ? import("../../fixtures/server").then((m) => m.fixtureFetch)
      : Promise.resolve(globalThis.fetch.bind(globalThis))
  return fetchImpl
}

// Same origin as the app (FastAPI serves both); an absolute base also lets tests build requests outside a browser.
export const api = createClient<paths>({
  baseUrl: globalThis.location?.origin ?? "",
  fetch: async (request: Request) => (await getFetch())(request),
})

api.use({
  onRequest({ request }) {
    if (idToken) request.headers.set("Authorization", `Bearer ${idToken}`)
    return request
  },
})

/** Unwrap an openapi-fetch result: the data, or an ApiError with the backend's message. */
export async function call<T>(
  pending: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> {
  let result: Awaited<typeof pending>
  try {
    result = await pending
  } catch {
    throw new ApiError(0, undefined, "The server could not be reached. Check your connection and try again.")
  }
  const { data, error, response } = result
  if (response.ok && data !== undefined) return data
  if (response.status === 401) onUnauthorized()
  throw new ApiError(
    response.status,
    error as Partial<Schemas["ErrorBody"]>,
    `The request failed (${response.status}).`,
  )
}
