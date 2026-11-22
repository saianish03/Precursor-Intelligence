import { ApiError, call, setSession } from "./client"

const response = (status: number, body?: unknown) =>
  new Response(body ? JSON.stringify(body) : null, { status, headers: { "Content-Type": "application/json" } })

describe("call", () => {
  it("returns the data of a successful answer", async () => {
    await expect(call(Promise.resolve({ data: { ok: 1 }, response: response(200) }))).resolves.toEqual({ ok: 1 })
  })
  it("keeps the backend's message, code and request id", async () => {
    const error = { error: { code: "over_cap", message: "Today's brief limit is reached.", request_id: "r1", retry_after_s: 60 } }
    const e = (await call(Promise.resolve({ error, response: response(429) })).catch((x: unknown) => x)) as ApiError
    expect(e).toBeInstanceOf(ApiError)
    expect([e.status, e.code, e.message, e.requestId, e.retryAfterS]).toEqual([429, "over_cap", "Today's brief limit is reached.", "r1", 60])
  })
  it("signs the user out on a 401", async () => {
    const unauthorized = vi.fn()
    setSession("token", unauthorized)
    await call(Promise.resolve({ error: {}, response: response(401) })).catch(() => {})
    expect(unauthorized).toHaveBeenCalledOnce()
    setSession(null, () => {})
  })
  it("reports a request that never got an answer as status 0", async () => {
    const e = (await call(Promise.reject(new TypeError("Failed to fetch"))).catch((x: unknown) => x)) as ApiError
    expect([e.status, e.message]).toEqual([0, "The server could not be reached. Check your connection and try again."])
  })
})
