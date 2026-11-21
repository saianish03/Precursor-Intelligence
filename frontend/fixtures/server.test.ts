import facilities from "./facilities.json"
import { fixtureFetch, listFacilities } from "./server"

const list = (q: Record<string, string>) => listFacilities(new URLSearchParams(q))

describe("the fixture list follows the API contract", () => {
  it("pages the riskiest first", () => {
    const page = list({ limit: "20", offset: "20" })
    expect(page.items).toHaveLength(20)
    expect(page.items[0]!.ccn).toBe(facilities.items[20]!.ccn)
    expect(page.total).toBe(facilities.items.length)
  })
  it("finds by exact CCN, by name or city, and filters by state and tenth", () => {
    expect(list({ q: "145584" }).items.map((r) => r.ccn)).toEqual(["145584"])
    expect(list({ q: "mattoon" }).items.every((r) => /mattoon/i.test(`${r.name} ${r.city}`))).toBe(true)
    const il = list({ state: "IL", decile: "1" })
    expect(il.total).toBeGreaterThan(0)
    expect(il.items.every((r) => r.state === "IL" && r.risk_decile === 1)).toBe(true)
  })
  it("caps the page size at 100", () => {
    expect(list({ limit: "500" }).items).toHaveLength(100)
  })
  it("answers an unknown facility with the API's 404 error body", async () => {
    const res = await fixtureFetch("/api/v1/facilities/ZZZZZZ")
    expect(res.status).toBe(404)
    expect((await res.json()).error.code).toBe("not_found")
  })
})
