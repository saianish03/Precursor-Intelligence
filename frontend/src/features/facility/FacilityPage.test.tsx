import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import profile from "../../../fixtures/facility/145584.json"
import { renderAt } from "@/test/render"

describe("facility profile on recorded data", () => {
  it("shows the facility, its probability and its standing", async () => {
    renderAt("/facilities/145584")
    expect(await screen.findByRole("heading", { level: 1, name: "Palm Garden of Mattoon" })).toBeInTheDocument()
    expect(screen.getByText("0.910")).toBeInTheDocument()
    expect(screen.getAllByText("Riskiest tenth").length).toBeGreaterThan(0)
    expect(screen.getByRole("region", { name: "Chain: Pointe Management" })).toBeInTheDocument()
  })

  it("marks a survey's citation when its mark on the strip is selected, until cleared", async () => {
    const user = userEvent.setup()
    renderAt("/facilities/145584")
    const first = profile.citations[0]!
    const strip = await screen.findAllByRole("button", { name: /citation/ })
    const mark = strip.find((b) => b.getAttribute("aria-label")?.startsWith("Aug 20, 2026"))!
    await user.click(mark)
    const row = document.getElementById(`rec-${first.evidence_id.replace(/[^A-Za-z0-9_-]/g, "_")}`)!
    expect(row).toHaveAttribute("data-marked", "true")
    expect(within(row).getByText(first.ftag)).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Clear" }))
    expect(row).not.toHaveAttribute("data-marked")
  })

  it("marks the chain's record from its chip", async () => {
    const user = userEvent.setup()
    renderAt("/facilities/145584")
    await user.click(await screen.findByRole("button", { name: "chain:776:2026-10" }))
    expect(document.getElementById("rec-chain_776_2026-10")).toHaveAttribute("data-marked", "true")
  })

  it("filters the citations to G to L", async () => {
    const user = userEvent.setup()
    renderAt("/facilities/145584")
    await user.click(await screen.findByLabelText("Only G to L citations"))
    const table = screen.getByRole("table", { name: /Citations in the last 36 months/ })
    const letters = within(table).getAllByTitle(/Harm or immediate jeopardy|No actual harm/).map((e) => e.textContent)
    expect(letters.every((l) => l! >= "G" && l! <= "L")).toBe(true)
  })

  it("says plainly when a CCN isn't scored this month", async () => {
    renderAt("/facilities/ZZZZZZ")
    expect(await screen.findByText("No scored facility with CCN ZZZZZZ this month")).toBeInTheDocument()
  })

  it("rejects a malformed CCN without calling the API", async () => {
    renderAt("/facilities/abc")
    expect(await screen.findByText(`"abc" isn't a CCN`)).toBeInTheDocument()
  })
})
