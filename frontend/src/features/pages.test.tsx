import { screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { renderAt } from "@/test/render"

describe("pages on recorded data", () => {
  it("this month lists the riskiest facilities and the month and model they come from", async () => {
    renderAt("/")
    expect(await screen.findByRole("heading", { level: 1, name: "October 2026" })).toBeInTheDocument()
    expect(await screen.findByRole("link", { name: "Palm Garden of Mattoon" })).toBeInTheDocument()
    expect(screen.getByText("Gold data not verified")).toBeInTheDocument()
  })

  it("facilities search shows an empty state when nothing matches", async () => {
    const user = userEvent.setup()
    renderAt("/facilities")
    await user.type(await screen.findByLabelText("Name, city or CCN"), "no such facility anywhere{Enter}")
    expect(await screen.findByText("No facilities match")).toBeInTheDocument()
  })

  it("model health reports the test-month PR-AUC and says gate results aren't published yet", async () => {
    renderAt("/model")
    expect(await screen.findByText(/^0\.470 \(95% CI 0\.454 to 0\.486\)$/)).toBeInTheDocument()
    expect(screen.getByText("Gate results aren't published yet")).toBeInTheDocument()
    expect(screen.getByText("staffing_rating_fn", { exact: false })).toBeInTheDocument()
  })
})
