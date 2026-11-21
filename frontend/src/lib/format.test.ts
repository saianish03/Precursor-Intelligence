import { decileLabel, displayName, fmtDate, isHarm } from "./format"

describe("format", () => {
  it("title-cases names CMS stores in capitals and keeps mixed-case names", () => {
    expect(displayName("PALM GARDEN OF MATTOON")).toBe("Palm Garden of Mattoon")
    expect(displayName("LANDMARK OF RICHTON PARK REHAB & NSG CTR")).toBe("Landmark of Richton Park Rehab & NSG CTR")
    expect(displayName("AUSTIN OASIS, THE")).toBe("Austin Oasis, The")
    expect(displayName("ALDEN LAKELAND REHAB & HCC")).toBe("Alden Lakeland Rehab & HCC")
    expect(displayName("Nexus at Alton")).toBe("Nexus at Alton")
    expect(displayName(null)).toBe("")
  })
  it("reads data dates as calendar days in any time zone", () => {
    expect(fmtDate("2026-08-20")).toBe("Aug 20, 2026")
  })
  it("names tenths and harm levels", () => {
    expect([decileLabel(1), decileLabel(2), decileLabel(3), decileLabel(10)]).toEqual([
      "Riskiest tenth",
      "2nd tenth",
      "3rd tenth",
      "Least risky tenth",
    ])
    expect(["F", "G", "L"].map(isHarm)).toEqual([false, true, true])
  })
})
