// Sign-in with fixture mode off: auth.tsx reads the env when it loads, so each test loads a fresh copy of the app
// (and of Testing Library, whose automatic cleanup only knows its own copy).
let cleanup = () => {}
async function renderGate(env: Record<string, string>) {
  vi.stubEnv("VITE_FIXTURES", "false")
  for (const [k, v] of Object.entries(env)) vi.stubEnv(k, v)
  vi.resetModules()
  const [{ render, screen, cleanup: c }, { QueryClient, QueryClientProvider }, { AuthGate }] = await Promise.all([
    import("@testing-library/react"),
    import("@tanstack/react-query"),
    import("./auth"),
  ])
  cleanup = c
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <AuthGate>
        <p>the app</p>
      </AuthGate>
    </QueryClientProvider>,
  )
  return screen
}

afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  delete window.google
  document.head.querySelectorAll("script").forEach((s) => s.remove())
})

describe("sign-in", () => {
  it("shows a signed-out user only the sign-in screen", async () => {
    const screen = await renderGate({})
    expect(await screen.findByRole("heading", { name: "Sign in" })).toBeInTheDocument()
    expect(screen.getByRole("alert")).toHaveTextContent("no client ID is configured")
    expect(screen.queryByText("the app")).not.toBeInTheDocument()
  })

  it("shows its own screen when the API refuses the account (403)", async () => {
    const body = { error: { code: "forbidden", message: "This account isn't on the allowlist.", request_id: "r1", retry_after_s: null } }
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(body), { status: 403, headers: { "Content-Type": "application/json" } })))
    let signedIn: (r: { credential: string }) => void = () => {}
    window.google = {
      accounts: {
        id: {
          initialize: (o) => (signedIn = o.callback),
          renderButton: () => {},
          prompt: () => {},
          disableAutoSelect: () => {},
        },
      },
    }
    const screen = await renderGate({ VITE_GOOGLE_CLIENT_ID: "test-client" })
    await screen.findByRole("heading", { name: "Sign in" })
    const { act } = await import("@testing-library/react")
    const payload = btoa(JSON.stringify({ email: "someone@example.com" }))
    await act(async () => {
      document.head.querySelector("script")!.dispatchEvent(new Event("load"))
      signedIn({ credential: `h.${payload}.s` })
    })
    expect(await screen.findByRole("heading", { name: "This account doesn't have access" })).toBeInTheDocument()
    expect(screen.getByText(/This account isn't on the allowlist\./)).toBeInTheDocument()
    expect(screen.queryByText("the app")).not.toBeInTheDocument()
  })
})
