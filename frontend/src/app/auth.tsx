import { useQueryClient } from "@tanstack/react-query"
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react"
import { ApiError, setSession } from "@/api/client"
import { useStatus } from "@/api/queries"
import { Wordmark } from "./Wordmark"

/*
 * Google sign-in (docs/adr/0001-ui-hosting-and-auth.md): Google Identity Services gives an ID token, kept in memory
 * only and sent on every API call; the API checks it and the email allowlist. A 401 drops the token and shows
 * sign-in again; a 403 means the account isn't on the allowlist. Fixture mode skips sign-in: no API is involved.
 */
type GoogleId = {
  initialize(o: { client_id: string; callback: (r: { credential: string }) => void; auto_select?: boolean }): void
  renderButton(el: HTMLElement, o: Record<string, unknown>): void
  prompt(): void
  disableAutoSelect(): void
}
declare global {
  interface Window {
    google?: { accounts: { id: GoogleId } }
  }
}

const FIXTURES = import.meta.env.VITE_FIXTURES === "true"
const CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID as string | undefined

type Session = { email: string | null; fixtures: boolean; signOut: () => void }
const SessionContext = createContext<Session>({ email: null, fixtures: FIXTURES, signOut: () => {} })
export const useSession = () => useContext(SessionContext)

function emailOf(token: string): string | null {
  try {
    const payload = JSON.parse(atob(token.split(".")[1]!.replace(/-/g, "+").replace(/_/g, "/")))
    return typeof payload.email === "string" ? payload.email : null
  } catch {
    return null
  }
}

export function AuthGate({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(null)
  const queryClient = useQueryClient()
  const signOut = useCallback(() => {
    window.google?.accounts.id.disableAutoSelect()
    setSession(null, () => {})
    setToken(null)
    queryClient.clear()
  }, [queryClient])
  const onToken = useCallback(
    (t: string) => {
      setSession(t, signOut)
      setToken(t)
    },
    [signOut],
  )

  if (FIXTURES) {
    return (
      <SessionContext.Provider value={{ email: null, fixtures: true, signOut: () => {} }}>{children}</SessionContext.Provider>
    )
  }
  if (!token) {
    return <SignIn onToken={onToken} />
  }
  return (
    <SessionContext.Provider value={{ email: emailOf(token), fixtures: false, signOut }}>
      <AllowlistCheck signOut={signOut}>{children}</AllowlistCheck>
    </SessionContext.Provider>
  )
}

/** The first API call proves the token and the allowlist; a 403 gets its own screen instead of an error panel. */
function AllowlistCheck({ children, signOut }: { children: ReactNode; signOut: () => void }) {
  const status = useStatus()
  if (status.error instanceof ApiError && status.error.status === 403) {
    return (
      <Screen>
        <h1 className="text-xl font-bold">This account doesn't have access</h1>
        <p className="mt-2 text-muted">
          {status.error.message} Ask the Precursor team to add your email to the allowlist, then sign in again.
        </p>
        <button type="button" onClick={signOut} className="mt-5 rounded border border-rule bg-paper px-3 py-2 font-semibold">
          Sign in with another account
        </button>
      </Screen>
    )
  }
  return children
}

function SignIn({ onToken }: { onToken: (token: string) => void }) {
  const button = useRef<HTMLDivElement>(null)
  const [failed, setFailed] = useState(!CLIENT_ID)
  useEffect(() => {
    if (!CLIENT_ID) return
    const script = document.createElement("script")
    script.src = "https://accounts.google.com/gsi/client"
    script.async = true
    script.onload = () => {
      const id = window.google?.accounts.id
      if (!id || !button.current) return setFailed(true)
      id.initialize({ client_id: CLIENT_ID, callback: (r) => onToken(r.credential), auto_select: true })
      id.renderButton(button.current, { theme: "outline", size: "large", text: "signin_with", width: 280 })
      id.prompt()
    }
    script.onerror = () => setFailed(true)
    document.head.append(script)
    return () => script.remove()
  }, [onToken])
  return (
    <Screen>
      <h1 className="text-xl font-bold">Sign in</h1>
      <p className="mt-2 text-muted">
        Monthly risk of a serious CMS health citation for every U.S. nursing home, with the records behind each score.
        Access is limited to approved accounts.
      </p>
      <div ref={button} className="mt-6 min-h-11" />
      {failed && (
        <p role="alert" className="mt-4 text-sm text-attn">
          Google sign-in couldn't load{CLIENT_ID ? "" : " (no client ID is configured for this build)"}. Reload the page
          to try again.
        </p>
      )}
    </Screen>
  )
}

function Screen({ children }: { children: ReactNode }) {
  return (
    <main className="grid min-h-dvh grid-rows-[auto_1fr] bg-ground md:grid-cols-[232px_1fr] md:grid-rows-1">
      <div className="bg-navy px-5 py-5 text-paper">
        <Wordmark />
      </div>
      <div className="flex items-center px-6 py-12 md:px-16">
        <div className="w-full max-w-md">{children}</div>
      </div>
    </main>
  )
}
