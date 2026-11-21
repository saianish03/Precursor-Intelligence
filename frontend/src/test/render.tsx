import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render } from "@testing-library/react"
import { createMemoryRouter, RouterProvider } from "react-router"
import { AuthGate } from "@/app/auth"
import { routes } from "@/app/routes"

/** The whole app at one URL, answered by the recorded fixtures (VITE_FIXTURES is set for tests). */
export function renderAt(path: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <AuthGate>
        <RouterProvider router={createMemoryRouter(routes, { initialEntries: [path] })} />
      </AuthGate>
    </QueryClientProvider>,
  )
}
