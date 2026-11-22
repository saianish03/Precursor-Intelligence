import { createBrowserRouter, isRouteErrorResponse, Link, useRouteError } from "react-router"
import { Empty } from "@/components/states"
import { FacilitiesPage } from "@/features/facilities/FacilitiesPage"
import { FacilityPage } from "@/features/facility/FacilityPage"
import { ModelPage } from "@/features/model/ModelPage"
import { OverviewPage } from "@/features/overview/OverviewPage"
import { Shell } from "./Shell"

export const routes = [
  {
    element: <Shell />,
    errorElement: <RouteError />,
    children: [
      { path: "/", element: <OverviewPage /> },
      { path: "/facilities", element: <FacilitiesPage /> },
      { path: "/facilities/:ccn", element: <FacilityPage /> },
      { path: "/model", element: <ModelPage /> },
      { path: "*", element: <NotFound /> },
    ],
  },
]

export const router = () => createBrowserRouter(routes)

function NotFound() {
  return (
    <Empty title="There's no page here">
      <Link to="/" className="text-link underline">
        Go to this month
      </Link>
    </Empty>
  )
}

function RouteError() {
  const error = useRouteError()
  return (
    <main className="p-8">
      <Empty title="This page failed to load">
        {isRouteErrorResponse(error) ? `${error.status} ${error.statusText}` : "An unexpected error stopped the page."}{" "}
        <a href="/" className="text-link underline">
          Reload the app
        </a>
      </Empty>
    </main>
  )
}
