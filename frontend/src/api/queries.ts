import { keepPreviousData, useQuery } from "@tanstack/react-query"
import { api, ApiError, call } from "./client"

// The data changes once a month, so a fetched answer stays fresh for the session.
const monthly = { staleTime: 30 * 60 * 1000 }
const retry = (count: number, error: unknown) =>
  count < 2 && !(error instanceof ApiError && error.status >= 400 && error.status < 500)

export const useStatus = () =>
  useQuery({ queryKey: ["status"], queryFn: () => call(api.GET("/api/v1/status")), ...monthly, retry })

export const useOverview = () =>
  useQuery({ queryKey: ["overview"], queryFn: () => call(api.GET("/api/v1/overview")), ...monthly, retry })

export type FacilityQuery = {
  q?: string
  state?: string
  decile?: number
  sort?: "p_desc" | "p_asc" | "name_asc"
  offset?: number
  limit?: number
}

export const useFacilities = (query: FacilityQuery) =>
  useQuery({
    queryKey: ["facilities", query],
    queryFn: () => call(api.GET("/api/v1/facilities", { params: { query } })),
    placeholderData: keepPreviousData,
    ...monthly,
    retry,
  })

export const useFacility = (ccn: string, enabled = true) =>
  useQuery({
    queryKey: ["facility", ccn],
    queryFn: () => call(api.GET("/api/v1/facilities/{ccn}", { params: { path: { ccn } } })),
    enabled,
    ...monthly,
    retry,
  })

export const useOps = () =>
  useQuery({ queryKey: ["ops"], queryFn: () => call(api.GET("/api/v1/ops")), ...monthly, retry })
