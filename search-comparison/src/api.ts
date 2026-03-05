import type { SearchParams, SearchResponse } from './types'

async function doSearch(baseUrl: string, params: SearchParams): Promise<SearchResponse> {
  const qs = new URLSearchParams({
    q: params.query,
    locale: params.locale,
    pageSize: String(params.pageSize),
  })
  if (params.contentType) qs.set('contentType', params.contentType)

  const res = await fetch(`${baseUrl}/search?${qs.toString()}`)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`HTTP ${res.status}: ${text}`)
  }
  return res.json() as Promise<SearchResponse>
}

export function searchApiA(params: SearchParams): Promise<SearchResponse> {
  return doSearch('/api-a', params)
}

export function searchApiB(params: SearchParams): Promise<SearchResponse> {
  return doSearch('/api-b', params)
}
