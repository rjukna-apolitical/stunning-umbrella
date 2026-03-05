import { useState } from 'react'
import { searchApiA, searchApiB } from './api'
import type { SearchParams, SearchResponse } from './types'
import { getContentId } from './types'
import SearchForm from './components/SearchForm'
import TimingPanel from './components/TimingPanel'
import ResultsList from './components/ResultsList'

interface QueryState {
  params: SearchParams
  responseA: SearchResponse | null
  responseB: SearchResponse | null
  errorA: string | null
  errorB: string | null
}

export default function App() {
  const [isLoading, setIsLoading] = useState(false)
  const [state, setState] = useState<QueryState | null>(null)

  async function handleSearch(params: SearchParams) {
    setIsLoading(true)
    setState({ params, responseA: null, responseB: null, errorA: null, errorB: null })

    const [resultA, resultB] = await Promise.allSettled([
      searchApiA(params),
      searchApiB(params),
    ])

    setState({
      params,
      responseA: resultA.status === 'fulfilled' ? resultA.value : null,
      responseB: resultB.status === 'fulfilled' ? resultB.value : null,
      errorA: resultA.status === 'rejected' ? String(resultA.reason) : null,
      errorB: resultB.status === 'rejected' ? String(resultB.reason) : null,
    })
    setIsLoading(false)
  }

  // Build overlap maps: contentId → rank in the OTHER list
  const overlapForA = new Map<string, number>()
  const overlapForB = new Map<string, number>()

  if (state?.responseA && state?.responseB) {
    const idsA = new Map(
      state.responseA.matches.map((m, i) => [getContentId(m.metadata), i + 1])
    )
    const idsB = new Map(
      state.responseB.matches.map((m, i) => [getContentId(m.metadata), i + 1])
    )
    for (const [id, rank] of idsB) if (idsA.has(id)) overlapForA.set(id, rank)
    for (const [id, rank] of idsA) if (idsB.has(id)) overlapForB.set(id, rank)
  }

  const overlapCount = overlapForA.size

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-screen-xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-gray-900">Search Comparison</h1>
            <p className="text-xs text-gray-500 mt-0.5">
              <span className="text-indigo-600 font-medium">Approach A</span>
              {' '}(Pinecone sparse) vs{' '}
              <span className="text-emerald-600 font-medium">Approach B</span>
              {' '}(corpus BM25)
            </p>
          </div>
          <div className="flex items-center gap-3 text-xs text-gray-400">
            <ApiStatus label="API A" url="/api-a/search?q=ping" color="indigo" />
            <ApiStatus label="API B" url="/api-b/search?q=ping" color="emerald" />
          </div>
        </div>
      </header>

      <main className="max-w-screen-xl mx-auto px-6 py-8 space-y-6">
        {/* Search form */}
        <SearchForm onSearch={handleSearch} isLoading={isLoading} />

        {/* Timing + overlap summary */}
        {(state?.responseA || state?.responseB) && (
          <>
            <TimingPanel
              timingA={state?.responseA?.timing ?? null}
              timingB={state?.responseB?.timing ?? null}
            />

            <div className="bg-white rounded-xl border border-gray-200 px-5 py-3 flex items-center gap-3 text-sm">
              <span className="text-gray-500">Result overlap:</span>
              <span className="font-bold text-gray-900">{overlapCount}</span>
              <span className="text-gray-400">
                of {Math.max(state?.responseA?.matches.length ?? 0, state?.responseB?.matches.length ?? 0)} results appear in both APIs
              </span>
              {overlapCount > 0 && (
                <span className="ml-auto text-xs text-pink-600 font-medium">
                  Overlap results marked with ≈
                </span>
              )}
            </div>
          </>
        )}

        {/* Side-by-side results */}
        {state && (
          <div className="grid grid-cols-2 gap-6">
            <ResultsList
              label="Approach A — Pinecone sparse"
              apiColor="indigo"
              response={state.responseA}
              error={state.errorA}
              isLoading={isLoading}
              locale={state.params.locale}
              overlapIds={overlapForA}
            />
            <ResultsList
              label="Approach B — BM25"
              apiColor="emerald"
              response={state.responseB}
              error={state.errorB}
              isLoading={isLoading}
              locale={state.params.locale}
              overlapIds={overlapForB}
            />
          </div>
        )}

        {/* Empty state */}
        {!state && !isLoading && (
          <div className="text-center py-24 text-gray-400">
            <svg className="mx-auto mb-4 w-12 h-12 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z" />
            </svg>
            <p className="text-sm">Enter a query above to compare both search approaches side by side</p>
          </div>
        )}
      </main>
    </div>
  )
}

// Tiny component that pings an API and shows a green/red dot
function ApiStatus({ label, url, color }: { label: string; url: string; color: 'indigo' | 'emerald' }) {
  const [status, setStatus] = useState<'unknown' | 'up' | 'down'>('unknown')

  // Fire a quick health check once on mount
  useState(() => {
    fetch(url, { signal: AbortSignal.timeout(3000) })
      .then(() => setStatus('up'))
      .catch(() => setStatus('down'))
  })

  const dot =
    status === 'up' ? 'bg-green-400' :
    status === 'down' ? 'bg-red-400' :
    'bg-gray-300'

  const textColor = color === 'indigo' ? 'text-indigo-600' : 'text-emerald-600'

  return (
    <span className="flex items-center gap-1.5">
      <span className={`w-2 h-2 rounded-full ${dot}`} />
      <span className={`font-medium ${textColor}`}>{label}</span>
    </span>
  )
}
