import type { SearchMatch, SearchResponse } from '../types'
import { getContentId } from '../types'
import MatchCard from './MatchCard'

interface Props {
  label: string
  apiColor: 'indigo' | 'emerald'
  response: SearchResponse | null
  error: string | null
  isLoading: boolean
  locale: string
  overlapIds: Map<string, number> // contentId → rank in OTHER list
}

export default function ResultsList({ label, apiColor, response, error, isLoading, locale, overlapIds }: Props) {
  const headerBg = apiColor === 'indigo' ? 'bg-indigo-600' : 'bg-emerald-600'
  const spinnerColor = apiColor === 'indigo' ? 'border-indigo-500' : 'border-emerald-500'

  return (
    <div className="flex flex-col gap-3">
      {/* Header */}
      <div className={`${headerBg} text-white rounded-xl px-5 py-3 flex items-center justify-between`}>
        <div>
          <div className="font-bold text-base">{label}</div>
          {response && (
            <div className="text-xs opacity-80 mt-0.5">
              {response.totalPrimary} primary
              {response.totalFallback > 0 && ` · ${response.totalFallback} fallback`}
            </div>
          )}
        </div>
        {response && (
          <div className="text-right">
            <div className="text-xl font-bold">{response.timing.totalMs}ms</div>
            <div className="text-xs opacity-70">total</div>
          </div>
        )}
      </div>

      {/* Body */}
      {isLoading && (
        <div className="flex items-center justify-center py-16">
          <div className={`w-8 h-8 border-4 border-gray-200 ${spinnerColor} border-t-transparent rounded-full animate-spin`} />
        </div>
      )}

      {error && !isLoading && (
        <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-sm text-red-700">
          <strong>Error:</strong> {error}
        </div>
      )}

      {response && !isLoading && (
        <div className="space-y-2.5">
          {response.matches.length === 0 ? (
            <div className="text-sm text-gray-400 text-center py-10">No results</div>
          ) : (
            response.matches.map((match: SearchMatch, idx: number) => {
              const contentId = getContentId(match.metadata)
              const isOverlap = overlapIds.has(contentId)
              const overlapRank = overlapIds.get(contentId)
              return (
                <MatchCard
                  key={match.id}
                  match={match}
                  rank={idx + 1}
                  locale={locale}
                  isOverlap={isOverlap}
                  overlapRank={overlapRank}
                  apiColor={apiColor}
                />
              )
            })
          )}
        </div>
      )}
    </div>
  )
}
