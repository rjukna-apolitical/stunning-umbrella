import type { SearchMatch } from '../types'
import { getTitle, getSnippet, getContentType, getLocale } from '../types'

const CONTENT_TYPE_COLORS: Record<string, string> = {
  solutionArticle: 'bg-blue-100 text-blue-700',
  event: 'bg-purple-100 text-purple-700',
  course: 'bg-amber-100 text-amber-700',
}

interface Props {
  match: SearchMatch
  rank: number
  locale: string
  isOverlap: boolean
  overlapRank?: number
  apiColor: 'indigo' | 'emerald'
}

export default function MatchCard({ match, rank, locale, isOverlap, overlapRank, apiColor }: Props) {
  const title = getTitle(match.metadata, locale)
  const snippet = getSnippet(match.metadata)
  const contentType = getContentType(match.metadata)
  const resultLocale = getLocale(match.metadata)
  const isBoosted = match.score !== match.originalScore

  const borderColor = apiColor === 'indigo' ? 'border-l-indigo-400' : 'border-l-emerald-400'
  const rankColor = apiColor === 'indigo' ? 'bg-indigo-100 text-indigo-700' : 'bg-emerald-100 text-emerald-700'

  return (
    <div className={`bg-white border border-gray-200 rounded-lg p-4 border-l-4 ${borderColor} hover:shadow-sm transition-shadow`}>
      {/* Top row: rank + badges */}
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`text-xs font-bold w-6 h-6 rounded-full flex items-center justify-center shrink-0 ${rankColor}`}>
            {rank}
          </span>

          {contentType && (
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${CONTENT_TYPE_COLORS[contentType] ?? 'bg-gray-100 text-gray-600'}`}>
              {contentType}
            </span>
          )}

          {resultLocale && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 font-mono">
              {resultLocale}
            </span>
          )}

          {match.isFallback && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-yellow-100 text-yellow-700 font-medium">
              fallback
            </span>
          )}

          {isOverlap && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-pink-100 text-pink-700 font-medium" title={`Also rank #${overlapRank} in other API`}>
              ≈ #{overlapRank} in {apiColor === 'indigo' ? 'B' : 'A'}
            </span>
          )}
        </div>

        {/* Score */}
        <div className="text-right shrink-0">
          <div className="text-sm font-bold text-gray-800">{match.score.toFixed(4)}</div>
          {isBoosted && (
            <div className="text-xs text-gray-400 line-through">{match.originalScore.toFixed(4)}</div>
          )}
        </div>
      </div>

      {/* Title */}
      <p className="text-sm font-semibold text-gray-900 leading-snug mb-1.5">{title}</p>

      {/* Snippet */}
      {snippet && (
        <p className="text-xs text-gray-500 leading-relaxed line-clamp-3">{snippet}</p>
      )}

      {/* Vector ID */}
      <p className="mt-2 text-[10px] text-gray-300 font-mono truncate" title={match.id}>{match.id}</p>
    </div>
  )
}
