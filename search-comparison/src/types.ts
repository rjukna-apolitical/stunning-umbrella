// ── Shared ────────────────────────────────────────────────────────────────────

export interface SearchMatch {
  id: string
  score: number
  originalScore: number
  isEnrolled: boolean
  isFallback: boolean
  metadata: Record<string, unknown>
}

export interface SearchTiming {
  embedMs: number
  /** Approach A: sparseMs (Pinecone-hosted), Approach B: bm25Ms (in-process) */
  sparseMs?: number
  bm25Ms?: number
  pineconeMs: number
  totalMs: number
}

export interface SearchResponse {
  matches: SearchMatch[]
  totalPrimary: number
  totalFallback: number
  page: number
  pageSize: number
  timing: SearchTiming
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Extract a display title from either approach's metadata */
export function getTitle(meta: Record<string, unknown>, locale = 'en'): string {
  // Approach B: has a `title` field
  if (typeof meta['title'] === 'string' && meta['title']) return meta['title']
  // Approach A: has `title_{locale}` fields
  const norm = locale.toLowerCase().replace('-', '_')
  if (typeof meta[`title_${norm}`] === 'string') return meta[`title_${norm}`] as string
  if (typeof meta['title_en'] === 'string') return meta['title_en'] as string
  // Fallback: first string value that looks like a title
  for (const [k, v] of Object.entries(meta)) {
    if (k.startsWith('title_') && typeof v === 'string' && v) return v
  }
  return '(no title)'
}

/** Extract a display snippet from either approach's metadata */
export function getSnippet(meta: Record<string, unknown>): string {
  // Approach B: `snippet` field
  if (typeof meta['snippet'] === 'string') return meta['snippet']
  // Approach A: `body` field (truncate to 300 chars)
  if (typeof meta['body'] === 'string') return meta['body'].slice(0, 300)
  return ''
}

/** Extract the content ID used for overlap detection */
export function getContentId(meta: Record<string, unknown>): string {
  return (meta['content_id'] ?? meta['entry_id'] ?? '') as string
}

/** Extract the content type */
export function getContentType(meta: Record<string, unknown>): string {
  return (meta['content_type'] ?? meta['type'] ?? '') as string
}

/** Extract locale */
export function getLocale(meta: Record<string, unknown>): string {
  return (meta['locale'] ?? '') as string
}

/** Sparse timing label differs between approaches */
export function getSparseLabel(timing: SearchTiming): string {
  return timing.bm25Ms !== undefined ? 'bm25Ms' : 'sparseMs'
}

export function getSparseMs(timing: SearchTiming): number {
  return timing.bm25Ms ?? timing.sparseMs ?? 0
}

// ── Search params ─────────────────────────────────────────────────────────────

export interface SearchParams {
  query: string
  locale: string
  contentType: string
  pageSize: number
}
