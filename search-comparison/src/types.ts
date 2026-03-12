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
  // Approach A courses: stored as `journey_title`
  if (typeof meta['journey_title'] === 'string' && meta['journey_title']) return meta['journey_title']
  // Approach A articles/events: has `title_{locale}` fields
  const norm = locale.toLowerCase().replace(/-/g, '_')
  if (typeof meta[`title_${norm}`] === 'string') return meta[`title_${norm}`] as string
  if (typeof meta['title_en'] === 'string') return meta['title_en'] as string
  // Fallback: first title_ field found
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

/** Extract the content ID used for overlap detection.
 *
 * Approach B always has `content_id` (course/article/event entry ID).
 * Approach A articles/events use `entry_id` (same Contentful ID → matches B).
 * Approach A courses use `journey_id` as the vector key, but `course_id` holds
 * the parent course entry ID which equals B's `content_id` for that course.
 */
export function getContentId(meta: Record<string, unknown>): string {
  // Approach B
  if (typeof meta['content_id'] === 'string' && meta['content_id']) return meta['content_id']
  // Approach A courses: course_id matches B's content_id for the same course
  if (typeof meta['course_id'] === 'string' && meta['course_id']) return meta['course_id']
  // Approach A articles/events
  if (typeof meta['entry_id'] === 'string' && meta['entry_id']) return meta['entry_id']
  return ''
}

/** Extract the content type */
export function getContentType(meta: Record<string, unknown>): string {
  return (meta['content_type'] ?? meta['type'] ?? '') as string
}

/** Extract locale */
export function getLocale(meta: Record<string, unknown>): string {
  return (meta['locale'] ?? '') as string
}

const CONTENT_TYPE_SEGMENT: Record<string, string> = {
  solutionArticle: 'articles',
  event: 'events',
  course: 'courses',
}

/**
 * Build an apolitical.co page URL from a result's metadata.
 *
 * Approach B: `locale` and `slug` fields are directly available.
 * Approach A: `slug_{normalized_locale}` fields (e.g. slug_en, slug_fr_ca).
 *   Falls back to slug_en if the requested locale has no slug.
 *   Uses `searchLocale` as the URL locale since vectors have no per-locale field.
 */
export function getPageUrl(
  meta: Record<string, unknown>,
  searchLocale: string,
): string | null {
  const contentType = getContentType(meta)
  const segment = CONTENT_TYPE_SEGMENT[contentType]
  if (!segment) return null

  // Resolve locale for the URL
  const urlLocale = (meta['locale'] as string | undefined) ?? searchLocale

  // Resolve slug
  let slug: string | null = null

  if (typeof meta['slug'] === 'string' && meta['slug']) {
    // Approach B: direct slug field
    slug = meta['slug']
  } else if (contentType === 'course' && typeof meta['course_slug'] === 'string') {
    // Approach A courses: /courses/:courseSlug/:journeySlug
    const courseSlug = meta['course_slug'] as string
    const journeySlug = typeof meta['journey_slug'] === 'string' ? meta['journey_slug'] : ''
    if (!courseSlug) return null
    const path = journeySlug ? `${courseSlug}/${journeySlug}` : courseSlug
    return `https://apolitical.co/${urlLocale}/${segment}/${path}`
  } else {
    // Approach A articles/events: slug_{normalized_locale} fields
    const norm = urlLocale.toLowerCase().replace(/-/g, '_')
    const localeSlug = meta[`slug_${norm}`]
    const enSlug = meta['slug_en']
    if (typeof localeSlug === 'string' && localeSlug) slug = localeSlug
    else if (typeof enSlug === 'string' && enSlug) slug = enSlug
    else {
      for (const [k, v] of Object.entries(meta)) {
        if (k.startsWith('slug_') && typeof v === 'string' && v) { slug = v; break }
      }
    }
  }

  if (!slug) return null

  return `https://apolitical.co/${urlLocale}/${segment}/${slug}`
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
