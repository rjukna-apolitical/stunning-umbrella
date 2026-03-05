import { useState, type FormEvent } from 'react'
import type { SearchParams } from '../types'

const LOCALES = [
  { value: 'en', label: 'English (en)' },
  { value: 'fr', label: 'French (fr)' },
  { value: 'fr-CA', label: 'French Canadian (fr-CA)' },
  { value: 'de', label: 'German (de)' },
  { value: 'es', label: 'Spanish (es)' },
  { value: 'es-419', label: 'Spanish LATAM (es-419)' },
  { value: 'pt', label: 'Portuguese (pt)' },
  { value: 'pt-BR', label: 'Portuguese Brazil (pt-BR)' },
  { value: 'it', label: 'Italian (it)' },
  { value: 'ar', label: 'Arabic (ar)' },
  { value: 'id', label: 'Indonesian (id)' },
  { value: 'ja', label: 'Japanese (ja)' },
  { value: 'ko', label: 'Korean (ko)' },
  { value: 'vi', label: 'Vietnamese (vi)' },
  { value: 'pl', label: 'Polish (pl)' },
  { value: 'uk', label: 'Ukrainian (uk)' },
  { value: 'sr-Cyrl', label: 'Serbian Cyrillic (sr-Cyrl)' },
]

const CONTENT_TYPES = [
  { value: '', label: 'All content types' },
  { value: 'solutionArticle', label: 'Solution Article' },
  { value: 'event', label: 'Event' },
  { value: 'course', label: 'Course' },
]

const PAGE_SIZES = [5, 10, 20]

interface Props {
  onSearch: (params: SearchParams) => void
  isLoading: boolean
}

export default function SearchForm({ onSearch, isLoading }: Props) {
  const [query, setQuery] = useState('')
  const [locale, setLocale] = useState('en')
  const [contentType, setContentType] = useState('')
  const [pageSize, setPageSize] = useState(10)

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    onSearch({ query: query.trim(), locale, contentType, pageSize })
  }

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      {/* Query */}
      <div className="flex gap-3">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter search query…"
          className="flex-1 px-4 py-2.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
        />
        <button
          type="submit"
          disabled={isLoading || !query.trim()}
          className="px-6 py-2.5 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
              </svg>
              Searching…
            </span>
          ) : (
            'Search'
          )}
        </button>
      </div>

      {/* Filters */}
      <div className="mt-4 flex flex-wrap gap-4">
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Locale</label>
          <select
            value={locale}
            onChange={(e) => setLocale(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            {LOCALES.map((l) => (
              <option key={l.value} value={l.value}>{l.label}</option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Content Type</label>
          <select
            value={contentType}
            onChange={(e) => setContentType(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            {CONTENT_TYPES.map((c) => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Results</label>
          <select
            value={pageSize}
            onChange={(e) => setPageSize(Number(e.target.value))}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            {PAGE_SIZES.map((n) => (
              <option key={n} value={n}>{n} per page</option>
            ))}
          </select>
        </div>
      </div>
    </form>
  )
}
