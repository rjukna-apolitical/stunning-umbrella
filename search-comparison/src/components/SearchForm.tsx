import { useState, useRef, useEffect, type FormEvent } from 'react'
import type { SearchParams } from '../types'
import { TEST_CASES, GROUPS, type TestCase } from '../testCases'

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

const CONTENT_TYPE_BADGE: Record<string, string> = {
  solutionArticle: 'bg-blue-100 text-blue-600',
  event: 'bg-purple-100 text-purple-600',
  course: 'bg-amber-100 text-amber-600',
}

interface Props {
  onSearch: (params: SearchParams) => void
  isLoading: boolean
}

export default function SearchForm({ onSearch, isLoading }: Props) {
  const [query, setQuery] = useState('')
  const [locale, setLocale] = useState('en')
  const [contentType, setContentType] = useState('')
  const [pageSize, setPageSize] = useState(10)

  const [showSuggestions, setShowSuggestions] = useState(false)
  const [activeGroup, setActiveGroup] = useState(GROUPS[0])
  const [filterText, setFilterText] = useState('')

  const suggestionsRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Close dropdown when clicking outside
  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (suggestionsRef.current && !suggestionsRef.current.contains(e.target as Node)) {
        setShowSuggestions(false)
      }
    }
    document.addEventListener('mousedown', onClickOutside)
    return () => document.removeEventListener('mousedown', onClickOutside)
  }, [])

  // Filter test cases: match query text against the filter string
  const filteredCases: TestCase[] = filterText.trim()
    ? TEST_CASES.filter(
        (tc) =>
          tc.query.toLowerCase().includes(filterText.toLowerCase()) ||
          tc.description.toLowerCase().includes(filterText.toLowerCase()) ||
          tc.group.toLowerCase().includes(filterText.toLowerCase()),
      )
    : TEST_CASES.filter((tc) => tc.group === activeGroup)

  function applyTestCase(tc: TestCase) {
    setQuery(tc.query)
    setLocale(tc.locale)
    setContentType(tc.contentType)
    setShowSuggestions(false)
    setFilterText('')
    // Auto-submit
    onSearch({ query: tc.query, locale: tc.locale, contentType: tc.contentType, pageSize })
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setShowSuggestions(false)
    onSearch({ query: query.trim(), locale, contentType, pageSize })
  }

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      {/* Query row */}
      <div className="flex gap-3 relative">
        <div className="relative flex-1">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              setFilterText(e.target.value)
              setShowSuggestions(true)
            }}
            onFocus={() => setShowSuggestions(true)}
            placeholder="Enter search query…"
            className="w-full px-4 py-2.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent pr-10"
          />
          {/* Clear button */}
          {query && (
            <button
              type="button"
              onClick={() => { setQuery(''); setFilterText(''); inputRef.current?.focus() }}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
            >
              ✕
            </button>
          )}
        </div>

        {/* Suggestions toggle */}
        <button
          type="button"
          onClick={() => setShowSuggestions((v) => !v)}
          title="Show test cases"
          className={`px-3 py-2.5 border rounded-lg text-sm transition-colors ${
            showSuggestions
              ? 'bg-indigo-50 border-indigo-300 text-indigo-700'
              : 'border-gray-300 text-gray-500 hover:bg-gray-50'
          }`}
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h10" />
          </svg>
        </button>

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

      {/* Suggestions dropdown */}
      {showSuggestions && (
        <div
          ref={suggestionsRef}
          className="mt-2 bg-white border border-gray-200 rounded-xl shadow-lg z-20 overflow-hidden"
        >
          {/* Group tabs (hidden when filtering by text) */}
          {!filterText.trim() && (
            <div className="flex gap-1 p-2 border-b border-gray-100 overflow-x-auto scrollbar-none">
              {GROUPS.map((g) => (
                <button
                  key={g}
                  type="button"
                  onClick={() => setActiveGroup(g)}
                  className={`px-3 py-1 rounded-full text-xs font-medium whitespace-nowrap transition-colors ${
                    activeGroup === g
                      ? 'bg-indigo-600 text-white'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {g}
                </button>
              ))}
            </div>
          )}

          {filterText.trim() && (
            <div className="px-4 py-2 border-b border-gray-100 text-xs text-gray-500">
              {filteredCases.length} result{filteredCases.length !== 1 ? 's' : ''} matching "{filterText}"
            </div>
          )}

          {/* Cases list */}
          <ul className="max-h-64 overflow-y-auto divide-y divide-gray-50">
            {filteredCases.length === 0 ? (
              <li className="px-4 py-6 text-sm text-gray-400 text-center">No matching test cases</li>
            ) : (
              filteredCases.map((tc, i) => (
                <li key={i}>
                  <button
                    type="button"
                    onClick={() => applyTestCase(tc)}
                    className="w-full text-left px-4 py-3 hover:bg-indigo-50 transition-colors flex items-start justify-between gap-3"
                  >
                    <div className="min-w-0">
                      <div className="text-sm font-medium text-gray-900 truncate">{tc.query}</div>
                      <div className="text-xs text-gray-400 mt-0.5">{tc.description}</div>
                    </div>
                    <div className="flex items-center gap-1.5 shrink-0 mt-0.5">
                      <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-gray-100 text-gray-600">
                        {tc.locale}
                      </span>
                      {tc.contentType && (
                        <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${CONTENT_TYPE_BADGE[tc.contentType] ?? 'bg-gray-100 text-gray-600'}`}>
                          {tc.contentType}
                        </span>
                      )}
                    </div>
                  </button>
                </li>
              ))
            )}
          </ul>

          <div className="px-4 py-2 border-t border-gray-100 text-xs text-gray-400 flex justify-between">
            <span>{TEST_CASES.length} test cases across {GROUPS.length} groups</span>
            <span>Click to apply &amp; search</span>
          </div>
        </div>
      )}

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
