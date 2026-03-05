import type { SearchTiming } from '../types'
import { getSparseMs, getSparseLabel } from '../types'

interface Props {
  timingA: SearchTiming | null
  timingB: SearchTiming | null
}

interface Metric {
  label: string
  labelA: string
  labelB: string
  valueA: number
  valueB: number
  isBold?: boolean
}

export default function TimingPanel({ timingA, timingB }: Props) {
  if (!timingA && !timingB) return null

  const a = timingA ?? { embedMs: 0, pineconeMs: 0, totalMs: 0 }
  const b = timingB ?? { embedMs: 0, pineconeMs: 0, totalMs: 0 }

  const metrics: Metric[] = [
    {
      label: 'Dense embed',
      labelA: 'embedMs',
      labelB: 'embedMs',
      valueA: a.embedMs,
      valueB: b.embedMs,
    },
    {
      label: 'Sparse encode',
      labelA: getSparseLabel(a),
      labelB: getSparseLabel(b),
      valueA: getSparseMs(a),
      valueB: getSparseMs(b),
    },
    {
      label: 'Pinecone query',
      labelA: 'pineconeMs',
      labelB: 'pineconeMs',
      valueA: a.pineconeMs,
      valueB: b.pineconeMs,
    },
    {
      label: 'Total',
      labelA: 'totalMs',
      labelB: 'totalMs',
      valueA: a.totalMs,
      valueB: b.totalMs,
      isBold: true,
    },
  ]

  const maxTotal = Math.max(...metrics.map((m) => Math.max(m.valueA, m.valueB)), 1)

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-4">
        Latency comparison
      </h2>

      <div className="space-y-4">
        {metrics.map((m) => {
          const maxRow = Math.max(m.valueA, m.valueB, 1)
          const pctA = (m.valueA / maxTotal) * 100
          const pctB = (m.valueB / maxTotal) * 100
          const winner = m.valueA < m.valueB ? 'a' : m.valueB < m.valueA ? 'b' : 'tie'

          return (
            <div key={m.label}>
              <div className="flex items-center justify-between mb-1.5">
                <span className={`text-xs ${m.isBold ? 'font-bold text-gray-900' : 'font-medium text-gray-600'}`}>
                  {m.label}
                </span>
                <span className="text-xs text-gray-400">
                  {winner === 'a' && timingA && timingB && (
                    <span className="text-indigo-600 font-medium">A faster by {(m.valueB - m.valueA).toFixed(1)}ms</span>
                  )}
                  {winner === 'b' && timingA && timingB && (
                    <span className="text-emerald-600 font-medium">B faster by {(m.valueA - m.valueB).toFixed(1)}ms</span>
                  )}
                </span>
              </div>

              {/* Row A */}
              <div className="flex items-center gap-2 mb-1">
                <span className="w-16 text-right text-xs font-mono text-indigo-600 shrink-0">
                  {timingA ? `${m.valueA}ms` : '—'}
                </span>
                <div className="flex-1 h-5 bg-gray-100 rounded overflow-hidden">
                  {timingA && (
                    <div
                      className="h-full bg-indigo-500 rounded transition-all duration-500 flex items-center pl-1.5"
                      style={{ width: `${pctA}%` }}
                    >
                      {pctA > 15 && (
                        <span className="text-white text-[10px] font-mono whitespace-nowrap">
                          A · {m.labelA}
                        </span>
                      )}
                    </div>
                  )}
                </div>
                {pctA <= 15 && timingA && (
                  <span className="text-[10px] text-indigo-400 font-mono shrink-0">A</span>
                )}
              </div>

              {/* Row B */}
              <div className="flex items-center gap-2">
                <span className="w-16 text-right text-xs font-mono text-emerald-600 shrink-0">
                  {timingB ? `${m.valueB}ms` : '—'}
                </span>
                <div className="flex-1 h-5 bg-gray-100 rounded overflow-hidden">
                  {timingB && (
                    <div
                      className="h-full bg-emerald-500 rounded transition-all duration-500 flex items-center pl-1.5"
                      style={{ width: `${pctB}%` }}
                    >
                      {pctB > 15 && (
                        <span className="text-white text-[10px] font-mono whitespace-nowrap">
                          B · {m.labelB}
                        </span>
                      )}
                    </div>
                  )}
                </div>
                {pctB <= 15 && timingB && (
                  <span className="text-[10px] text-emerald-400 font-mono shrink-0">B</span>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Legend */}
      <div className="mt-4 pt-4 border-t border-gray-100 flex gap-6 text-xs text-gray-500">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded bg-indigo-500 inline-block" />
          Approach A — Pinecone-hosted sparse
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded bg-emerald-500 inline-block" />
          Approach B — corpus-fitted BM25
        </span>
      </div>
    </div>
  )
}
