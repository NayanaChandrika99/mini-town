import React, { useEffect, useState } from 'react'
import { getApiBaseUrl } from '../lib/config'

const REFRESH_INTERVAL_MS = 2000

interface Memory {
  id: number
  ts: string
  content: string
  importance: number
}

interface AgentDetail {
  id: number
  name: string
  state: string
  goal?: string
  personality?: string
  current_plan?: string
  plan_source?: string | null
  plan_last_updated?: string | null
  x?: number
  y?: number
  memories: Memory[]
  latest_reflection?: string | null
}

interface AgentInspectorProps {
  agentId: number | null
}

const apiBaseUrl = getApiBaseUrl().replace(/\/$/, '')

function formatTimestamp(ts: string): string {
  const date = new Date(ts)
  if (Number.isNaN(date.getTime())) {
    return ts
  }
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function formatRelativeFromISO(iso: string | null | undefined): string {
  if (!iso) {
    return 'Unknown'
  }
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) {
    return 'Unknown'
  }
  const deltaSeconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000))
  if (deltaSeconds < 60) {
    return `${deltaSeconds}s ago`
  }
  const minutes = Math.round(deltaSeconds / 60)
  if (minutes < 60) {
    return `${minutes}m ago`
  }
  const hours = Math.round(minutes / 60)
  return `${hours}h ago`
}

function formatPlanSource(source: string | null | undefined): string {
  if (!source) {
    return 'unknown'
  }
  return source.replace(/_/g, ' ')
}

function badgeClass(importance: number): string {
  if (importance >= 0.75) {
    return 'badge badge-success'
  }
  if (importance >= 0.5) {
    return 'badge badge-info'
  }
  return 'badge badge-warning'
}

export default function AgentInspector({ agentId }: AgentInspectorProps) {
  const [agent, setAgent] = useState<AgentDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!agentId) {
      setAgent(null)
      setError(null)
      setLoading(false)
      return
    }

    let isMounted = true
    let controller: AbortController | null = null

    const fetchAgent = async (withSpinner: boolean) => {
      try {
        controller?.abort()
        controller = new AbortController()
        if (withSpinner) {
          setLoading(true)
        }

        const response = await fetch(`${apiBaseUrl}/api/agents/${agentId}`, {
          signal: controller.signal
        })

        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`)
        }

        const data: AgentDetail = await response.json()
        if (!isMounted) {
          return
        }

        setAgent(data)
        setError(null)
      } catch (err) {
        if (!isMounted) {
          return
        }
        if ((err as Error).name === 'AbortError') {
          return
        }
        console.error('Failed to load agent details', err)
        setError('Unable to load agent details')
      } finally {
        if (withSpinner && isMounted) {
          setLoading(false)
        }
      }
    }

    fetchAgent(true)
    const intervalId = setInterval(() => fetchAgent(false), REFRESH_INTERVAL_MS)

    return () => {
      isMounted = false
      controller?.abort()
      clearInterval(intervalId)
    }
  }, [agentId])

  if (!agentId) {
    return (
      <div>
        <h2>Agent Inspector</h2>
        <p className="muted">Select an agent on the map to inspect details.</p>
      </div>
    )
  }

  if (loading && !agent) {
    return (
      <div>
        <h2>Agent Inspector</h2>
        <p className="muted">Loading agent details...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div>
        <h2>Agent Inspector</h2>
        <p className="muted">{error}</p>
      </div>
    )
  }

  if (!agent) {
    return (
      <div>
        <h2>Agent Inspector</h2>
        <p className="muted">No data available.</p>
      </div>
    )
  }

  return (
    <div>
      <h2>{agent.name}</h2>
      <div className="muted">Agent #{agent.id}</div>

      <div className="section-title">Status</div>
      <div>
        {agent.state}
        {typeof agent.x === 'number' && typeof agent.y === 'number'
          ? ` · (${agent.x.toFixed(1)}, ${agent.y.toFixed(1)})`
          : ''}
      </div>

      <div className="section-title">Goal</div>
      <div>{agent.goal || '-'}</div>

      <div className="section-title">Personality</div>
      <div>{agent.personality || '-'}</div>

      <div className="section-title">Current Plan</div>
      <div className="muted" style={{ fontSize: '0.85rem', marginBottom: 4 }}>
        Source: {formatPlanSource(agent.plan_source)} · Last updated {formatRelativeFromISO(agent.plan_last_updated)}
      </div>
      <div className="plan-box">
        {agent.current_plan
          ? agent.current_plan.split(/\n+/).map((line, idx) => (
              <div key={idx}>{line}</div>
            ))
          : 'No plan available'}
      </div>

      <div className="section-title">Recent Memories</div>
      {agent.memories && agent.memories.length > 0 ? (
        <div className="memories-list">
          {agent.memories.map((memory) => (
            <div key={memory.id} className="memory-item">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span className="muted">{formatTimestamp(memory.ts)}</span>
                <span className={badgeClass(memory.importance)}>
                  {(memory.importance * 100).toFixed(0)}%
                </span>
              </div>
              <div>{memory.content}</div>
            </div>
          ))}
        </div>
      ) : (
        <div className="muted">No stored memories yet.</div>
      )}

      {agent.latest_reflection ? (
        <>
          <div className="section-title">Latest Reflection</div>
          <div className="memory-item">{agent.latest_reflection}</div>
        </>
      ) : null}
    </div>
  )
}
