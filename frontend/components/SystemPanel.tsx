import React, { useMemo } from 'react'
import type { AgentMsg, SystemState, TownEvent } from '../lib/websocket'

interface SystemPanelProps {
  systemState: SystemState | null
  connected: boolean
  tick: number
  lastMessageAt: number | null
  events: TownEvent[]
  agents: AgentMsg[]
}

function formatRelativeTime(timestamp: number): string {
  const deltaSeconds = Math.max(0, Math.round((Date.now() - timestamp) / 1000))

  if (deltaSeconds < 5) {
    return 'Updated just now'
  }
  if (deltaSeconds < 60) {
    return `Updated ${deltaSeconds}s ago`
  }
  const minutes = Math.round(deltaSeconds / 60)
  if (minutes < 60) {
    return `Updated ${minutes}m ago`
  }
  const hours = Math.round(minutes / 60)
  return `Updated ${hours}h ago`
}

export default function SystemPanel({
  systemState,
  connected,
  tick,
  lastMessageAt,
  events,
  agents
}: SystemPanelProps) {
  const paused = systemState?.paused ?? false
  const latencyDisplay = systemState ? `${Math.round(systemState.avg_latency)} ms` : '—'
  const townScoreDisplay = systemState ? systemState.town_score.toFixed(2) : '—'
  const tickIntervalDisplay = systemState ? `${systemState.tick_interval.toFixed(1)} s` : '—'

  const planSummaries = useMemo(() => {
    return agents
      .filter((agent) => typeof agent.current_plan === 'string' && agent.current_plan.trim().length > 0)
      .map((agent) => {
        const updatedAtMs = agent.plan_last_updated ? Date.parse(agent.plan_last_updated) : null
        const firstLine = agent.current_plan!.split(/\n+/)[0]
        const sourceLabel = agent.plan_source ? agent.plan_source.replace(/_/g, ' ') : 'unknown'
        return {
          id: agent.id,
          name: agent.name,
          summary: firstLine,
          source: sourceLabel,
          updatedAtMs
        }
      })
      .sort((a, b) => {
        if (a.updatedAtMs === null && b.updatedAtMs === null) return 0
        if (a.updatedAtMs === null) return 1
        if (b.updatedAtMs === null) return -1
        return b.updatedAtMs - a.updatedAtMs
      })
      .slice(0, 4)
  }, [agents])

  const formatPlanTimestamp = (timestampMs: number | null): string => {
    if (timestampMs === null) {
      return 'Unknown'
    }
    return formatRelativeTime(timestampMs)
  }

  return (
    <>
      <h2>System Status</h2>
      <div className="status-cluster">
        <span>
          <span
            className="status-dot"
            style={{ backgroundColor: connected ? 'var(--accent-green)' : 'var(--accent-red)' }}
          />
          {connected ? 'Connected' : 'Disconnected'}
        </span>
        <span>{paused ? 'Paused' : 'Running'}</span>
        <span>Tick {tick}</span>
        {lastMessageAt ? <span>{formatRelativeTime(lastMessageAt)}</span> : null}
      </div>

      {systemState ? (
        <div className="system-grid">
          <div>
            <div className="section-title">LLM Provider</div>
            <div>{systemState.llm_provider}</div>
            <div className="muted">{systemState.llm_model}</div>
          </div>
          <div>
            <div className="section-title">Optimizer</div>
            <div>{systemState.optimizer}</div>
          </div>
          <div>
            <div className="section-title">Town Score</div>
            <div>{townScoreDisplay}</div>
          </div>
          <div>
            <div className="section-title">Avg Latency</div>
            <div>{latencyDisplay}</div>
          </div>
          <div>
            <div className="section-title">Tick Interval</div>
            <div>{tickIntervalDisplay}</div>
          </div>
        </div>
      ) : null}

      {events.length > 0 ? (
        <div>
          <div className="section-title">Recent Events</div>
          <ul className="event-list">
            {events.slice(0, 5).map((event) => (
              <li key={event.id}>
                <span className="event-type">{event.type}</span>
                <span className="muted">
                  {' '}
                  · {new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                </span>
              </li>
            ))}
          </ul>
      </div>
      ) : (
        <div className="muted">No recent events.</div>
      )}

      {planSummaries.length > 0 ? (
        <div>
          <div className="section-title">Agent Plans</div>
          <ul className="plan-list">
            {planSummaries.map((summary) => (
              <li key={summary.id}>
                <div className="muted">{summary.name}</div>
                <div>{summary.summary || 'Plan ready'}</div>
                <div className="muted" style={{ fontSize: '0.85rem' }}>
                  Source: {summary.source} · {formatPlanTimestamp(summary.updatedAtMs)}
                </div>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </>
  )
}
