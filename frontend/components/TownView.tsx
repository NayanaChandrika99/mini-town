'use client'

import { useEffect, useMemo, useState } from 'react'
import MapCanvas from './MapCanvas'
import AgentInspector from './AgentInspector'
import SystemPanel from './SystemPanel'
import GodModeControls from './GodModeControls'
import { useWebSocket } from '../lib/websocket'

export default function TownView() {
  const {
    agents,
    systemState,
    connected,
    tick,
    mapConfig,
    lastMessageAt,
    error,
    events
  } = useWebSocket()

  const [selectedAgentId, setSelectedAgentId] = useState<number | null>(null)

  useEffect(() => {
    if (selectedAgentId === null) {
      return
    }
    if (!agents.some((agent) => agent.id === selectedAgentId)) {
      setSelectedAgentId(null)
    }
  }, [agents, selectedAgentId])

  const mapDimensions = useMemo(() => {
    const width = mapConfig?.map_width ?? 800
    const height = mapConfig?.map_height ?? 600
    return { width, height }
  }, [mapConfig])

  const selectedAgent = useMemo(() => {
    if (selectedAgentId === null) {
      return null
    }
    return agents.find((agent) => agent.id === selectedAgentId) ?? null
  }, [agents, selectedAgentId])

  const agentCountLabel = agents.length === 1 ? '1 agent' : `${agents.length} agents`

  return (
    <div className="app-shell">
      <header className="top-bar">
        <div>
          <h1>Mini-Town Control Room</h1>
          <p className="muted">DSPy-compiled agents with real-time controls</p>
        </div>
        <div className="status-cluster">
          <span>
            <span
              className="status-dot"
              style={{ backgroundColor: connected ? 'var(--accent-green)' : 'var(--accent-red)' }}
            />
            {connected ? 'Connected' : 'Disconnected'}
          </span>
          <span>{systemState?.paused ? 'Paused' : 'Running'}</span>
          <span>Tick {tick}</span>
        </div>
      </header>

      <div className="main-layout">
        <div className="panel map-card">
          <div className="map-header">
            <span className="section-title">Town Map</span>
            <div className="map-meta">
              <span>{agentCountLabel}</span>
              {selectedAgent ? <span className="muted">Selected: {selectedAgent.name}</span> : null}
            </div>
          </div>
          {error ? <div className="muted">WebSocket: {error}</div> : null}
          <MapCanvas
            width={mapDimensions.width}
            height={mapDimensions.height}
            agents={agents}
            selectedAgentId={selectedAgentId}
            onAgentClick={setSelectedAgentId}
          />
        </div>

        <div className="stack">
          <div className="panel system-card">
            <SystemPanel
              systemState={systemState}
              connected={connected}
              tick={tick}
              lastMessageAt={lastMessageAt}
              events={events}
              agents={agents}
            />
          </div>

          <div className="panel inspector-card">
            <AgentInspector agentId={selectedAgentId} />
          </div>

          <div className="panel god-card">
            <GodModeControls systemState={systemState} />
          </div>
        </div>
      </div>
    </div>
  )
}
