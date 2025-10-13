import React, { useEffect, useRef, useState } from 'react'

interface Agent {
  id: number
  name: string
  x: number
  y: number
  state: string
  observations?: string[]
}

interface SimulationState {
  tick: number
  timestamp: string
  agents: Agent[]
}

interface MapProps {
  width?: number
  height?: number
}

export default function Map({ width = 800, height = 600 }: MapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [agents, setAgents] = useState<Agent[]>([])
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null)
  const [connected, setConnected] = useState(false)
  const [tick, setTick] = useState(0)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    // Connect to WebSocket (use environment variable or default to localhost)
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws'
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('Connected to Mini-Town server')
      setConnected(true)
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)

      if (data.type === 'init') {
        console.log('Received initial state:', data)
        setAgents(data.agents)
      } else if (data.agents) {
        // Simulation update
        setAgents(data.agents)
        setTick(data.tick)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.onclose = () => {
      console.log('Disconnected from server')
      setConnected(false)
    }

    return () => {
      ws.close()
    }
  }, [])

  useEffect(() => {
    // Render the map
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.fillStyle = '#2a2a2a'
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    ctx.strokeStyle = '#3a3a3a'
    ctx.lineWidth = 1
    for (let x = 0; x < width; x += 50) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()
    }
    for (let y = 0; y < height; y += 50) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // Draw agents
    agents.forEach((agent) => {
      // Agent circle
      ctx.beginPath()
      ctx.arc(agent.x, agent.y, 15, 0, 2 * Math.PI)

      // Color based on whether agent is selected
      if (selectedAgent?.id === agent.id) {
        ctx.fillStyle = '#ffcc00'
      } else {
        ctx.fillStyle = '#4a90e2'
      }
      ctx.fill()

      // Border
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2
      ctx.stroke()

      // Perception radius (if selected)
      if (selectedAgent?.id === agent.id) {
        ctx.beginPath()
        ctx.arc(agent.x, agent.y, 50, 0, 2 * Math.PI)
        ctx.strokeStyle = 'rgba(255, 204, 0, 0.3)'
        ctx.lineWidth = 1
        ctx.stroke()
      }

      // Name label
      ctx.fillStyle = '#fff'
      ctx.font = '12px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(agent.name, agent.x, agent.y - 20)
    })
  }, [agents, selectedAgent, width, height])

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    // Find clicked agent
    const clickedAgent = agents.find((agent) => {
      const dx = agent.x - x
      const dy = agent.y - y
      const distance = Math.sqrt(dx * dx + dy * dy)
      return distance <= 15
    })

    setSelectedAgent(clickedAgent || null)
  }

  return (
    <div style={{ padding: '20px' }}>
      <div style={{ marginBottom: '10px', display: 'flex', justifyContent: 'space-between' }}>
        <h1>Mini-Town (Day 0.5)</h1>
        <div>
          <span style={{
            color: connected ? '#4ade80' : '#f87171',
            marginRight: '15px'
          }}>
            {connected ? '● Connected' : '● Disconnected'}
          </span>
          <span>Tick: {tick}</span>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        onClick={handleCanvasClick}
        style={{
          border: '2px solid #444',
          cursor: 'pointer',
          display: 'block',
          marginBottom: '20px'
        }}
      />

      {selectedAgent && (
        <div style={{
          background: '#2a2a2a',
          border: '1px solid #444',
          borderRadius: '8px',
          padding: '15px',
          maxWidth: '800px'
        }}>
          <h2>{selectedAgent.name} (Agent {selectedAgent.id})</h2>
          <p>Position: ({selectedAgent.x.toFixed(1)}, {selectedAgent.y.toFixed(1)})</p>
          <p>State: {selectedAgent.state}</p>

          {selectedAgent.observations && selectedAgent.observations.length > 0 && (
            <div style={{ marginTop: '10px' }}>
              <h3>Observations:</h3>
              <ul style={{ marginLeft: '20px' }}>
                {selectedAgent.observations.map((obs, idx) => (
                  <li key={idx}>{obs}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      <div style={{ marginTop: '20px', color: '#888', fontSize: '14px' }}>
        <p>Click on an agent to see details</p>
        <p>Agents perform random walks and perceive nearby agents within 50 pixels</p>
      </div>
    </div>
  )
}
