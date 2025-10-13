'use client'

import React, { useEffect, useRef, useState } from 'react'
import type { AgentMsg } from '../lib/websocket'

interface MapCanvasProps {
  width: number
  height: number
  agents: AgentMsg[]
  selectedAgentId?: number | null
  onAgentClick?: (agentId: number | null) => void
}

const STATE_COLORS: Record<string, string> = {
  active: '#58a6ff',
  navigating: '#34d399',
  waiting: '#facc15',
  loitering: '#fbbf24',
  conversing: '#f472b6',
  observing: '#22d3ee',
  confused: '#f87171',
  idle: '#94a3b8',
  alert: '#f97316'
}

const SPRITE_SIZE = 32

export default function MapCanvas({
  width,
  height,
  agents,
  selectedAgentId = null,
  onAgentClick
}: MapCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [spriteSheet, setSpriteSheet] = useState<HTMLImageElement | null>(null)

  useEffect(() => {
    const image = new Image()
    image.src = '/assets/32x32folk.png'
    image.onload = () => setSpriteSheet(image)
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return
    }

    const dpr = window.devicePixelRatio ?? 1
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, width, height)

    ctx.fillStyle = '#0f1419'
    ctx.fillRect(0, 0, width, height)

    ctx.strokeStyle = '#1f2a34'
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

    agents.forEach((agent) => {
      const isSelected = agent.id === selectedAgentId
      const color = STATE_COLORS[agent.state] ?? '#58a6ff'

      if (spriteSheet) {
        ctx.drawImage(
          spriteSheet,
          0,
          0,
          SPRITE_SIZE,
          SPRITE_SIZE,
          agent.x - SPRITE_SIZE / 2,
          agent.y - SPRITE_SIZE / 2,
          SPRITE_SIZE,
          SPRITE_SIZE
        )
      } else {
        ctx.beginPath()
        ctx.arc(agent.x, agent.y, 14, 0, Math.PI * 2)
        ctx.fillStyle = isSelected ? '#facc15' : color
        ctx.fill()
      }

      ctx.lineWidth = isSelected ? 3 : 2
      ctx.strokeStyle = isSelected ? '#fef08a' : '#1f2937'
      ctx.beginPath()
      ctx.arc(agent.x, agent.y, spriteSheet ? SPRITE_SIZE / 2 + 2 : 14, 0, Math.PI * 2)
      ctx.stroke()

      ctx.fillStyle = '#e6edf3'
      ctx.font = '12px Inter, sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'bottom'
      ctx.fillText(agent.name, agent.x, agent.y - (spriteSheet ? SPRITE_SIZE / 2 + 6 : 20))
    })
  }, [agents, height, selectedAgentId, spriteSheet, width])

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onAgentClick) {
      return
    }

    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const rect = canvas.getBoundingClientRect()
    const scaleX = width / rect.width
    const scaleY = height / rect.height
    const clickX = (event.clientX - rect.left) * scaleX
    const clickY = (event.clientY - rect.top) * scaleY

    const hit = agents.find((agent) => {
      const dx = agent.x - clickX
      const dy = agent.y - clickY
      const radius = spriteSheet ? SPRITE_SIZE / 2 : 14
      return Math.sqrt(dx * dx + dy * dy) <= radius
    })

    onAgentClick(hit ? hit.id : null)
  }

  return (
    <canvas
      ref={canvasRef}
      className="map-canvas"
      width={width}
      height={height}
      onClick={handleCanvasClick}
      style={{ imageRendering: 'pixelated' }}
    />
  )
}
