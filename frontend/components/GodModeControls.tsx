'use client'

import React, { useEffect, useState, useMemo } from 'react'
import { getApiBaseUrl } from '../lib/config'
import type { Agent, SystemState } from '../lib/websocket'

// Type definitions for our new data structures
interface Landmark {
  id: string
  name: string
  position: { x: number; y: number }
}

interface PlanPreset {
  id: string
  agent_name: string
  description: string
  plan_text: string
}

interface GodModeProps {
  systemState: SystemState | null
  agents: Agent[]
}

interface ToastState {
  message: string
  tone: 'info' | 'error'
}

const apiBase = getApiBaseUrl().replace(/\/$/, '')

export default function GodModeControls({ systemState, agents }: GodModeProps) {
  const [busy, setBusy] = useState<boolean>(false)
  const [toast, setToast] = useState<ToastState | null>(null)

  // State for the demo controls
  const [landmarks, setLandmarks] = useState<Landmark[]>([])
  const [presets, setPresets] = useState<PlanPreset[]>([])
  const [selectedAgentId, setSelectedAgentId] = useState<string>('')
  const [selectedLandmarkId, setSelectedLandmarkId] = useState<string>('')
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')

  // Toast message timeout
  useEffect(() => {
    if (!toast) return
    const timeout = setTimeout(() => setToast(null), 4000)
    return () => clearTimeout(timeout)
  }, [toast])

  // Fetch initial data for the panel
  useEffect(() => {
    const fetchData = async () => {
      try {
        const landmarksRes = await fetch(`${apiBase}/ai-town/control/landmarks`)
        if (!landmarksRes.ok) throw new Error('Failed to fetch landmarks')
        setLandmarks(await landmarksRes.json())

        const presetsRes = await fetch(`${apiBase}/ai-town/control/presets`)
        if (!presetsRes.ok) throw new Error('Failed to fetch presets')
        setPresets(await presetsRes.json())
      } catch (err) {
        console.error('Failed to fetch demo data', err)
        setToast({ message: 'Could not load demo data', tone: 'error' })
      }
    }
    fetchData()
  }, [])

  // API call helper
  const postJson = async (path: string, body?: unknown) => {
    setBusy(true)
    try {
      const response = await fetch(`${apiBase}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : undefined
      })
      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Request failed: ${response.status} ${errorText}`)
      }
      return await response.json()
    } finally {
      setBusy(false)
    }
  }

  // Filter presets for the selected agent
  const availablePresets = useMemo(() => {
    if (!selectedAgentId) return []
    const selectedAgent = agents.find((a) => a.id === Number(selectedAgentId))
    if (!selectedAgent) return []
    return presets.filter((p) => p.agent_name === selectedAgent.name)
  }, [selectedAgentId, presets, agents])

  // Event Handlers
  const handleTeleport = async () => {
    if (!selectedAgentId || !selectedLandmarkId) {
      setToast({ message: 'Select an agent and a landmark first', tone: 'error' })
      return
    }
    try {
      await postJson('/ai-town/control/teleport', {
        agent_id: Number(selectedAgentId),
        landmark_id: selectedLandmarkId
      })
      setToast({ message: 'Teleport command sent', tone: 'info' })
    } catch (err) {
      console.error('Teleport failed', err)
      setToast({ message: 'Teleport failed', tone: 'error' })
    }
  }

  const handleApplyPlan = async () => {
    if (!selectedAgentId || !selectedPresetId) {
      setToast({ message: 'Select an agent and a plan first', tone: 'error' })
      return
    }
    try {
      await postJson('/ai-town/control/apply_plan', {
        agent_id: Number(selectedAgentId),
        preset_id: selectedPresetId
      })
      setToast({ message: 'Plan applied successfully', tone: 'info' })
    } catch (err) {
      console.error('Apply plan failed', err)
      setToast({ message: 'Failed to apply plan', tone: 'error' })
    }
  }

  return (
    <>
      <h2>Demo Controls</h2>
      <p className="muted">Directly control agents to demonstrate DSPy-optimized plans.</p>

      <div className="demo-grid">
        {/* Agent Selection */}
        <div className="control-group">
          <label htmlFor="agent-select">1. Select Agent</label>
          <select
            id="agent-select"
            value={selectedAgentId}
            onChange={(e) => {
              setSelectedAgentId(e.target.value)
              setSelectedPresetId('') // Reset plan selection
            }}
            disabled={busy}
          >
            <option value="">-- Choose Agent --</option>
            {agents.map((agent) => (
              <option key={agent.id} value={agent.id}>
                {agent.name}
              </option>
            ))}
          </select>
        </div>

        {/* Landmark Teleport */}
        <div className="control-group">
          <label htmlFor="landmark-select">2. Teleport To</label>
          <select
            id="landmark-select"
            value={selectedLandmarkId}
            onChange={(e) => setSelectedLandmarkId(e.target.value)}
            disabled={busy || !selectedAgentId}
          >
            <option value="">-- Choose Landmark --</option>
            {landmarks.map((landmark) => (
              <option key={landmark.id} value={landmark.id}>
                {landmark.name}
              </option>
            ))}
          </select>
          <button onClick={handleTeleport} disabled={busy || !selectedAgentId || !selectedLandmarkId}>
            Teleport
          </button>
        </div>

        {/* Plan Preset Application */}
        <div className="control-group">
          <label htmlFor="preset-select">3. Apply Plan</label>
          <select
            id="preset-select"
            value={selectedPresetId}
            onChange={(e) => setSelectedPresetId(e.target.value)}
            disabled={busy || !selectedAgentId}
          >
            <option value="">-- Choose Plan --</option>
            {availablePresets.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.description}
              </option>
            ))}
          </select>
          <button onClick={handleApplyPlan} disabled={busy || !selectedAgentId || !selectedPresetId}>
            Apply Plan
          </button>
        </div>
      </div>

      {toast ? (
        <div className="toast" style={{ color: toast.tone === 'error' ? 'var(--accent-red)' : 'var(--text-secondary)' }}>
          {toast.message}
        </div>
      ) : null}
    </>
  )
}