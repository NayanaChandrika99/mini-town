import React, { useEffect, useState } from 'react'
import { getApiBaseUrl } from '../lib/config'
import type { SystemState } from '../lib/websocket'

type BusyAction = 'pause' | 'step' | 'inject' | 'refresh' | null

interface GodModeProps {
  systemState: SystemState | null
}

interface ToastState {
  message: string
  tone: 'info' | 'error'
}

const apiBase = getApiBaseUrl().replace(/\/$/, '')

export default function GodModeControls({ systemState }: GodModeProps) {
  const [busy, setBusy] = useState<BusyAction>(null)
  const [toast, setToast] = useState<ToastState | null>(null)
  const [eventType, setEventType] = useState('')
  const [severity, setSeverity] = useState<string>('')
  const [location, setLocation] = useState<string>('')
  const [refreshAgentId, setRefreshAgentId] = useState<string>('')

  useEffect(() => {
    if (!toast) {
      return
    }
    const timeout = setTimeout(() => setToast(null), 4000)
    return () => clearTimeout(timeout)
  }, [toast])

  const paused = systemState?.paused ?? false

  const postJson = async (path: string, body?: unknown) => {
    const response = await fetch(`${apiBase}${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: body ? JSON.stringify(body) : undefined
    })

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`)
    }

    try {
      return await response.json()
    } catch (err) {
      console.warn('Non-JSON response from', path, err)
      return null
    }
  }

  const togglePause = async () => {
    try {
      setBusy('pause')
      const desiredState = !paused
      const result = await postJson('/god/pause', { paused: desiredState })
      setToast({
        message: result?.status ? `Simulation ${result.status}` : 'Pause toggled',
        tone: 'info'
      })
    } catch (err) {
      console.error('Failed to toggle pause', err)
      setToast({ message: 'Unable to toggle pause', tone: 'error' })
    } finally {
      setBusy(null)
    }
  }

  const stepSimulation = async () => {
    try {
      setBusy('step')
      await postJson('/god/step')
      setToast({ message: 'Advanced one tick', tone: 'info' })
    } catch (err) {
      console.error('Failed to step simulation', err)
      setToast({ message: 'Unable to step simulation', tone: 'error' })
    } finally {
      setBusy(null)
    }
  }

  const injectEvent = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    try {
      setBusy('inject')

      const payload: Record<string, unknown> = {
        type: eventType.trim() || 'custom_event'
      }

      if (severity !== '') {
        const severityValue = Number(severity)
        if (!Number.isNaN(severityValue)) {
          payload.severity = severityValue
        }
      }

      if (location.trim()) {
        const parts = location.split(',').map((part) => Number(part.trim()))
        if (parts.length === 2 && parts.every((value) => !Number.isNaN(value))) {
          payload.location = parts as [number, number]
        }
      }

      await postJson('/god/inject_event', payload)
      setToast({ message: 'Event injected', tone: 'info' })
      setEventType('')
      setSeverity('')
      setLocation('')
    } catch (err) {
      console.error('Failed to inject event', err)
      setToast({ message: 'Unable to inject event', tone: 'error' })
    } finally {
      setBusy(null)
    }
  }

  const refreshPlans = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    try {
      setBusy('refresh')
      let payload: Record<string, unknown> = {}
      if (refreshAgentId.trim()) {
        const numericId = Number(refreshAgentId.trim())
        if (Number.isNaN(numericId)) {
          setToast({ message: 'Agent ID must be a number', tone: 'error' })
          return
        }
        payload = { agent_id: numericId }
      }
      const result = await postJson('/god/refresh_plan', payload)
      const count = typeof result?.refreshed === 'number' ? result.refreshed : 0
      setToast({ message: `Refreshed ${count} plan${count === 1 ? '' : 's'}`, tone: 'info' })
      setRefreshAgentId('')
    } catch (err) {
      console.error('Failed to refresh plans', err)
      setToast({ message: 'Unable to refresh plans', tone: 'error' })
    } finally {
      setBusy(null)
    }
  }

  return (
    <>
      <h2>God Mode</h2>
      <p className="muted">Pause, step, or inject narrative events for debugging.</p>

      <div className="god-actions">
        <button type="button" onClick={togglePause} disabled={busy === 'pause'}>
          {paused ? 'Resume Simulation' : 'Pause Simulation'}
        </button>
        <button type="button" onClick={stepSimulation} disabled={busy === 'step'}>
          Step Tick
        </button>
      </div>

      <form className="god-form" onSubmit={injectEvent}>
      <div className="section-title">Inject Event</div>
        <input
          type="text"
          placeholder="Event type (e.g., fire_alarm)"
          value={eventType}
          onChange={(event) => setEventType(event.target.value)}
        />
        <input
          type="text"
          placeholder="Severity (0-1)"
          value={severity}
          onChange={(event) => setSeverity(event.target.value)}
        />
        <input
          type="text"
          placeholder="Location x,y"
          value={location}
          onChange={(event) => setLocation(event.target.value)}
        />
        <button type="submit" disabled={busy === 'inject'}>
          Inject Event
        </button>
      </form>

      <form className="god-form" onSubmit={refreshPlans} style={{ marginTop: 16 }}>
        <div className="section-title">Regenerate Plans</div>
        <input
          type="text"
          placeholder="Agent ID (blank = all)"
          value={refreshAgentId}
          onChange={(event) => setRefreshAgentId(event.target.value)}
        />
        <button type="submit" disabled={busy === 'refresh'}>
          Refresh Plans
        </button>
      </form>

      {toast ? (
        <div className="toast" style={{ color: toast.tone === 'error' ? 'var(--accent-red)' : 'var(--text-secondary)' }}>
          {toast.message}
        </div>
      ) : null}
    </>
  )
}
