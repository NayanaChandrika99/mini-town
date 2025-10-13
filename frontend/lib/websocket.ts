import { useEffect, useRef, useState } from 'react'
import { getWebSocketUrl } from './config'

export interface AgentMsg {
  id: number
  name: string
  x: number
  y: number
  state: string
  observations?: string[]
  current_plan?: string | null
  plan_source?: string | null
  plan_last_updated?: string | null
}

export interface TownEvent {
  id: string
  type: string
  timestamp: string
  severity?: number
  location?: [number, number]
}

export interface SystemState {
  llm_provider: string
  llm_model: string
  optimizer: string
  town_score: number
  avg_latency: number
  tick_interval: number
  paused?: boolean
  recent_events?: TownEvent[]
}

export interface MapConfig {
  map_width: number
  map_height: number
}

interface InitMessage {
  type: 'init'
  agents: AgentMsg[]
  system?: SystemState
  config?: MapConfig
}

interface AgentsUpdateMessage {
  type: 'agents_update'
  agents: AgentMsg[]
  tick?: number
  timestamp?: string
}

interface SystemUpdateMessage {
  type: 'system_update'
  state: SystemState
}

interface EventBroadcastMessage {
  type: 'event'
  event: TownEvent
}

type ServerMessage =
  | InitMessage
  | AgentsUpdateMessage
  | SystemUpdateMessage
  | EventBroadcastMessage

export interface WebSocketState {
  agents: AgentMsg[]
  systemState: SystemState | null
  connected: boolean
  tick: number
  mapConfig: MapConfig | null
  lastMessageAt: number | null
  error: string | null
  events: TownEvent[]
}

const RECONNECT_DELAY_MS = 3000

export function useWebSocket(customUrl?: string): WebSocketState {
  const wsUrl = customUrl ?? getWebSocketUrl()
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const [agents, setAgents] = useState<AgentMsg[]>([])
  const [systemState, setSystemState] = useState<SystemState | null>(null)
  const [connected, setConnected] = useState(false)
  const [tick, setTick] = useState(0)
  const [mapConfig, setMapConfig] = useState<MapConfig | null>(null)
  const [lastMessageAt, setLastMessageAt] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [events, setEvents] = useState<TownEvent[]>([])

  useEffect(() => {
    let active = true

    const cleanupSocket = () => {
      if (wsRef.current) {
        wsRef.current.onopen = null
        wsRef.current.onmessage = null
        wsRef.current.onclose = null
        wsRef.current.onerror = null
        try {
          wsRef.current.close()
        } catch (_) {
          // Ignore errors closing stale sockets
        }
        wsRef.current = null
      }
    }

    const scheduleReconnect = () => {
      if (!active || reconnectRef.current) {
        return
      }

      reconnectRef.current = setTimeout(() => {
        reconnectRef.current = null
        if (active) {
          connect()
        }
      }, RECONNECT_DELAY_MS)
    }

    const connect = () => {
      cleanupSocket()
      const socket = new WebSocket(wsUrl)
      wsRef.current = socket

      socket.onopen = () => {
        if (!active) {
          return
        }
        setConnected(true)
        setError(null)
      }

      socket.onclose = () => {
        if (!active) {
          return
        }
        setConnected(false)
        scheduleReconnect()
      }

      socket.onerror = () => {
        if (!active) {
          return
        }
        setError('WebSocket error')
      }

      socket.onmessage = (event: MessageEvent) => {
        if (!active) {
          return
        }

        try {
          const raw = JSON.parse(event.data)
          if (!raw || typeof raw !== 'object' || typeof (raw as { type?: unknown }).type !== 'string') {
            return
          }

          const message = raw as ServerMessage
          setLastMessageAt(Date.now())

          switch (message.type) {
            case 'init':
              setAgents(message.agents ?? [])
              if (message.system) {
                setSystemState(message.system)
                if (message.system.recent_events) {
                  setEvents(message.system.recent_events)
                }
              }
              if (message.config) {
                setMapConfig(message.config)
              }
              setTick(0)
              break
            case 'agents_update':
              setAgents(message.agents ?? [])
              if (typeof message.tick === 'number') {
                setTick(message.tick)
              }
              break
            case 'system_update':
              setSystemState(message.state)
              if (message.state.recent_events) {
                setEvents(message.state.recent_events)
              }
              break
            case 'event':
              if (message.event) {
                setEvents((prev) => {
                  const next = [message.event, ...prev]
                  // Ensure deduplication and limit to 10 entries.
                  const unique = []
                  const seen = new Set<string>()
                  for (const evt of next) {
                    if (seen.has(evt.id)) {
                      continue
                    }
                    unique.push(evt)
                    seen.add(evt.id)
                    if (unique.length >= 10) {
                      break
                    }
                  }
                  return unique
                })
              }
              break
            default:
              // Ignore unknown message types
              break
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message', err)
          setError('Failed to parse server message')
        }
      }
    }

    connect()

    return () => {
      active = false
      if (reconnectRef.current) {
        clearTimeout(reconnectRef.current)
      }
      cleanupSocket()
    }
  }, [wsUrl])

  return {
    agents,
    systemState,
    connected,
    tick,
    mapConfig,
    lastMessageAt,
    error,
    events
  }
}
