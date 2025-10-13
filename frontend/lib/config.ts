const DEFAULT_API_BASE = 'http://localhost:8000'
const DEFAULT_WS = 'ws://localhost:8000/ws'

/**
 * Derive the API base URL for REST calls.
 */
export function getApiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_BASE
}

/**
 * Derive the WebSocket URL for simulation updates.
 * Prefers NEXT_PUBLIC_WS_URL, then NEXT_PUBLIC_API_URL, then defaults.
 */
export function getWebSocketUrl(): string {
  const explicit = process.env.NEXT_PUBLIC_WS_URL
  if (explicit) {
    return explicit
  }

  const apiBase = getApiBaseUrl()
  if (apiBase.startsWith('http://')) {
    return `${apiBase.replace(/^http:\/\//, 'ws://').replace(/\/$/, '')}/ws`
  }

  if (apiBase.startsWith('https://')) {
    return `${apiBase.replace(/^https:\/\//, 'wss://').replace(/\/$/, '')}/ws`
  }

  return DEFAULT_WS
}
