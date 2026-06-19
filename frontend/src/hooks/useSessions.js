import { useState, useEffect, useCallback } from 'react'
import { getSessions, getSessionHistory, deleteSession, createSession as apiCreateSession } from '../api/ragApi'

export function useSessions() {
  const [sessions, setSessions] = useState([])

  const fetchSessions = useCallback(async (signal) => {
    try {
      const data = await getSessions(signal)
      if (signal?.aborted) return
      // normalize: backend returns objects or strings depending on version
      setSessions(Array.isArray(data) ? data : [])
    } catch {
      if (signal?.aborted) return
      setSessions([])
    }
  }, [])

  // Initial load. Wrapped with an AbortController so that React StrictMode's
  // dev-only mount → cleanup → mount cycle cancels the first (now-stale)
  // request instead of letting both land — this is what was causing every
  // API call to fire twice on page load/refresh.
  useEffect(() => {
    const controller = new AbortController()
    fetchSessions(controller.signal)
    return () => controller.abort()
  }, [fetchSessions])

  // Manual refresh (e.g. after creating/deleting a session) — no signal
  // needed since this isn't called from a mount effect.
  const refresh = useCallback(() => fetchSessions(), [fetchSessions])

  // calls real API, returns session object { session_id, title, created_at }
  const createSession = useCallback(async (title) => {
    try {
      const session = await apiCreateSession(title || null)
      setSessions(prev => [session, ...prev])
      return session
    } catch {
      // fallback: create a local session object if API fails
      const fallback = { session_id: `session-${Date.now()}`, title: title || null, created_at: new Date().toISOString() }
      setSessions(prev => [fallback, ...prev])
      return fallback
    }
  }, [])

  const removeSession = useCallback(async (id) => {
    await deleteSession(id)
    setSessions(prev => prev.filter(s => (s.session_id || s) !== id))
  }, [])

  const loadHistory = useCallback(async (id, signal) => {
    try {
      return await getSessionHistory(id, 1, 50, signal)
    } catch {
      return []
    }
  }, [])

  return { sessions, createSession, removeSession, loadHistory, refresh }
}
