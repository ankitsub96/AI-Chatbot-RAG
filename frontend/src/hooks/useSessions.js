import { useState, useEffect, useCallback } from 'react'
import { getSessions, getSessionHistory, deleteSession, createSession as apiCreateSession } from '../api/ragApi'

export function useSessions() {
  const [sessions, setSessions] = useState([])

  const refresh = useCallback(async () => {
    try {
      const data = await getSessions()
      // normalize: backend returns objects or strings depending on version
      setSessions(Array.isArray(data) ? data : [])
    } catch {
      setSessions([])
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

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

  const loadHistory = useCallback(async (id) => {
    try {
      return await getSessionHistory(id)
    } catch {
      return []
    }
  }, [])

  return { sessions, createSession, removeSession, loadHistory, refresh }
}
