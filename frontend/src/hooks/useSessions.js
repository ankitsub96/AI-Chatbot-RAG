import { useState, useEffect, useCallback } from 'react'
import { getSessions, getSessionHistory, deleteSession } from '../api/ragApi'

export function useSessions() {
  const [sessions, setSessions] = useState([])

  const refresh = useCallback(async () => {
    try {
      const data = await getSessions()
      setSessions(data)
    } catch { setSessions([]) }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const createSession = useCallback(() => {
    const id = `session-${Date.now()}`
    setSessions(prev => [id, ...prev])
    return id
  }, [])

  const removeSession = useCallback(async (id) => {
    await deleteSession(id)
    setSessions(prev => prev.filter(s => s !== id))
  }, [])

  const loadHistory = useCallback(async (id) => {
    try {
      return await getSessionHistory(id)
    } catch { return [] }
  }, [])

  return { sessions, removeSession, loadHistory, refresh }
}