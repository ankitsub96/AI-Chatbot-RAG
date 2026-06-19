import { useState, useEffect, useCallback } from 'react'
import { getSessionDocuments } from '../api/ragApi'

export function useDocumentSelection(sessionId) {
  const [documents, setDocuments] = useState([])   // full doc objects from API
  const [selectedIds, setSelectedIds] = useState([]) // document_id strings
  const [loading, setLoading] = useState(false)

  const fetchDocuments = useCallback(async (signal) => {
    if (!sessionId) {
      setDocuments([])
      setSelectedIds([])
      return
    }
    setLoading(true)
    try {
      const docs = await getSessionDocuments(sessionId, signal)
      if (signal?.aborted) return
      setDocuments(docs)
      // auto-select all ready docs on first load
      const readyIds = docs.filter(d => d.status === 'ready').map(d => d.document_id)
      setSelectedIds(readyIds)
    } catch {
      if (signal?.aborted) return
      setDocuments([])
      setSelectedIds([])
    } finally {
      if (!signal?.aborted) setLoading(false)
    }
  }, [sessionId])

  // Guarded against React StrictMode's dev-mode double-invoke the same way
  // as useSessions.js — the AbortController cancels the first, stale
  // request on the synthetic mount → cleanup → mount cycle.
  useEffect(() => {
    const controller = new AbortController()
    fetchDocuments(controller.signal)
    return () => controller.abort()
  }, [fetchDocuments])

  // Manual refresh (e.g. the refresh button in the doc picker) — no signal,
  // not called from a mount effect so there's nothing to race against.
  const refresh = useCallback(() => fetchDocuments(), [fetchDocuments])

  const toggleDocument = useCallback((documentId) => {
    setSelectedIds(prev => {
      const next = prev.includes(documentId)
        ? prev.filter(id => id !== documentId)
        : [...prev, documentId]
      return next.length ? next : prev // always keep at least one selected
    })
  }, [])

  const selectAll = useCallback(() => {
    setSelectedIds(documents.filter(d => d.status === 'ready').map(d => d.document_id))
  }, [documents])

  return { documents, selectedIds, toggleDocument, selectAll, refresh, loading }
}
