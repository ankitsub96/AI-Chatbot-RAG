import { useState, useEffect, useCallback } from 'react'
import { getSessionDocuments } from '../api/ragApi'

export function useDocumentSelection(sessionId) {
  const [documents, setDocuments] = useState([])   // full doc objects from API
  const [selectedIds, setSelectedIds] = useState([]) // document_id strings
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async () => {
    if (!sessionId) { setDocuments([]); setSelectedIds([]); return }
    setLoading(true)
    try {
      const docs = await getSessionDocuments(sessionId)
      setDocuments(docs)
      // auto-select all ready docs on first load
      const readyIds = docs.filter(d => d.status === 'ready').map(d => d.document_id)
      setSelectedIds(readyIds)
    } catch {
      setDocuments([])
      setSelectedIds([])
    } finally {
      setLoading(false)
    }
  }, [sessionId])

  useEffect(() => { refresh() }, [refresh])

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
