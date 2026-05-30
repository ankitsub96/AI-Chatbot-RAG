import { useState, useEffect } from 'react'
import { getDocuments } from '../api/ragApi'

const STORAGE_KEY = 'rag_session_docs'

function getStoredDocs(sessionId) {
  try {
    const map = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
    return map[sessionId] || null
  } catch { return null }
}

function storeDocs(sessionId, filenames) {
  try {
    const map = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
    map[sessionId] = filenames
    localStorage.setItem(STORAGE_KEY, JSON.stringify(map))
  } catch {}
}

export function useDocumentSelection(sessionId) {
  const [documents, setDocuments] = useState([])
  const [filenames, setFilenamesState] = useState([])

  useEffect(() => {
    getDocuments().then(docs => {
      setDocuments(docs)
      if (!docs.length) return

      const stored = sessionId ? getStoredDocs(sessionId) : null
      const valid = stored?.filter(f => docs.includes(f))
      const initial = (valid?.length) ? valid : [docs[0]]
      setFilenamesState(initial)
    }).catch(() => {})
  }, [sessionId])

  const setFilenames = (names) => {
    setFilenamesState(names)
    if (sessionId) storeDocs(sessionId, names)
  }

  const toggleFile = (name) => {
    const updated = filenames.includes(name)
      ? filenames.filter(f => f !== name)
      : [...filenames, name]
    const final = updated.length ? updated : [name] // always keep at least one
    setFilenames(final)
  }

  return { documents, filenames, setFilenames, toggleFile }
}