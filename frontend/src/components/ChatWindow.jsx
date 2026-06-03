import { useState, useEffect, useRef } from 'react'
import { useDocumentSelection } from '../hooks/useDocumentSelection'
import { useChat } from '../hooks/useChat'
import MessageList from './MessageList'
import ChatInput from './ChatInput'
import '../styles/ChatWindow.scss'

const FileIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2" fill="none">
    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
    <polyline points="14 2 14 8 20 8" />
  </svg>
)

const ChevronIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2" fill="none">
    <polyline points="6 9 12 15 18 9" />
  </svg>
)

const RefreshIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2" fill="none">
    <polyline points="23 4 23 10 17 10" />
    <path d="M20.49 15a9 9 0 11-2.12-9.36L23 10" />
  </svg>
)

export default function ChatWindow({ sessionId, sessions, loadHistory }) {
  const [streaming, setStreaming] = useState(true)
  const [docOpen, setDocOpen] = useState(false)
  const docRef = useRef(null)

  const { documents, selectedIds, toggleDocument, selectAll, refresh: refreshDocs, loading: docsLoading } = useDocumentSelection(sessionId)
  const { messages, loading, streamingMsg, thinkingSteps, sessionName, isThinking, send, setMessages } = useChat(
    sessionId, sessions, selectedIds, streaming
  )

  // close dropdown on outside click
  useEffect(() => {
    const handler = (e) => {
      if (docRef.current && !docRef.current.contains(e.target)) setDocOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  // load history when session changes
  useEffect(() => {
    if (!sessionId) { setMessages([]); return }
    loadHistory(sessionId).then(items => {
      setMessages((items || []).map(item => ([
        { role: 'user', content: item.question, ts: new Date(item.created_at || Date.now()).getTime() },
        { role: 'assistant', content: item.answer, ts: new Date(item.created_at || Date.now()).getTime() + 1 },
      ])).flat())
    })
  }, [sessionId])

  if (!sessionId) return (
    <div className="chat-window chat-window--empty">
      <div className="empty-glyph">◈</div>
      <p>Select or create a session<br />to start querying</p>
      <span className="empty-hint">← Use the sidebar</span>
    </div>
  )

  const readyDocs = documents.filter(d => d.status === 'ready')
  const selectedCount = selectedIds.length

  return (
    <div className="chat-window">
      <div className="chat-window__header">

        {/* Document picker */}
        <div className="chat-window__doc-picker" ref={docRef}>
          <button
            className={`chat-window__doc-trigger ${docOpen ? 'open' : ''}`}
            onClick={() => setDocOpen(o => !o)}
          >
            <FileIcon />
            {selectedCount > 0
              ? <><span className="chat-window__doc-badge">{selectedCount}</span> doc{selectedCount !== 1 ? 's' : ''}</>
              : 'No docs'
            }
            <ChevronIcon />
          </button>

          {docOpen && (
            <div className="chat-window__doc-dropdown">
              <div className="chat-window__doc-dropdown-header">
                Documents in session
                <button
                  style={{ float: 'right', color: 'var(--accent)', fontSize: '10px', fontFamily: 'inherit', cursor: 'pointer', background: 'none', border: 'none' }}
                  onClick={refreshDocs}
                >
                  <RefreshIcon />
                </button>
              </div>

              {documents.length === 0 && (
                <div className="chat-window__doc-dropdown-empty">
                  {docsLoading ? 'Loading…' : 'No documents uploaded yet'}
                </div>
              )}

              {documents.map(doc => (
                <label
                  key={doc.document_id}
                  className={`chat-window__doc-item ${selectedIds.includes(doc.document_id) ? 'selected' : ''}`}
                >
                  <input
                    type="checkbox"
                    checked={selectedIds.includes(doc.document_id)}
                    onChange={() => toggleDocument(doc.document_id)}
                    disabled={doc.status !== 'ready'}
                  />
                  <span title={doc.original_filename}>{doc.original_filename}</span>
                  <span className={`chat-window__doc-status chat-window__doc-status--${doc.status}`}>
                    {doc.status}
                  </span>
                </label>
              ))}
            </div>
          )}
        </div>

        <div className="chat-window__header-meta">
          <span className="chat-window__session-tag">{sessionName}</span>
        </div>
      </div>

      <MessageList
        messages={messages}
        streamingMsg={streamingMsg}
        loading={loading}
        thinkingSteps={thinkingSteps}
        isThinking={isThinking}
      />

      <ChatInput
        onSend={send}
        loading={loading}
        streaming={streaming}
        onToggleStream={() => setStreaming(s => !s)}
      />
    </div>
  )
}
