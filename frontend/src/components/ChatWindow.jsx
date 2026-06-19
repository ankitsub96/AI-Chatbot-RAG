import { useState, useEffect, useRef } from 'react'
import { useDocumentSelection } from '../hooks/useDocumentSelection'
import { useChat } from '../hooks/useChat'
import { useTheme } from '../hooks/useTheme'
import {
  ENDPOINTS,
  DEFAULT_ENDPOINT_ID,
  DEFAULT_STRICTNESS,
  getEndpointConfig,
} from '../config/endpoints'
import MessageList from './MessageList'
import ChatInput from './ChatInput'
import UploadModal from './UploadModal'
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
const UploadIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2" fill="none">
    <polyline points="16 16 12 12 8 16" />
    <line x1="12" y1="12" x2="12" y2="21" />
    <path d="M20.39 18.39A5 5 0 0018 9h-1.26A8 8 0 103 16.3" />
  </svg>
)
export default function ChatWindow({ sessionId,
  activeId,
  sessions,
  loadHistory,
  theme,
  themes,
  setTheme, }) {
  const [streaming, setStreaming] = useState(
    () => localStorage.getItem('rag.streaming') !== 'false'
  )
  const [endpointId, setEndpointId] = useState(
    () => localStorage.getItem('rag.endpoint') || DEFAULT_ENDPOINT_ID
  )
  const [strictness, setStrictness] = useState(
    () => localStorage.getItem('rag.strictness') || DEFAULT_STRICTNESS
  )
  const [useWeb, setUseWeb] = useState(
    () => localStorage.getItem('rag.useWeb') === 'true'
  )
  const [docOpen, setDocOpen] = useState(false)
  const docRef = useRef(null)
  const [showUpload, setShowUpload] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const settingsRef = useRef(null)
  // const [appearance, setAppearance] = useState('light')
  const { documents, selectedIds, toggleDocument, selectAll, refresh: refreshDocs, loading: docsLoading } = useDocumentSelection(sessionId)
  const {
    currentTheme,
    mode,
    setMode, appearance,
    // setTheme,
    setAppearance,
  } = useTheme()
  const {
    messages,
    loading,
    sessionName,
    send,
    setMessages,
  } = useChat(
    sessionId,
    sessions,
    selectedIds,
    streaming,
    endpointId,
    strictness,
    useWeb,
  )
  // close dropdown on outside click
  useEffect(() => {
    const handler = (e) => {
      if (
        docRef.current &&
        !docRef.current.contains(e.target)
      ) {
        setDocOpen(false)
      }
      if (
        settingsRef.current &&
        !settingsRef.current.contains(e.target)
      ) {
        setShowSettings(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () =>
      document.removeEventListener(
        'mousedown',
        handler
      )
  }, [])
  // Load history when session changes. Guarded with an AbortController so
  // React StrictMode's dev-only mount → cleanup → mount cycle cancels the
  // first, stale request instead of letting both land (this — plus the same
  // fix in useSessions.js and useDocumentSelection.js — is what was causing
  // every API call to fire twice on page load/refresh). It also protects
  // against a real race: rapidly switching sessions could otherwise let an
  // older session's history response overwrite a newer one's.
  useEffect(() => {
    if (!sessionId) { setMessages([]); return }
    const controller = new AbortController()
    loadHistory(sessionId, controller.signal).then(items => {
      if (controller.signal.aborted) return
      setMessages(
        (items || [])
          .map(item => ([
            {
              role: 'user',
              content: item.question,
              ts: new Date(item.created_at || Date.now()).getTime(),
            },
            {
              role: 'assistant',
              content: item.answer,
              ts: new Date(item.created_at || Date.now()).getTime() + 1,
              thoughts: item.thoughts || [],
              hasThoughts: Array.isArray(item.thoughts) && item.thoughts.length > 0,
              isThinking: false,
              endpoint: item.endpoint || null,
              strictness: item.strictness || null,
            },
          ]))
          .flat()
      )
    })
    return () => controller.abort()
  }, [sessionId, loadHistory, setMessages])
  useEffect(() => {
    localStorage.setItem('rag.endpoint', endpointId)
  }, [endpointId])
  useEffect(() => {
    localStorage.setItem('rag.strictness', strictness)
  }, [strictness])
  useEffect(() => {
    localStorage.setItem('rag.useWeb', String(useWeb))
  }, [useWeb])
  useEffect(() => {
    localStorage.setItem('rag.streaming', String(streaming))
  }, [streaming])
  useEffect(() => {
    const config = getEndpointConfig(endpointId)
    if (!config?.supportsWeb) {
      setUseWeb(false)
    }
    if (!config?.supportsStrictness) {
      setStrictness(DEFAULT_STRICTNESS)
    }
    if (!config?.supportsStream) {
      setStreaming(false)
    }
  }, [endpointId])
  if (!sessionId) return (
    <div className="chat-window chat-window--empty">
      <div className="empty-glyph">◈</div>
      <p>Select or create a session<br />to start querying</p>
      <span className="empty-hint">← Use the sidebar</span>
    </div>
  )
  const readyDocs = documents.filter(d => d.status === 'ready')
  const selectedCount = selectedIds.length
  const endpointConfig = getEndpointConfig(endpointId)
  return (
    <div className="chat-window">
      <div className="chat-window__header">
        <div className="chat-window__header-left">
          <div
            className="chat-window__doc-picker"
            ref={docRef}
          >
            <button
              className={`chat-window__doc-trigger ${docOpen ? 'open' : ''
                }`}
              onClick={() => setDocOpen(o => !o)}
            >
              <FileIcon />
              {selectedCount > 0
                ? (
                  <>
                    <span className="chat-window__doc-badge">
                      {selectedCount}
                    </span>
                    docs
                  </>
                )
                : 'Documents'}
              <ChevronIcon />
            </button>
            {docOpen && (
              <div className="chat-window__doc-dropdown">
                <div className="chat-window__doc-dropdown-header">
                  Documents
                  <button
                    onClick={refreshDocs}
                    className="chat-window__refresh-btn"
                  >
                    <RefreshIcon />
                  </button>
                </div>
                {documents.length === 0 && (
                  <div className="chat-window__doc-dropdown-empty">
                    {docsLoading
                      ? 'Loading...'
                      : 'No documents'}
                  </div>
                )}
                {documents.map(doc => (
                  <label
                    key={doc.document_id}
                    className={`chat-window__doc-item ${selectedIds.includes(
                      doc.document_id
                    )
                      ? 'selected'
                      : ''
                      }`}
                  >
                    <input
                      type="checkbox"
                      checked={selectedIds.includes(
                        doc.document_id
                      )}
                      onChange={() =>
                        toggleDocument(
                          doc.document_id
                        )
                      }
                      disabled={
                        doc.status !== 'ready'
                      }
                    />
                    <span
                      title={
                        doc.original_filename
                      }
                    >
                      {doc.original_filename}
                    </span>
                    <span
                      className={`chat-window__doc-status chat-window__doc-status--${doc.status}`}
                    >
                      {doc.status}
                    </span>
                  </label>
                ))}
              </div>
            )}
          </div>
          <button
            className="header-upload"
            onClick={() => setShowUpload(true)}
            title="Upload document"
          >
            <UploadIcon />
            Upload
          </button>
        </div>
        <div className="chat-window__header-center">
          <span className="chat-window__session-tag">
            {sessionName}
          </span>
        </div>
        <div className="chat-window__header-right">
          <div
            className="chat-window__settings"
            ref={settingsRef}
          >
            <button
              className="settings-trigger"
              onClick={() => setShowSettings(v => !v)}
            >
              ⚙
            </button>
            {showSettings && (
              <div className="settings-menu">
                <div className="settings-menu__section">
                  <div className="settings-menu__title">
                    Endpoint
                  </div>
                  {ENDPOINTS.map(endpoint => (
                    <label
                      key={endpoint.id}
                      className="settings-menu__radio"
                    >
                      <input
                        type="radio"
                        checked={
                          endpointId ===
                          endpoint.id
                        }
                        onChange={() =>
                          setEndpointId(
                            endpoint.id
                          )
                        }
                      />
                      <span>
                        {endpoint.label}
                      </span>
                    </label>
                  ))}
                </div>
                {endpointConfig?.supportsStrictness && (
                  <div className="settings-menu__section">
                    <div className="settings-menu__title">
                      Retrieval
                    </div>
                    {[
                      'strict',
                      'balanced',
                      'creative',
                    ].map(mode => (
                      <label
                        key={mode}
                        className="settings-menu__radio"
                      >
                        <input
                          type="radio"
                          checked={
                            strictness ===
                            mode
                          }
                          onChange={() =>
                            setStrictness(
                              mode
                            )
                          }
                        />
                        <span>
                          {mode}
                        </span>
                      </label>
                    ))}
                  </div>
                )}
                {endpointConfig?.supportsWeb && (
                  <label className="settings-menu__toggle">
                    <span>
                      Web Search
                    </span>
                    <input
                      type="checkbox"
                      checked={useWeb}
                      onChange={e =>
                        setUseWeb(
                          e.target.checked
                        )
                      }
                    />
                  </label>
                )}
                <div className="settings-section" style={{
                  padding: '0 1rem', gap: '1rem'
                }}>
                  <div className="settings-title">Theme</div>

                  <div className="theme-swatches">
                    {themes.map(theme => (
                      <button
                        key={theme.id}
                        className={`theme-swatch ${currentTheme === theme.id ? 'active' : ''
                          }`}
                        style={{ background: theme.swatch }}
                        onClick={() => setTheme(theme.id)}
                      />
                    ))}
                  </div>
                </div>
                <div className="settings-section" style={{
                  display: 'flex', padding: '1rem', gap: '1rem'
                }}>
                  {/* <div className="settings-title">Dark/Light Mode</div> */}
                  <button
                    className={appearance === 'dark' ? 'appearance-btn active' : 'appearance-btn '}
                    onClick={() => setAppearance('dark')}
                    style={{ background: '#080810', color: '#e8e8f0' }}
                  >
                    Dark
                  </button>

                  <button
                    className={appearance === 'light' ? 'appearance-btn active' : 'appearance-btn '}
                    onClick={() => setAppearance('light')}
                    style={{ background: '#f8fafc', color: '#111827' }}
                  >
                    Light
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      <MessageList
        messages={messages}
        loading={loading}
      />
      <ChatInput
        onSend={send}
        loading={loading}
        streaming={streaming}
        onToggleStream={() => {
          if (endpointConfig?.supportsStream) {
            setStreaming(s => !s)
          }
        }}
      />
      {showUpload && (
        <UploadModal
          sessionId={activeId}
          onClose={() => setShowUpload(false)}
          onUploaded={(result) => { onUploaded?.(result); setShowUpload(false) }}
        />
      )}
    </div>
  )
}
