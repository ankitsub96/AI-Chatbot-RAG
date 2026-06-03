import { useState } from 'react'
import NewSessionModal from './NewSessionModal'
import UploadModal from './UploadModal'
import ThemePicker from './ThemePicker'
import '../styles/Sidebar.scss'

const UploadIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2" fill="none">
    <polyline points="16 16 12 12 8 16"/>
    <line x1="12" y1="12" x2="12" y2="21"/>
    <path d="M20.39 18.39A5 5 0 0018 9h-1.26A8 8 0 103 16.3"/>
  </svg>
)

const PlusIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2.5" fill="none">
    <line x1="12" y1="5" x2="12" y2="19"/>
    <line x1="5" y1="12" x2="19" y2="12"/>
  </svg>
)

const SessionIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2" fill="none">
    <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
  </svg>
)

const TrashIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2" fill="none">
    <polyline points="3 6 5 6 21 6"/>
    <path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/>
    <path d="M10 11v6M14 11v6"/>
  </svg>
)

const PaletteIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2" fill="none">
    <circle cx="13.5" cy="6.5" r=".5" fill="currentColor"/>
    <circle cx="17.5" cy="10.5" r=".5" fill="currentColor"/>
    <circle cx="8.5" cy="7.5" r=".5" fill="currentColor"/>
    <circle cx="6.5" cy="12.5" r=".5" fill="currentColor"/>
    <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 011.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.554C21.965 6.012 17.461 2 12 2z"/>
  </svg>
)

function getSessionLabel(session) {
  if (typeof session === 'string') return session
  return session.title || session.session_id?.slice(0, 16) || 'Untitled'
}

function getSessionId(session) {
  if (typeof session === 'string') return session
  return session.session_id || session.id || session
}

function getSessionTime(session) {
  if (typeof session === 'string') return null
  if (!session.created_at) return null
  return new Date(session.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

export default function Sidebar({
  sessions, activeId, onSelect, onCreate, onDelete, onUploaded,
  theme, themes, onThemeChange,
}) {
  const [hoverId, setHoverId] = useState(null)
  const [showNewSession, setShowNewSession] = useState(false)
  const [showUpload, setShowUpload] = useState(false)
  const [showTheme, setShowTheme] = useState(false)

  const handleCreate = async (name) => {
    const session = await onCreate(name)
    setShowNewSession(false)
    return session
  }

  return (
    <aside className="sidebar">
      <div className="sidebar__header">
        <span className="sidebar__logo">RAG<em>atouille</em></span>
        <div className="sidebar__header-actions">
          <button
            className="sidebar__action"
            onClick={() => setShowUpload(true)}
            title="Upload document"
          >
            <UploadIcon />
          </button>
          <button
            className="sidebar__new"
            onClick={() => setShowNewSession(true)}
            title="New session"
          >
            <PlusIcon />
          </button>
        </div>
      </div>

      <div className="sidebar__section">
        <span className="sidebar__label">Sessions</span>
      </div>

      <div className="sidebar__sessions">
        {sessions.length === 0 && (
          <p className="sidebar__empty">
            <strong>No sessions yet</strong>
            Create one to start querying documents.
          </p>
        )}
        {sessions.map(session => {
          const id = getSessionId(session)
          const label = getSessionLabel(session)
          const time = getSessionTime(session)
          return (
            <div
              key={id}
              className={`sidebar__item ${id === activeId ? 'active' : ''}`}
              onClick={() => onSelect(id)}
              onMouseEnter={() => setHoverId(id)}
              onMouseLeave={() => setHoverId(null)}
            >
              <span className="sidebar__item-icon"><SessionIcon /></span>
              <span className="sidebar__item-label">{label}</span>
              {time && <span className="sidebar__item-time">{time}</span>}
              {hoverId === id && (
                <button
                  className="sidebar__delete"
                  onClick={e => { e.stopPropagation(); onDelete(id) }}
                  title="Delete session"
                >
                  <TrashIcon />
                </button>
              )}
            </div>
          )
        })}
      </div>

      <div className="sidebar__footer">
        <div className="sidebar__status">
          <span className="sidebar__status-dot" />
          live
        </div>
        <button
          className="sidebar__theme-btn"
          onClick={() => setShowTheme(true)}
          title="Change theme"
        >
          <PaletteIcon />
          {themes?.find(t => t.id === theme)?.name || 'Theme'}
        </button>
      </div>

      {showNewSession && (
        <NewSessionModal
          onConfirm={handleCreate}
          onClose={() => setShowNewSession(false)}
        />
      )}

      {showUpload && (
        <UploadModal
          sessionId={activeId}
          onClose={() => setShowUpload(false)}
          onUploaded={(result) => { onUploaded?.(result); setShowUpload(false) }}
        />
      )}

      {showTheme && (
        <ThemePicker
          theme={theme}
          themes={themes || []}
          onSelect={onThemeChange}
          onClose={() => setShowTheme(false)}
        />
      )}
    </aside>
  )
}
