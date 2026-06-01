import { useState } from 'react'
import NewSessionModal from './NewSessionModal'
import UploadModal from './UploadModal'
import '../styles/Sidebar.scss'

export default function Sidebar({ sessions, activeId, onSelect, onCreate, onDelete, onUploaded }) {
  const [hoverId, setHoverId] = useState(null)
  const [showNewSession, setShowNewSession] = useState(false)
  const [showUpload, setShowUpload] = useState(false)

  const handleCreate = (id) => {
    onCreate(id)
    setShowNewSession(false)
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
          >↑</button>
          <button
            className="sidebar__new"
            onClick={() => setShowNewSession(true)}
            title="New session"
          >+</button>
        </div>
      </div>

      <div className="sidebar__label">Sessions</div>

      <div className="sidebar__sessions">
        {sessions.length === 0 && (
          <p className="sidebar__empty">No sessions yet.<br />Create one to begin.</p>
        )}
        {sessions.map(id => (
          <div
            key={id}
            className={`sidebar__item ${id === activeId ? 'active' : ''}`}
            onClick={() => onSelect(id)}
            onMouseEnter={() => setHoverId(id)}
            onMouseLeave={() => setHoverId(null)}
          >
            <span className="sidebar__item-icon">§</span>
            <span className="sidebar__item-label">{id}</span>
            {hoverId === id && (
              <button
                className="sidebar__delete"
                onClick={e => { e.stopPropagation(); onDelete(id) }}
                title="Delete"
              >✕</button>
            )}
          </div>
        ))}
      </div>

      <div className="sidebar__footer">RAG · v2 · local</div>

      {showNewSession && (
        <NewSessionModal
          onConfirm={handleCreate}
          onClose={() => setShowNewSession(false)}
        />
      )}

      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onUploaded={() => { onUploaded?.(); setShowUpload(false) }}
        />
      )}
    </aside>
  )
}
