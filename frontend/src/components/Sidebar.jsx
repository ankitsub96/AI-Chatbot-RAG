import { useState } from 'react'
import '../styles/Sidebar.scss'

export default function Sidebar({
  sessions, activeId, onSelect, onCreate, onDelete
}) {
  const [hoverId, setHoverId] = useState(null)

  return (
    <aside className="sidebar">
      <div className="sidebar__header">
        <span className="sidebar__logo">RAG<em>Chat</em></span>
        <button className="sidebar__new" onClick={onCreate} title="New session">+</button>
      </div>

      <div className="sidebar__sessions">
        {sessions.length === 0 && (
          <p className="sidebar__empty">No sessions yet</p>
        )}
        {sessions.map(id => (
          <div
            key={id}
            className={`sidebar__item ${id === activeId ? 'active' : ''}`}
            onClick={() => onSelect(id)}
            onMouseEnter={() => setHoverId(id)}
            onMouseLeave={() => setHoverId(null)}
          >
            <span className="sidebar__item-label">
              {id.replace('session-', '#')}
            </span>
            {hoverId === id && (
              <button
                className="sidebar__delete"
                onClick={e => { e.stopPropagation(); onDelete(id) }}
                title="Delete"
              >×</button>
            )}
          </div>
        ))}
      </div>
    </aside>
  )
}
