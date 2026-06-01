import { useState, useEffect, useRef } from 'react'
import '../styles/Modal.scss'

export default function NewSessionModal({ onConfirm, onClose }) {
  const [name, setName] = useState('')
  const inputRef = useRef(null)

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const submit = () => {
    const id = name.trim() || `session-${Date.now()}`
    onConfirm(id)
  }

  const onKey = (e) => {
    if (e.key === 'Enter') submit()
    if (e.key === 'Escape') onClose()
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal__header">
          <span className="modal__title">New Session</span>
          <button className="modal__close" onClick={onClose}>✕</button>
        </div>
        <div className="modal__body">
          <label className="modal__label">Session name</label>
          <input
            ref={inputRef}
            className="modal__input"
            value={name}
            onChange={e => setName(e.target.value)}
            onKeyDown={onKey}
            placeholder="e.g. research-notes, project-alpha..."
            maxLength={48}
          />
          <span className="modal__hint">Leave blank to auto-generate</span>
        </div>
        <div className="modal__footer">
          <button className="modal__btn modal__btn--ghost" onClick={onClose}>Cancel</button>
          <button className="modal__btn modal__btn--primary" onClick={submit}>Create</button>
        </div>
      </div>
    </div>
  )
}
