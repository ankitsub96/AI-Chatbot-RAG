import { useState, useEffect, useRef } from 'react'
import '../styles/Modal.scss'

const PlusIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2" fill="none">
    <line x1="12" y1="5" x2="12" y2="19"/>
    <line x1="5" y1="12" x2="19" y2="12"/>
  </svg>
)

const XIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2" fill="none">
    <line x1="18" y1="6" x2="6" y2="18"/>
    <line x1="6" y1="6" x2="18" y2="18"/>
  </svg>
)

export default function NewSessionModal({ onConfirm, onClose }) {
  const [name, setName] = useState('')
  const [loading, setLoading] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => { inputRef.current?.focus() }, [])

  const submit = async () => {
    if (loading) return
    setLoading(true)
    await onConfirm(name.trim() || null)
    setLoading(false)
  }

  const onKey = (e) => {
    if (e.key === 'Enter') submit()
    if (e.key === 'Escape') onClose()
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal__header">
          <div className="modal__title-wrap">
            <div className="modal__title-icon"><PlusIcon /></div>
            <span className="modal__title">New Session</span>
          </div>
          <button className="modal__close" onClick={onClose}><XIcon /></button>
        </div>

        <div className="modal__body">
          <label className="modal__label">Session name</label>
          <input
            ref={inputRef}
            className="modal__input"
            value={name}
            onChange={e => setName(e.target.value)}
            onKeyDown={onKey}
            placeholder="e.g. research-notes, project-alpha…"
            maxLength={48}
            disabled={loading}
          />
          <span className="modal__hint">Leave blank to auto-generate</span>
        </div>

        <div className="modal__footer">
          <button className="modal__btn modal__btn--ghost" onClick={onClose}>Cancel</button>
          <button
            className="modal__btn modal__btn--primary"
            onClick={submit}
            disabled={loading}
          >
            <PlusIcon />
            {loading ? 'Creating…' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  )
}
