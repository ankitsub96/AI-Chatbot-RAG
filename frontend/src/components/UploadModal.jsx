import { useState, useRef } from 'react'
import { uploadDocument } from '../api/ragApi'
import '../styles/Modal.scss'

const UploadIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2" fill="none">
    <polyline points="16 16 12 12 8 16" />
    <line x1="12" y1="12" x2="12" y2="21" />
    <path d="M20.39 18.39A5 5 0 0018 9h-1.26A8 8 0 103 16.3" />
  </svg>
)

const FileIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2" fill="none">
    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
    <polyline points="14 2 14 8 20 8" />
  </svg>
)

const XIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2" fill="none">
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>
)

const CheckIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2.5" fill="none">
    <polyline points="20 6 9 17 4 12" />
  </svg>
)

const AlertIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2" fill="none">
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="8" x2="12" y2="12" />
    <line x1="12" y1="16" x2="12.01" y2="16" />
  </svg>
)

export default function UploadModal({ sessionId, onClose, onUploaded }) {
  const [file, setFile] = useState(null)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState('idle') // idle | uploading | done | error
  const [error, setError] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef(null)

  const acceptFile = (f) => {
    if (f && f.type === 'application/pdf') {
      setFile(f)
      setError('')
      setStatus('idle')
    } else {
      setError('Only PDF files are supported')
    }
  }

  const onFileChange = (e) => acceptFile(e.target.files[0])

  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    acceptFile(e.dataTransfer.files[0])
  }

  const upload = async () => {
    if (!file || !sessionId) return
    setStatus('uploading')
    setProgress(0)
    try {
      const result = await uploadDocument(file, sessionId, setProgress)
      setStatus('done')
      setTimeout(() => {
        if (onUploaded) onUploaded(result)
        onClose()
      }, 900)
    } catch (err) {
      setStatus('error')
      setError(err.response?.data?.detail || 'Upload failed')
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal__header">
          <div className="modal__title-wrap">
            <div className="modal__title-icon"><UploadIcon /></div>
            <span className="modal__title">Upload Document</span>
          </div>
          <button className="modal__close" onClick={onClose}><XIcon /></button>
        </div>

        <div className="modal__body">
          <div
            className={`upload-zone ${file ? 'upload-zone--has-file' : ''} ${dragOver ? 'upload-zone--drag-over' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDrop={onDrop}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".pdf"
              style={{ display: 'none' }}
              onChange={onFileChange}
            />
            {file ? (
              <>
                <span className="upload-zone__icon"><FileIcon /></span>
                <span className="upload-zone__name">{file.name}</span>
                <span className="upload-zone__size">
                  {(file.size / 1024 / 1024).toFixed(1)} MB
                </span>
              </>
            ) : (
              <>
                <span className="upload-zone__icon"><UploadIcon /></span>
                <span className="upload-zone__text">Drop PDF here or click to browse</span>
                <span className="upload-zone__sub">PDF files only · max 50MB</span>
              </>
            )}
          </div>

          {status === 'uploading' && (
            <div className="upload-progress">
              <div className="upload-progress__bar">
                <div className="upload-progress__fill" style={{ width: `${progress}%` }} />
              </div>
              <span className="upload-progress__label">
                <span className="upload-progress__spinner" />
                {progress < 100 ? `Uploading ${progress}%` : 'Processing…'}
              </span>
            </div>
          )}

          {status === 'done' && (
            <span className="modal__success">
              <CheckIcon />
              Upload complete — indexing in background
            </span>
          )}

          {error && (
            <span className="modal__error">
              <AlertIcon />
              {error}
            </span>
          )}

          {!sessionId && (
            <span className="modal__error">
              <AlertIcon />
              No active session — create a session first
            </span>
          )}
        </div>

        <div className="modal__footer">
          <button className="modal__btn modal__btn--ghost" onClick={onClose}>Cancel</button>
          <button
            className="modal__btn modal__btn--primary"
            onClick={upload}
            disabled={!file || !sessionId || status === 'uploading' || status === 'done'}
          >
            <UploadIcon />
            {status === 'uploading' ? 'Uploading…' : 'Upload'}
          </button>
        </div>
      </div>
    </div>
  )
}
