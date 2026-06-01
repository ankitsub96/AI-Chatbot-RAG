import { useState, useRef } from 'react'
import { uploadDocument } from '../api/ragApi'
import '../styles/Modal.scss'

export default function UploadModal({ onClose, onUploaded }) {
  const [file, setFile] = useState(null)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState('idle') // idle | uploading | done | error
  const [error, setError] = useState('')
  const inputRef = useRef(null)

  const onFileChange = (e) => {
    const f = e.target.files[0]
    if (f && f.type === 'application/pdf') {
      setFile(f)
      setError('')
    } else {
      setError('Only PDF files are supported')
    }
  }

  const onDrop = (e) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f && f.type === 'application/pdf') {
      setFile(f)
      setError('')
    } else {
      setError('Only PDF files are supported')
    }
  }

  const upload = async () => {
    if (!file) return
    setStatus('uploading')
    setProgress(0)
    try {
      await uploadDocument(file, setProgress)
      setStatus('done')
      setTimeout(() => {
        onUploaded()
        onClose()
      }, 800)
    } catch (err) {
      setStatus('error')
      setError(err.response?.data?.detail || 'Upload failed')
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal__header">
          <span className="modal__title">Upload Document</span>
          <button className="modal__close" onClick={onClose}>✕</button>
        </div>

        <div className="modal__body">
          <div
            className={`upload-zone ${file ? 'upload-zone--has-file' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDrop={onDrop}
            onDragOver={e => e.preventDefault()}
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
                <span className="upload-zone__icon">📄</span>
                <span className="upload-zone__name">{file.name}</span>
                <span className="upload-zone__size">
                  {(file.size / 1024 / 1024).toFixed(1)} MB
                </span>
              </>
            ) : (
              <>
                <span className="upload-zone__icon">⬆</span>
                <span className="upload-zone__text">Drop PDF here or click to browse</span>
              </>
            )}
          </div>

          {status === 'uploading' && (
            <div className="upload-progress">
              <div className="upload-progress__bar">
                <div
                  className="upload-progress__fill"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <span className="upload-progress__label">
                {progress < 100 ? `Uploading ${progress}%` : 'Processing...'}
              </span>
            </div>
          )}

          {status === 'done' && (
            <span className="modal__success">✓ Upload complete — indexing in background</span>
          )}

          {error && <span className="modal__error">{error}</span>}
        </div>

        <div className="modal__footer">
          <button className="modal__btn modal__btn--ghost" onClick={onClose}>Cancel</button>
          <button
            className="modal__btn modal__btn--primary"
            onClick={upload}
            disabled={!file || status === 'uploading' || status === 'done'}
          >
            {status === 'uploading' ? 'Uploading...' : 'Upload'}
          </button>
        </div>
      </div>
    </div>
  )
}
