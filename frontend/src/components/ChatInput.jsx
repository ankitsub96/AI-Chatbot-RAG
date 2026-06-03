import { useState } from 'react'
import '../styles/ChatInput.scss'

const SendIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2" fill="none">
    <line x1="22" y1="2" x2="11" y2="13"/>
    <polygon points="22 2 15 22 11 13 2 9 22 2"/>
  </svg>
)

export default function ChatInput({ onSend, loading, streaming, onToggleStream }) {
  const [value, setValue] = useState('')

  const submit = () => {
    if (!value.trim() || loading) return
    onSend(value.trim())
    setValue('')
  }

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="chat-input">
      <div className="chat-input__toolbar">
        <label className="chat-input__toggle">
          <input type="checkbox" checked={streaming} onChange={onToggleStream} />
          Stream response
        </label>
        <span className="chat-input__hint">
          <kbd>Enter</kbd> send · <kbd>Shift+Enter</kbd> newline
        </span>
      </div>

      <div className="chat-input__row">
        <textarea
          className="chat-input__area"
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={onKey}
          placeholder="Enter your query…"
          rows={1}
          disabled={loading}
        />
        <button
          className="chat-input__send"
          onClick={submit}
          disabled={loading || !value.trim()}
        >
          <SendIcon />
        </button>
      </div>
    </div>
  )
}
