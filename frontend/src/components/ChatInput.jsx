import { useState } from 'react'
import '../styles/ChatInput.scss'

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
          <input
            type="checkbox"
            checked={streaming}
            onChange={onToggleStream}
          />
          <span>Stream</span>
        </label>
      </div>

      <div className="chat-input__row">
        <textarea
          className="chat-input__area"
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={onKey}
          placeholder="Ask about the document..."
          rows={1}
          disabled={loading}
        />
        <button
          className="chat-input__send"
          onClick={submit}
          disabled={loading || !value.trim()}
        >
          {loading ? '…' : '↑'}
        </button>
      </div>
    </div>
  )
}
