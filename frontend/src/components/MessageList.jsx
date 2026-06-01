import { useEffect, useRef } from 'react'
import '../styles/MessageList.scss'

function formatTime() {
  return new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })
}

export default function MessageList({ messages, streamingMsg, loading }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingMsg])

  if (messages.length === 0 && !loading && !streamingMsg) {
    return (
      <div className="messages" style={{ justifyContent: 'center', alignItems: 'center' }}>
        <div className="messages-empty">
          <div className="messages-empty__glyph">✦</div>
          <div className="messages-empty__text">Awaiting query</div>
        </div>
      </div>
    )
  }

  return (
    <div className="messages">
      {messages.map((msg, i) => (
        <div key={i} className={`message message--${msg.role}`}>
          <div className="message__meta">
            <span className="message__role">{msg.role === 'user' ? 'Researcher' : 'Archive'}</span>
            <span className="message__time">{formatTime()}</span>
          </div>
          <p className="message__content">{msg.content}</p>
        </div>
      ))}

      {streamingMsg && (
        <div className="message message--assistant">
          <div className="message__meta">
            <span className="message__role">Archive</span>
          </div>
          <p className="message__content">
            {streamingMsg}
            <span className="message__cursor" />
          </p>
        </div>
      )}

      {loading && !streamingMsg && (
        <div className="message message--assistant message--loading">
          <div className="message__meta">
            <span className="message__role">Archive</span>
          </div>
          <div className="message__dots">
            <span /><span /><span />
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}
