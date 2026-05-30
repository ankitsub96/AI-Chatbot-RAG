import { useEffect, useRef } from 'react'
import '../styles/MessageList.scss'

export default function MessageList({ messages, streamingMsg, loading }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingMsg])

  return (
    <div className="messages">
      {messages.map((msg, i) => (
        <div key={i} className={`message message--${msg.role}`}>
          <span className="message__role">{msg.role === 'user' ? 'You' : 'AI'}</span>
          <p className="message__content">{msg.content}</p>
        </div>
      ))}

      {streamingMsg && (
        <div className="message message--assistant">
          <span className="message__role">AI</span>
          <p className="message__content">
            {streamingMsg}
            <span className="message__cursor" />
          </p>
        </div>
      )}

      {loading && !streamingMsg && (
        <div className="message message--assistant message--loading">
          <span className="message__role">AI</span>
          <div className="message__dots">
            <span /><span /><span />
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}
