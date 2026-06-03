import { useEffect, useRef } from 'react'
import ThinkingDrawer from './ThinkingDrawer'
import '../styles/MessageList.scss'

function formatTime(ts) {
  return new Date(ts || Date.now()).toLocaleTimeString('en-US', {
    hour: '2-digit', minute: '2-digit', hour12: false
  })
}

const UserIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2">
    <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/>
    <circle cx="12" cy="7" r="4"/>
  </svg>
)

const BotIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2">
    <rect x="3" y="11" width="18" height="10" rx="2"/>
    <circle cx="12" cy="5" r="2"/>
    <path d="M12 7v4M8 15h.01M16 15h.01"/>
  </svg>
)

export default function MessageList({ messages, streamingMsg, loading, thinkingSteps, isThinking }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingMsg, thinkingSteps])

  return (
    <div className="messages-wrap">
      <ThinkingDrawer steps={thinkingSteps} isThinking={isThinking} />

      <div className="messages">
        {messages.length === 0 && !loading && !streamingMsg && (
          <div className="messages-empty">
            <div className="messages-empty__glyph">◈</div>
            <div className="messages-empty__text">Awaiting query</div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message message--${msg.role}`}>
            <div className="message__meta">
              <div className="message__avatar">
                {msg.role === 'user' ? <UserIcon /> : <BotIcon />}
              </div>
              <span className="message__role">
                {msg.role === 'user' ? 'Researcher' : 'Archive'}
              </span>
              <span className="message__time">{formatTime(msg.ts)}</span>
            </div>
            <p className="message__content">{msg.content}</p>
          </div>
        ))}

        {streamingMsg && (
          <div className="message message--assistant">
            <div className="message__meta">
              <div className="message__avatar"><BotIcon /></div>
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
              <div className="message__avatar"><BotIcon /></div>
              <span className="message__role">Archive</span>
            </div>
            <div className="message__dots">
              <span /><span /><span />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  )
}
