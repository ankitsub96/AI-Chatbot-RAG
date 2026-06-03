import { useState, useEffect, useRef } from 'react'
import '../styles/MessageList.scss'

export default function ThinkingDrawer({ steps, isThinking }) {
  const [open, setOpen] = useState(false)
  const stepsRef = useRef(null)

  // auto-open when thinking starts
  useEffect(() => {
    if (isThinking && steps.length > 0) setOpen(true)
  }, [isThinking, steps.length])

  // auto-scroll steps
  useEffect(() => {
    if (stepsRef.current) {
      stepsRef.current.scrollTop = stepsRef.current.scrollHeight
    }
  }, [steps])

  if (steps.length === 0 && !isThinking) return null

  return (
    <div className="thinking-drawer">
      <button
        className={`thinking-drawer__toggle ${open ? 'open' : ''}`}
        onClick={() => setOpen(o => !o)}
      >
        {isThinking ? (
          <span className="thinking-drawer__spinner" />
        ) : (
          <svg viewBox="0 0 24 24" strokeWidth="2">
            <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
          </svg>
        )}
        <span className="thinking-drawer__label">
          {isThinking ? 'Thinking…' : 'Reasoning steps'}
        </span>
        <span className="thinking-drawer__count">{steps.length}</span>
        <span className="thinking-drawer__chevron-wrap">
          <svg className="chevron" viewBox="0 0 24 24" strokeWidth="2">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </span>
      </button>

      <div className={`thinking-drawer__body ${open ? 'open' : ''}`}>
        <div className="thinking-drawer__steps" ref={stepsRef}>
          {steps.map((step, i) => (
            <div key={i} className="thinking-drawer__step">
              <span className="thinking-drawer__step-dot" />
              <div className="thinking-drawer__step-content">
                <div className="thinking-drawer__step-event">{step.event}</div>
                <div className="thinking-drawer__step-msg">{step.message}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
