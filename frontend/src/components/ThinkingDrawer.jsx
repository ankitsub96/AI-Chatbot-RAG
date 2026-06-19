import { useState, useEffect, useRef } from 'react'
import '../styles/MessageList.scss'
const STAGE_LABELS = {
  embedding: 'Creating Embedding',
  embedding_done: 'Embedding Ready',

  cache: 'Checking Cache',
  cache_check: 'Checking Cache',
  cache_miss: 'Cache Miss',

  memory_search: 'Searching Memory',
  memory_retrieval: 'Retrieving Memory',
  memory_done: 'Memory Retrieved',

  retrieval: 'Searching Documents',

  react: 'Reasoning',

  planner: 'Building Plan',

  answer_generation: 'Generating Answer',

  evaluation: 'Evaluating Answer',

  save_and_return: 'Saving Conversation',

  llm: 'Generating Response',
}
const STAGE_ICONS = {
  embedding: '🔢',
  embedding_done: '✅',

  cache: '🔎',
  cache_check: '🔎',
  cache_miss: '⚠️',

  memory_search: '🧠',
  memory_retrieval: '🧠',
  memory_done: '✅',

  retrieval: '📚',

  react: '🤔',

  planner: '🗺️',

  answer_generation: '✍️',

  evaluation: '📊',

  save_and_return: '💾',

  llm: '✨',
}
export default function ThinkingDrawer({ steps, isThinking }) {
  const [open, setOpen] = useState(isThinking)
  const [expandedStep, setExpandedStep] = useState(null)
  const stepsRef = useRef(null)

  // auto-open when thinking starts
  const autoOpenedRef = useRef(false)
  useEffect(() => {
    if (isThinking) {
      setOpen(true)
    }
  }, [isThinking])
  useEffect(() => {
    if (
      isThinking &&
      steps.length > 0 &&
      !autoOpenedRef.current
    ) {
      setOpen(true)
      autoOpenedRef.current = true
    }

    if (!isThinking) {
      autoOpenedRef.current = false
    }
  }, [isThinking, steps.length])

  // auto-scroll steps
  useEffect(() => {
    if (stepsRef.current) {
      stepsRef.current.scrollTop = stepsRef.current.scrollHeight
    }
  }, [steps])

  if ((!steps || steps.length === 0) && !isThinking) {
    return null
  }

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
            <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
        )}
        <span className="thinking-drawer__label">
          {isThinking ? 'Reasoning…' : 'Reasoning steps'}
        </span>
        <span className="thinking-drawer__count">{steps.length}</span>
        <span className="thinking-drawer__chevron-wrap">
          <svg className="chevron" viewBox="0 0 24 24" strokeWidth="2">
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </span>
      </button>

      <div className={`thinking-drawer__body ${open ? 'open' : ''}`}>
        <div className="thinking-drawer__steps" ref={stepsRef}>
          {steps.map((step, i) => {
            const nextStep = steps[i + 1]

            const duration =
              nextStep?.ts && step?.ts
                ? Math.max(0, nextStep.ts - step.ts)
                : null

            const stage =
              step.stage ||
              step.node ||
              step.event

            const evalResult = step.data?.eval_result
            const decision = step.data?.decision

            const hasDetails =
              !!(
                (evalResult &&
                  Object.keys(evalResult).length) ||
                (decision &&
                  Object.keys(decision).length)
              )

            const isExpanded =
              expandedStep === i

            return (
              <div
                key={i}
                className="thinking-drawer__step"
              >
                <span
                  className={`thinking-drawer__step-dot thinking-drawer__step-dot--${stage}`}
                />

                <div className="thinking-drawer__step-content">

                  <div
                    className={`thinking-drawer__step-header ${hasDetails ? 'clickable' : ''
                      }`}
                    onClick={() => {
                      if (!hasDetails) return

                      setExpandedStep(
                        isExpanded ? null : i
                      )
                    }}
                  >
                    <div
                      className="thinking-drawer__step-event thinking-drawer__step-event">
                      <span className={`thinking-stage thinking-stage--${stage}`}>
                        {STAGE_ICONS[stage] || '⚙️'}
                        {STAGE_LABELS[stage] || stage}
                      </span>
                    </div>

                    {decision && (
                      <span className="thinking-drawer__inline-tag">
                        {decision.action}
                      </span>
                    )}

                    {evalResult && (
                      <span
                        className={`thinking-drawer__inline-tag ${evalResult.confidence >= 8
                          ? 'good'
                          : evalResult.confidence >= 5
                            ? 'medium'
                            : 'bad'
                          }`}
                      >
                        {evalResult.confidence}/10
                      </span>
                    )}

                    {duration !== null && (
                      <span className="thinking-drawer__duration">
                        {duration < 1
                          ? `${Math.round(duration * 1000)}ms`
                          : `${duration.toFixed(1)}s`}
                      </span>
                    )}

                    {hasDetails && (
                      <span
                        className={`thinking-drawer__chevron ${isExpanded ? 'open' : ''
                          }`}
                      >
                        ▼
                      </span>
                    )}
                  </div>

                  <div className="thinking-drawer__step-msg">
                    {step.message}
                  </div>

                  {hasDetails && (
                    <div
                      className={`thinking-drawer__details ${isExpanded ? 'open' : ''
                        }`}
                    >
                      <div className="thinking-drawer__details-inner">

                        {decision && (
                          <>
                            <div>
                              <strong>Action:</strong>{' '}
                              {decision.action}
                            </div>

                            {decision.reason && (
                              <div>
                                <strong>Reason:</strong>{' '}
                                {decision.reason}
                              </div>
                            )}
                          </>
                        )}

                        {evalResult && (
                          <>
                            <div>
                              <strong>Supported:</strong>{' '}
                              {String(
                                evalResult.supported
                              )}
                            </div>

                            <div>
                              <strong>Confidence:</strong>{' '}
                              {evalResult.confidence}/10
                            </div>

                            {evalResult.retry !==
                              undefined && (
                                <div>
                                  <strong>Retry:</strong>{' '}
                                  {String(
                                    evalResult.retry
                                  )}
                                </div>
                              )}

                            {evalResult.reason && (
                              <div>
                                <strong>Reason:</strong>{' '}
                                {evalResult.reason}
                              </div>
                            )}
                          </>
                        )}

                      </div>
                    </div>
                  )}

                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div >
  )
}
