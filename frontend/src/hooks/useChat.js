import { useState, useCallback, useEffect } from 'react'
import { askEndpoint } from '../api/ragApi'
import { createSSEStreamReader } from '../utils/sseParser'
import { DEFAULT_ENDPOINT_ID, DEFAULT_STRICTNESS, getEndpointConfig } from '../config/endpoints'

export function useChat(
  sessionId,
  sessions,
  selectedIds,
  streaming,
  endpointId = DEFAULT_ENDPOINT_ID,
  strictness = DEFAULT_STRICTNESS,
  useWeb = false,
) {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [sessionName, setSessionName] = useState('Untitled')

  useEffect(() => {
    setSessionName(findSessionName())
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessions, sessionId])

  function findSessionName() {
    const session = (sessions || []).find(s => s.session_id === sessionId || s.id === sessionId)
    if (!session) return 'Untitled'
    return session.title || session.session_id?.slice(0, 16) || 'Untitled'
  }

  // ── Message-array helpers ──────────────────────────────
  // Every assistant message owns its own thoughts/isThinking/endpoint/
  // strictness — nothing about streaming state lives outside `messages`
  // anymore. Appending a new user+assistant pair never touches earlier
  // entries, which is what keeps a previous answer from disappearing when
  // a second question is asked.

  const appendUserMessage = (text) =>
    setMessages(prev => [...prev, { role: 'user', content: text, ts: Date.now() }])

  const appendAssistantPlaceholder = (meta) =>
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: '',
      ts: Date.now(),
      thoughts: [],
      hasThoughts: false,
      isThinking: false,
      endpoint: meta.endpoint,
      strictness: meta.strictness,
    }])

  // Mutates the *last* message in place. Safe to assume it's the one
  // in-flight assistant placeholder: `send()` guards on `loading` so only
  // one message can ever be streaming at a time.
  const updateLastAssistant = (updater) =>
    setMessages(prev => {
      if (prev.length === 0) return prev
      const next = prev.slice()
      next[next.length - 1] = updater(next[next.length - 1])
      return next
    })

  const appendThought = (thought) =>
    updateLastAssistant(msg => ({
      ...msg,
      isThinking: true,
      hasThoughts: true,
      thoughts: [...msg.thoughts, thought],
    }))

  const appendToken = (token) =>
    updateLastAssistant(msg => ({
      ...msg,
      isThinking: false,
      content: msg.content + token,
    }))

  const finalizeAssistant = () =>
    updateLastAssistant(msg => ({ ...msg, isThinking: false }))

  const setAssistantContent = (content, thoughts) =>
    updateLastAssistant(msg => ({
      ...msg,
      content,
      thoughts,
      hasThoughts: thoughts.length > 0,
      isThinking: false,
    }))

  const setAssistantError = (text) =>
    updateLastAssistant(msg => ({ ...msg, content: text, isThinking: false }))

  // ── send ─────────────────────────────────────────────────

  const send = useCallback(async (question) => {
    if (!question.trim() || loading) return

    appendUserMessage(question)
    setLoading(true)

    // pass null if all ready docs selected (backend resolves from session)
    const docIds = selectedIds.length ? selectedIds : null
    const config = getEndpointConfig(endpointId)

    appendAssistantPlaceholder({
      endpoint: endpointId,
      // only record strictness when this endpoint actually used it —
      // otherwise it'd misleadingly imply a value that was never sent
      strictness: config?.supportsStrictness ? strictness : null,
    })

    try {
      const result = await askEndpoint(endpointId, {
        sessionId,
        question,
        documentIds: docIds,
        stream: streaming,
        useWeb,
        strictness,
      })

      if (result instanceof Response) {
        // streaming: askEndpoint handed back the raw fetch Response —
        // pipe it through the one shared SSE reader, for both legacy and
        // new wire formats alike
        if (!result.ok) {
          throw new Error(`Request failed (${result.status})`)
        }

        await createSSEStreamReader(result, (event) => {
          console.log('SSE EVENT', event)
          switch (event.kind) {
            case 'thinking':
              appendThought({
                stage: event.stage ||
                  event.node ||
                  event.event ||
                  event.thinking ||
                  'thinking',
                message: event.message || '',
                data: event.data || null,
                ts: Date.now(),
              })
              break
            case 'token':
              appendToken(event.token)
              break
            case 'done':
              finalizeAssistant()
              break
            default:
              break
          }
        })
        // belt-and-suspenders: not every server is guaranteed to emit an
        // explicit `done` event before closing the stream
        finalizeAssistant()
      } else {
        // non-streaming: askEndpoint already resolved the parsed JSON answer
        setAssistantContent(result?.answer ?? '', result?.thoughts || [])
      }
    } catch (err) {
      setAssistantError(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }, [sessionId, selectedIds, streaming, loading, endpointId, strictness, useWeb])

  return {
    messages,
    loading,
    sessionName,
    send,
    setMessages,
  }
}
