import { useState, useCallback, useEffect } from 'react'
import { askDocument, askDocumentStream } from '../api/ragApi'

export function useChat(sessionId, sessions, selectedIds, streaming) {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [streamingMsg, setStreamingMsg] = useState('')
  const [thinkingSteps, setThinkingSteps] = useState([])
  const [isThinking, setIsThinking] = useState(false)
  const [sessionName, setSessionName] = useState('Untitled')
  console.log({ sessions })

  const appendUser = (text) =>
    setMessages(prev => [...prev, { role: 'user', content: text, ts: Date.now() }])

  const appendAssistant = (text) =>
    setMessages(prev => [...prev, { role: 'assistant', content: text, ts: Date.now() }])
  useEffect(() => {
    setSessionName(findSessionName())
  }, [sessions])
  function findSessionName() {
    const session = (sessions || []).find(s => s.id === sessionId)

    if (!session) return 'Untitled'

    return session.title || session.session_id?.slice(0, 16) || 'Untitled'
  }
  const send = useCallback(async (question) => {
    if (!question.trim() || loading) return

    appendUser(question)
    setLoading(true)
    setStreamingMsg('')
    setThinkingSteps([])
    setIsThinking(false)

    // pass null if all ready docs selected (backend resolves from session)
    const docIds = selectedIds.length ? selectedIds : null

    try {
      if (streaming) {
        const response = await askDocumentStream(sessionId, question, docIds)
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let full = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          for (const line of decoder.decode(value).split('\n')) {
            if (!line.startsWith('data:')) continue
            try {
              const data = JSON.parse(line.slice(5).trim())

              // thinking event
              if (data.thinking) {
                setIsThinking(true)
                setThinkingSteps(prev => [...prev, {
                  event: data.thinking,
                  message: data.message || '',
                  ts: Date.now(),
                }])
                continue
              }

              // token
              if (data.token) {
                setIsThinking(false)
                full += data.token
                setStreamingMsg(full)
              }

              // done
              if (data.done) {
                appendAssistant(full)
                setStreamingMsg('')
                setIsThinking(false)
              }
            } catch { /* incomplete chunk */ }
          }
        }
      } else {
        const data = await askDocument(sessionId, question, docIds)
        appendAssistant(data.answer)
      }
    } catch (err) {
      appendAssistant(`Error: ${err.message}`)
    } finally {
      setLoading(false)
      setIsThinking(false)
    }
  }, [sessionId, selectedIds, streaming, loading])

  return {
    messages,
    loading,
    streamingMsg,
    thinkingSteps,
    isThinking, sessionName,
    send,
    setMessages,
  }
}
