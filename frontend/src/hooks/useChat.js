import { useState, useCallback } from 'react'
import { askDocument, askDocumentStream } from '../api/ragApi'

export function useChat(filenames, sessionId, streaming) {
  const [messages, setMessages]   = useState([])
  const [loading, setLoading]     = useState(false)
  const [streamingMsg, setStreamingMsg] = useState('')

  const appendUser = (text) =>
    setMessages(prev => [...prev, { role: 'user', content: text }])

  const appendAssistant = (text) =>
    setMessages(prev => [...prev, { role: 'assistant', content: text }])

  const send = useCallback(async (question) => {
    if (!question.trim() || loading || !filenames.length) return

    appendUser(question)
    setLoading(true)
    setStreamingMsg('')

    try {
      if (streaming) {
        const response = await askDocumentStream(filenames, question, sessionId)
        const reader   = response.body.getReader()
        const decoder  = new TextDecoder()
        let full = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          for (const line of decoder.decode(value).split('\n')) {
            if (!line.startsWith('data:')) continue
            try {
              const data = JSON.parse(line.slice(5).trim())
              if (data.token) {
                full += data.token
                setStreamingMsg(full)
              }
              if (data.done) {
                appendAssistant(full)
                setStreamingMsg('')
              }
            } catch { /* incomplete chunk */ }
          }
        }
      } else {
        const data = await askDocument(filenames, question, sessionId)
        appendAssistant(data.answer)
      }
    } catch (err) {
      appendAssistant(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }, [filenames, sessionId, streaming, loading])

  return { messages, loading, streamingMsg, send, setMessages }
}
