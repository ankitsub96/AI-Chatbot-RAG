import { useState, useEffect } from 'react'
import { useDocumentSelection } from '../hooks/useDocumentSelection'
import { getDocuments } from '../api/ragApi'
import { useChat } from '../hooks/useChat'
import MessageList from './MessageList'
import ChatInput from './ChatInput'
import '../styles/ChatWindow.scss'

export default function ChatWindow({ sessionId, loadHistory }) { 
  const [streaming, setStreaming]   = useState(true)
  const [docOpen, setDocOpen] = useState(false)

  const { documents, filenames, setFilename, toggleFile } = useDocumentSelection(sessionId)
  const { messages, loading, streamingMsg, send, setMessages } = useChat(
    filenames, sessionId, streaming
  )
 


  // load history when session changes
  useEffect(() => {
    if (!sessionId) { setMessages([]); return }
    loadHistory(sessionId).then(items => {
      setMessages(items.map(item => ([
        { role: 'user',      content: item.question },
        { role: 'assistant', content: item.answer   },
      ])).flat())
    })
  }, [sessionId])

  if (!sessionId) return (
    <div className="chat-window chat-window--empty">
      <p>Select or create a session to start</p>
    </div>
  )

  return (
    <div className="chat-window">
      <div className="chat-window__header">
        <div className="chat-window__doc-multi">
          <span
            className="chat-window__doc-label"
            onClick={() => setDocOpen(o => !o)}
          >
            {filenames.length} file{filenames.length !== 1 ? 's' : ''} selected ▾
          </span>
          {docOpen && (
            <div className="chat-window__doc-list">
              {documents.map(d => (
                <label key={d} className={`chat-window__doc-item ${filenames.includes(d) ? 'selected' : ''}`}>
                  <input type="checkbox" checked={filenames.includes(d)} onChange={() => toggleFile(d)} />
                  <span>{d}</span>
                </label>
              ))}
            </div>
          )}
        </div>
        <span className="chat-window__session">{sessionId}</span>
      </div>

      <MessageList
        messages={messages}
        streamingMsg={streamingMsg}
        loading={loading}
      />

      <ChatInput
        onSend={send}
        loading={loading}
        streaming={streaming}
        onToggleStream={() => setStreaming(s => !s)}
      />
    </div>
  )
}
