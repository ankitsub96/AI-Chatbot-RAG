import { Routes, Route, useNavigate, useParams } from 'react-router-dom'
import { useSessions } from './hooks/useSessions'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import './styles/global.scss'

function Chat() {
  const { sessionId } = useParams()
  const { sessions, createSession, removeSession, loadHistory } = useSessions()
  const navigate = useNavigate()

  const handleSelect = (id) => navigate(`/chat/${id}`)
  const handleCreate = () => {
    const id = createSession()
    navigate(`/chat/${id}`)
  }
  const handleDelete = async (id) => {
    await removeSession(id)
    navigate('/')
  }

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        activeId={sessionId}
        onSelect={handleSelect}
        onCreate={handleCreate}
        onDelete={handleDelete}
      />
      <ChatWindow
        sessionId={sessionId}
        loadHistory={loadHistory}
      />
    </div>
  )
}

export default function App() {
  const { sessions, createSession, removeSession, loadHistory } = useSessions()
  const navigate = useNavigate()

  return (
    <Routes>
      <Route path="/" element={
        <div className="app">
          <Sidebar
            sessions={sessions}
            activeId={null}
            onSelect={(id) => navigate(`/chat/${id}`)}
            onCreate={() => {
              const id = `session-${Date.now()}`
              navigate(`/chat/${id}`)
            }}
            onDelete={async (id) => { await removeSession(id); navigate('/') }}
          />
          <ChatWindow sessionId={null} loadHistory={loadHistory} />
        </div>
      } />
      <Route path="/chat/:sessionId" element={<Chat />} />
    </Routes>
  )
}