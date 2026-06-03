import { Routes, Route, useNavigate, useParams } from 'react-router-dom'
import { useSessions } from './hooks/useSessions'
import { useTheme } from './hooks/useTheme'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import './styles/global.scss'

function Chat({ sessions, createSession, removeSession, loadHistory, theme, themes, setTheme }) {
  console.log({ sessions })
  const { sessionId } = useParams()
  const navigate = useNavigate()

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        activeId={sessionId}
        onSelect={(id) => navigate(`/chat/${id}`)}
        onCreate={async (name) => {
          const session = await createSession(name)
          const id = session?.session_id || session?.id || session
          navigate(`/chat/${id}`)
          return session
        }}
        onDelete={async (id) => {
          await removeSession(id)
          navigate('/')
        }}
        onUploaded={() => { }}
        theme={theme}
        themes={themes}
        onThemeChange={setTheme}
      />
      <ChatWindow
        sessionId={sessionId}
        sessions={sessions}
        loadHistory={loadHistory}
      />
    </div>
  )
}

export default function App() {
  const { sessions, createSession, removeSession, loadHistory } = useSessions()
  const { theme, setTheme, themes } = useTheme()
  const navigate = useNavigate()

  const sharedProps = { sessions, createSession, removeSession, loadHistory, theme, themes, setTheme }

  return (
    <Routes>
      <Route path="/" element={
        <div className="app">
          <Sidebar
            sessions={sessions}
            activeId={null}
            onSelect={(id) => navigate(`/chat/${id}`)}
            onCreate={async (name) => {
              const session = await createSession(name)
              const id = session?.session_id || session?.id || session
              navigate(`/chat/${id}`)
              return session
            }}
            onDelete={async (id) => { await removeSession(id); navigate('/') }}
            onUploaded={() => { }}
            theme={theme}
            themes={themes}
            onThemeChange={setTheme}
          />
          <ChatWindow sessionId={null} loadHistory={loadHistory} />
        </div>
      } />
      <Route path="/chat/:sessionId" element={<Chat {...sharedProps} />} />
    </Routes>
  )
}
