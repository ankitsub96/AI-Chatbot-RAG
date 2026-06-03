import { useState, useEffect, useCallback } from 'react'

const THEMES = [
  { id: 'void',     name: 'Void',     swatch: '#6c6cff', bg: '#080810' },
  { id: 'terminal', name: 'Terminal', swatch: '#00ff41', bg: '#010a01' },
  { id: 'ember',    name: 'Ember',    swatch: '#ff6b35', bg: '#0d0805' },
  { id: 'arctic',   name: 'Arctic',   swatch: '#2563eb', bg: '#f5f7fa' },
  { id: 'noir',     name: 'Noir',     swatch: '#eddb3a', bg: '#111111' },
]

const STORAGE_KEY = 'rag_theme'

export function useTheme() {
  const [theme, setThemeState] = useState(() => {
    return localStorage.getItem(STORAGE_KEY) || 'void'
  })

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem(STORAGE_KEY, theme)
  }, [theme])

  const setTheme = useCallback((id) => {
    // brief transition class for smooth color swap
    document.documentElement.classList.add('theme-transition')
    setThemeState(id)
    setTimeout(() => {
      document.documentElement.classList.remove('theme-transition')
    }, 350)
  }, [])

  return { theme, setTheme, themes: THEMES }
}
