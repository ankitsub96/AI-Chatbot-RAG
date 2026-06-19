import { useState, useEffect, useCallback } from 'react'
export const THEMES = [
  { id: 'void', name: 'Void', swatch: '#6c6cff' },
  { id: 'terminal', name: 'Terminal', swatch: '#00ff41' },
  { id: 'ember', name: 'Ember', swatch: '#ff6b35' },
  { id: 'arctic', name: 'Arctic', swatch: '#2563eb' },
  { id: 'noir', name: 'Noir', swatch: '#eddb3a' },
  { id: 'ocean', name: 'Ocean', swatch: '#38bdf8' },
  { id: 'emerald', name: 'Emerald', swatch: '#10b981' },
  { id: 'rose', name: 'Rose', swatch: '#ec4899' },
  { id: 'cyberpunk', name: 'Cyberpunk', swatch: '#a855f7' },
  { id: 'ruby', name: 'Ruby', swatch: '#ef4444' },
  { id: 'gold', name: 'Gold', swatch: '#f59e0b' },
  { id: 'mint', name: 'Mint', swatch: '#22c55e' },
  { id: 'sky', name: 'Sky', swatch: '#0ea5e9' },
  { id: 'lavender', name: 'Lavender', swatch: '#8b5cf6' },
  { id: 'sapphire', name: 'Sapphire', swatch: '#3b82f6' },
  { id: 'amethyst', name: 'Amethyst', swatch: '#c084fc' },
  { id: 'slate', name: 'Slate', swatch: '#64748b' },
]
export const APPEARANCES = [
  {
    id: 'dark',
    name: 'Dark',
  },
  {
    id: 'light',
    name: 'Light',
  },
]
const THEME_KEY = 'rag_theme'
const APPEARANCE_KEY = 'rag_appearance'
export function useTheme() {
  const [theme, setThemeState] = useState(
    () => localStorage.getItem(THEME_KEY) || 'void'
  )
  const [appearance, setAppearanceState] = useState(
    () => localStorage.getItem(APPEARANCE_KEY) || 'dark'
  )
  useEffect(() => {
    console.log({ theme, appearance })
    document.documentElement.setAttribute(
      'data-theme',
      theme
    )
    localStorage.setItem(THEME_KEY, theme)
  }, [theme,])
  useEffect(() => {
    console.log({ theme, appearance })
    document.documentElement.setAttribute(
      'data-appearance',
      appearance
    )
    localStorage.setItem(
      APPEARANCE_KEY,
      appearance
    )
  }, [appearance])
  const applyTransition = useCallback(() => {
    document.documentElement.classList.add(
      'theme-transition'
    )
    window.clearTimeout(
      window.__themeTransitionTimer
    )
    window.__themeTransitionTimer = window.setTimeout(
      () => {
        document.documentElement.classList.remove(
          'theme-transition'
        )
      },
      300
    )
  }, [])
  const setTheme = useCallback(
    (id) => {
      setThemeState(id)
      applyTransition()
    },
    [applyTransition]
  )
  const setAppearance = useCallback(
    (value) => {
      setAppearanceState(value)
      applyTransition()
    },
    [applyTransition]
  )
  return {
    theme,
    appearance,
    setTheme,
    setAppearance,
    themes: THEMES,
    appearances: APPEARANCES,
  }
}