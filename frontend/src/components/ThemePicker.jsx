import '../styles/Modal.scss'

const CheckIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2.5">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
)

const PaletteIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2" fill="none">
    <circle cx="13.5" cy="6.5" r=".5" fill="currentColor"/>
    <circle cx="17.5" cy="10.5" r=".5" fill="currentColor"/>
    <circle cx="8.5" cy="7.5" r=".5" fill="currentColor"/>
    <circle cx="6.5" cy="12.5" r=".5" fill="currentColor"/>
    <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 011.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.554C21.965 6.012 17.461 2 12 2z"/>
  </svg>
)

const XIcon = () => (
  <svg viewBox="0 0 24 24" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18"/>
    <line x1="6" y1="6" x2="18" y2="18"/>
  </svg>
)

export default function ThemePicker({ theme, themes, onSelect, onClose }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal__header">
          <div className="modal__title-wrap">
            <div className="modal__title-icon"><PaletteIcon /></div>
            <span className="modal__title">Choose Theme</span>
          </div>
          <button className="modal__close" onClick={onClose}><XIcon /></button>
        </div>

        <div className="modal__body">
          <div className="theme-picker">
            {themes.map(t => (
              <button
                key={t.id}
                className={`theme-picker__option ${theme === t.id ? 'active' : ''}`}
                onClick={() => { onSelect(t.id); onClose() }}
              >
                <span
                  className="theme-picker__swatch"
                  style={{ background: t.bg, border: `2px solid ${t.swatch}` }}
                />
                <span className="theme-picker__name">{t.name}</span>
                <span className="theme-picker__check">
                  {theme === t.id && <CheckIcon />}
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="modal__footer">
          <button className="modal__btn modal__btn--ghost" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  )
}
