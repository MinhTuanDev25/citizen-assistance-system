const PATHS = {
  home: <><path d="M3 10.8 12 3l9 7.8"/><path d="M5.5 9.5V21h13V9.5"/><path d="M9.5 21v-6h5v6"/></>,
  chat: <><path d="M21 12a8.5 8.5 0 0 1-9 8.5 10 10 0 0 1-4-.8L3 21l1.4-4.2A8.4 8.4 0 0 1 3 12a8.5 8.5 0 0 1 9-8.5A8.5 8.5 0 0 1 21 12Z"/><path d="M8 12h.01M12 12h.01M16 12h.01"/></>,
  procedures: <><path d="M8 6h13M8 12h13M8 18h13"/><path d="M3.5 6h.01M3.5 12h.01M3.5 18h.01"/></>,
  documents: <><path d="M6 2.8h8l4 4V21H6z"/><path d="M14 2.8V7h4M9 12h6M9 16h6"/></>,
  drafts: <><path d="M4 20h4l11-11a2.8 2.8 0 0 0-4-4L4 16v4Z"/><path d="m13.5 6.5 4 4M4 20h16"/></>,
  account: <><circle cx="12" cy="8" r="4"/><path d="M4.5 21a7.5 7.5 0 0 1 15 0"/></>,
  switch: <><path d="M7 7h11l-3-3M17 17H6l3 3"/><path d="m18 7-3 3M6 17l3-3"/></>,
  search: <><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></>,
  arrow: <><path d="M5 12h14M14 7l5 5-5 5"/></>,
  upload: <><path d="M12 16V3M7 8l5-5 5 5"/><path d="M4 15v5h16v-5"/></>,
  shield: <><path d="M12 3 4.5 6v5.8c0 4.6 3.2 7.7 7.5 9.2 4.3-1.5 7.5-4.6 7.5-9.2V6z"/><path d="m8.5 12 2.2 2.2 4.8-5"/></>,
  sparkle: <><path d="m12 3 1.2 3.8L17 8l-3.8 1.2L12 13l-1.2-3.8L7 8l3.8-1.2z"/><path d="m18.5 14 .7 2.3 2.3.7-2.3.7-.7 2.3-.7-2.3-2.3-.7 2.3-.7z"/><path d="M5 14.5v5M2.5 17h5"/></>,
}

export function UiIcon({ name, size = 20, className = '' }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {PATHS[name] || PATHS.sparkle}
    </svg>
  )
}

export function ServiceMark() {
  return (
    <span className="brand-icon" aria-hidden="true">
      <svg viewBox="0 0 32 32" fill="none">
        <path d="M5 12.5 16 6l11 6.5" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M7.5 12.5h17V25h-17z" stroke="currentColor" strokeWidth="2.4" strokeLinejoin="round" />
        <path d="M12 25v-7h8v7M5 25h22" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
      </svg>
    </span>
  )
}
