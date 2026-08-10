/** Decorative scene for citizen intro — inline SVG, no remote assets. */
export function CommuneScene({ className = '' }) {
  return (
    <svg
      className={className}
      viewBox="0 0 640 360"
      role="img"
      aria-label="Minh họa bộ phận một cửa tại xã"
    >
      <defs>
        <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#7eb89a" />
          <stop offset="55%" stopColor="#3d8f6a" />
          <stop offset="100%" stopColor="#1a4d3a" />
        </linearGradient>
        <linearGradient id="hill" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#2f6b52" />
          <stop offset="100%" stopColor="#0c1f18" />
        </linearGradient>
        <linearGradient id="building" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#f7faf8" />
          <stop offset="100%" stopColor="#d8e6de" />
        </linearGradient>
      </defs>
      <rect width="640" height="360" fill="url(#sky)" rx="28" />
      <circle cx="520" cy="72" r="36" fill="#f0e6c8" opacity="0.85" />
      <path
        d="M0 220 C120 170 220 250 320 210 C420 170 520 230 640 190 L640 360 L0 360 Z"
        fill="url(#hill)"
      />
      <path
        d="M0 280 C150 250 280 300 400 270 C500 250 580 290 640 275 L640 360 L0 360 Z"
        fill="#143528"
        opacity="0.55"
      />
      {/* One-stop building */}
      <rect x="170" y="130" width="300" height="170" rx="10" fill="url(#building)" />
      <rect x="150" y="118" width="340" height="28" rx="6" fill="#1a4d3a" />
      <text
        x="320"
        y="138"
        textAnchor="middle"
        fill="#e8f0ea"
        fontFamily="Be Vietnam Pro, sans-serif"
        fontSize="14"
        fontWeight="600"
      >
        BỘ PHẬN MỘT CỬA
      </text>
      {/* Windows */}
      <rect x="195" y="165" width="58" height="48" rx="4" fill="#9ec9b4" />
      <rect x="270" y="165" width="58" height="48" rx="4" fill="#9ec9b4" />
      <rect x="345" y="165" width="58" height="48" rx="4" fill="#9ec9b4" />
      <rect x="420" y="165" width="28" height="48" rx="4" fill="#c4782a" opacity="0.75" />
      {/* Door */}
      <rect x="290" y="230" width="60" height="70" rx="4" fill="#1a4d3a" />
      <circle cx="340" cy="268" r="3" fill="#d4c4a8" />
      {/* People dots */}
      <circle cx="220" cy="288" r="10" fill="#2f6b52" />
      <rect x="212" y="298" width="16" height="22" rx="4" fill="#3d8f6a" />
      <circle cx="420" cy="286" r="10" fill="#c4782a" />
      <rect x="412" y="296" width="16" height="22" rx="4" fill="#a86422" />
      {/* Flag */}
      <line x1="470" y1="95" x2="470" y2="170" stroke="#0c1f18" strokeWidth="3" />
      <path d="M470 98 L520 112 L470 126 Z" fill="#da251d" />
      <circle cx="485" cy="112" r="5" fill="#ffff00" />
    </svg>
  )
}
