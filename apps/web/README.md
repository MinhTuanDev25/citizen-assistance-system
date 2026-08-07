# Web (React + JavaScript)

Citizen Portal (chat) and Admin Portal (upload / review / publish) in one React app
with route-based separation (e.g. `/` citizen, `/admin` officer).

## Planned layout

```text
apps/web/
├── public/
├── src/
│   ├── pages/
│   ├── components/
│   ├── api/
│   └── App.jsx
├── package.json
├── Dockerfile
└── README.md
```

Phase 1 target: minimal Citizen chat UI calling Go `POST /api/v1/chat/turns`.
