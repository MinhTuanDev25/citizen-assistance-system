import { useEffect, useRef, useState } from 'react'
import { isRetryableSpeechError, translateSpeech } from '../api/speech.js'
import {
  SPEECH_MAX_AUDIO_SECONDS,
  SPEECH_MAX_WAV_BYTES,
} from '../config/speech.js'
import { startBahnarCapture } from '../lib/micRecorder.js'
import { isCurrentSpeechPending, makeSpeechPending } from '../lib/speechPending.js'
import { assertWavUploadSize, validateBahnarWav, wavDurationSeconds } from '../lib/wav.js'

function formatClock(ms) {
  const sec = Math.max(0, Math.floor((ms || 0) / 1000))
  const m = String(Math.floor(sec / 60)).padStart(2, '0')
  const s = String(sec % 60).padStart(2, '0')
  return `${m}:${s}`
}

function formatBytes(n) {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`
  return `${(n / (1024 * 1024)).toFixed(1)} MiB`
}

const idleHint =
  'Ghi tiếng Bahnar rồi bấm Dừng & Dịch. Cần xem và xác nhận bản tiếng Việt trước khi hỏi trợ lý.'

export function BahnarSpeechPanel({
  sessionId,
  guestToken,
  disabled = false,
  onConfirm,
  translate = translateSpeech,
  startCapture = startBahnarCapture,
}) {
  const [mode, setMode] = useState('closed')
  const [phase, setPhase] = useState('idle')
  const [elapsedMs, setElapsedMs] = useState(0)
  const [error, setError] = useState('')
  const [fileMeta, setFileMeta] = useState(null)
  const [asr, setAsr] = useState('')
  const [vi, setVi] = useState('')
  const [status, setStatus] = useState('')
  const [retryAllowed, setRetryAllowed] = useState(false)
  const [speechRequestId, setSpeechRequestId] = useState('')
  const captureRef = useRef(null)
  const pendingRef = useRef(null)
  const abortRef = useRef(null)
  const captureAbortRef = useRef(null)
  const seqRef = useRef(0)
  const opRef = useRef(0)
  const fileGenRef = useRef(0)
  const tickRef = useRef(null)
  const fileRef = useRef(null)
  const stopLockRef = useRef(false)
  const sendLockRef = useRef(false)
  const recLockRef = useRef(false)
  const aliveRef = useRef(true)

  function bumpOp({ files = true } = {}) {
    opRef.current += 1
    if (files) fileGenRef.current += 1
    abortRef.current?.abort()
    abortRef.current = null
    captureAbortRef.current?.abort()
    captureAbortRef.current = null
    sendLockRef.current = false
    stopLockRef.current = false
    recLockRef.current = false
  }

  function stillValid(op) {
    return aliveRef.current && op === opRef.current
  }

  function clearTick() {
    if (tickRef.current) {
      clearInterval(tickRef.current)
      tickRef.current = null
    }
  }

  async function releaseCapture() {
    clearTick()
    const cap = captureRef.current
    captureRef.current = null
    if (cap) {
      try {
        await cap.cancel()
      } catch {
        /* ignore */
      }
    }
  }

  function resetResult() {
    setAsr('')
    setVi('')
    setSpeechRequestId('')
  }

  function resetPending() {
    pendingRef.current = null
    setRetryAllowed(false)
    setFileMeta(null)
    if (fileRef.current) fileRef.current.value = ''
  }

  function currentPending() {
    return isCurrentSpeechPending(pendingRef.current, fileGenRef.current)
      ? pendingRef.current
      : null
  }

  function adoptPending(encoded, { name, source }) {
    const pending = makeSpeechPending(encoded, {
      name,
      source,
      gen: fileGenRef.current,
    })
    pendingRef.current = pending
    setRetryAllowed(true)
    return pending
  }

  useEffect(() => {
    aliveRef.current = true
    return () => {
      aliveRef.current = false
      bumpOp()
      void releaseCapture()
    }
  }, [])

  const busy =
    disabled ||
    phase === 'requesting_permission' ||
    phase === 'recording' ||
    phase === 'encoding' ||
    phase === 'reading_file' ||
    phase === 'translating' ||
    phase === 'submitting_chat'

  function openMode(next) {
    if (phase === 'encoding' || phase === 'submitting_chat') return
    bumpOp()
    void releaseCapture()
    resetPending()
    resetResult()
    setError('')
    setStatus('')
    setElapsedMs(0)
    setPhase('idle')
    setMode(next)
  }

  async function beginRecord() {
    if (busy || recLockRef.current) return
    bumpOp()
    recLockRef.current = true
    const op = opRef.current
    const captureAbort = new AbortController()
    captureAbortRef.current = captureAbort
    setError('')
    resetResult()
    resetPending()
    setPhase('requesting_permission')
    setStatus('Đang xin quyền micro…')
    try {
      const cap = await startCapture({
        signal: captureAbort.signal,
        onMaxDuration: (session) => {
          if (!stillValid(op)) {
            void session.cancel()
            return
          }
          void finalizeRecord(session, { auto: true })
        },
      })
      if (!stillValid(op) || captureAbort.signal.aborted) {
        recLockRef.current = false
        try {
          await cap.cancel()
        } catch {
          /* ignore */
        }
        return
      }
      captureRef.current = cap
      recLockRef.current = false
      setPhase('recording')
      setStatus('Đang ghi. Chưa gửi máy dịch.')
      setElapsedMs(0)
      clearTick()
      tickRef.current = setInterval(() => {
        if (!stillValid(op)) {
          clearTick()
          return
        }
        setElapsedMs(cap.elapsedMs)
      }, 200)
    } catch (err) {
      recLockRef.current = false
      if (!stillValid(op) || err?.code === 'CAPTURE_CANCELLED') return
      setRetryAllowed(false)
      setPhase('translation_error')
      setError(err.message || 'Không bật được micro.')
      setStatus('')
    }
  }

  async function finalizeRecord(session, { auto = false } = {}) {
    const cap = session || captureRef.current
    if (!cap || stopLockRef.current) return
    stopLockRef.current = true
    const op = opRef.current
    captureRef.current = null
    clearTick()
    if (stillValid(op)) {
      setPhase('encoding')
      setStatus(auto ? 'Đã đủ thời lượng tối đa. Đang tạo tệp âm thanh…' : 'Đang tạo tệp âm thanh…')
    }
    try {
      const encoded = await cap.stopAndEncode()
      if (!stillValid(op)) return
      const pending = adoptPending(encoded, {
        name: 'microphone.wav',
        source: 'microphone',
      })
      setFileMeta({
        name: pending.name,
        size: pending.bytes.byteLength,
        duration: pending.durationMs / 1000,
      })
      if (auto) {
        setPhase('upload_ready')
        setStatus('Đã ghi xong. Bấm Dịch âm thanh để gửi máy chủ.')
        return
      }
      await sendAudio(pending.bytes, pending.name, op)
    } catch (err) {
      if (!stillValid(op)) return
      pendingRef.current = null
      setRetryAllowed(false)
      setPhase('translation_error')
      setError(err.message || 'Không tạo được tệp ghi âm.')
      setStatus('')
    } finally {
      if (op === opRef.current) stopLockRef.current = false
    }
  }

  async function cancelRecord() {
    bumpOp()
    const op = opRef.current
    await releaseCapture()
    if (!stillValid(op)) return
    pendingRef.current = null
    resetPending()
    resetResult()
    setPhase('idle')
    setElapsedMs(0)
    setError('')
    setStatus('Đã hủy ghi âm. Không gửi máy chủ.')
  }

  function onPickFile(event) {
    const file = event.target.files?.[0]
    if (!file) return
    void loadFile(file)
  }

  async function loadFile(file) {
    const gen = (fileGenRef.current += 1)
    pendingRef.current = null
    setRetryAllowed(false)
    const sized = assertWavUploadSize(file)
    if (!sized.ok) {
      if (gen !== fileGenRef.current || !aliveRef.current) return
      resetPending()
      resetResult()
      setPhase('translation_error')
      setError(sized.error)
      setStatus('')
      return
    }
    if (aliveRef.current && gen === fileGenRef.current) {
      setError('')
      resetResult()
      setFileMeta({ name: file.name, size: sized.size, duration: 0 })
      setPhase('reading_file')
      setStatus('Đang đọc tệp…')
    }
    let raw
    try {
      raw = new Uint8Array(await file.arrayBuffer())
    } catch {
      if (gen !== fileGenRef.current || !aliveRef.current) return
      resetPending()
      setPhase('translation_error')
      setError('Không đọc được tệp.')
      setStatus('')
      return
    }
    if (gen !== fileGenRef.current || !aliveRef.current) return
    if (raw.byteLength > SPEECH_MAX_WAV_BYTES) {
      resetPending()
      setPhase('translation_error')
      setError('File vượt quá dung lượng cho phép.')
      setStatus('')
      return
    }
    const check = validateBahnarWav(raw)
    if (!check.ok) {
      resetPending()
      setPhase('translation_error')
      setError(check.error)
      return
    }
    adoptPending(
      {
        bytes: raw,
        durationMs: Math.round(check.duration * 1000),
        sampleCount: check.parsed.dataSize / 2,
      },
      { name: file.name, source: 'upload' },
    )
    setFileMeta({
      name: file.name,
      size: raw.byteLength,
      duration: check.duration,
    })
    setPhase('upload_ready')
    setStatus('Đã chọn tệp. Bấm Dịch âm thanh để gửi.')
  }

  function clearFile() {
    fileGenRef.current += 1
    resetPending()
    resetResult()
    setPhase('idle')
    setError('')
    setStatus('')
  }

  async function sendAudio(bytes, filename, opHint) {
    const op = opHint ?? opRef.current
    if (!sessionId || !bytes || sendLockRef.current || !stillValid(op)) return
    sendLockRef.current = true
    const seq = (seqRef.current += 1)
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    if (stillValid(op)) {
      setPhase('translating')
      setError('')
      setStatus('Đang dịch Bahnar → tiếng Việt…')
    }
    try {
      const result = await translate({
        sessionId,
        guestToken,
        audio: bytes,
        filename,
        signal: controller.signal,
      })
      if (!stillValid(op) || seq !== seqRef.current) return
      setAsr(result.asrTextBahnar)
      setVi(result.translatedTextVi)
      setSpeechRequestId(result.requestId || '')
      setPhase('confirming')
      setStatus('Kiểm tra bản dịch rồi xác nhận trước khi hỏi trợ lý.')
    } catch (err) {
      if (!stillValid(op) || seq !== seqRef.current) return
      if (err?.code === 'SPEECH_CANCELLED') {
        setPhase(currentPending() ? 'upload_ready' : 'idle')
        setStatus('Đã hủy yêu cầu dịch. Chưa gửi hội thoại.')
        setError('')
        return
      }
      const keep = currentPending()
      resetResult()
      setRetryAllowed(Boolean(keep) && isRetryableSpeechError(err))
      setPhase('translation_error')
      setError(err.message || 'Không dịch được.')
      setStatus('')
    } finally {
      if (op === opRef.current) sendLockRef.current = false
    }
  }

  async function translateNow() {
    if (phase === 'reading_file' || busy) return
    if (phase !== 'upload_ready' && phase !== 'translation_error') return
    const pending = currentPending()
    if (!pending) {
      setRetryAllowed(false)
      setError('Không còn bản ghi hợp lệ để dịch. Hãy ghi âm hoặc chọn lại tệp.')
      setStatus('')
      return
    }
    if (phase === 'translation_error' && !retryAllowed) {
      setError('File âm thanh không hợp lệ. Chọn hoặc ghi bản ghi khác.')
      setStatus('')
      return
    }
    const name = pending.name || fileMeta?.name || 'bahnar.wav'
    await sendAudio(pending.bytes, name)
  }

  function cancelTranslate() {
    if (phase !== 'translating' && phase !== 'encoding') return
    bumpOp({ files: false })
    if (!aliveRef.current) return
    resetResult()
    setPhase(currentPending() ? 'upload_ready' : 'idle')
    setError('')
    setStatus('Đã hủy yêu cầu dịch. Chưa gửi hội thoại.')
  }

  async function confirm() {
    const text = vi.trim()
    if (!text || !onConfirm || phase === 'submitting_chat' || sendLockRef.current) return
    sendLockRef.current = true
    const op = opRef.current
    setPhase('submitting_chat')
    setStatus('Đang gửi câu tiếng Việt vào hội thoại…')
    try {
      const ok = await onConfirm(
        text,
        speechRequestId ? { speechRequestId } : undefined,
      )
      if (!stillValid(op)) return
      if (ok === false) {
        setPhase('confirming')
        setError('Không gửi được câu hỏi. Bản dịch vẫn còn để thử lại.')
        setStatus('')
        return
      }
      resetPending()
      resetResult()
      setPhase('idle')
      setMode('closed')
      setStatus('')
      setError('')
    } catch (err) {
      if (!stillValid(op)) return
      setPhase('confirming')
      setError(err.message || 'Không gửi được câu hỏi.')
      setStatus('')
    } finally {
      if (op === opRef.current) sendLockRef.current = false
    }
  }

  function retryFromResult() {
    resetResult()
    setError('')
    if (currentPending()) {
      setRetryAllowed(true)
      setPhase('upload_ready')
      setStatus('Có thể dịch lại tệp hiện tại hoặc ghi âm, chọn tệp mới.')
      return
    }
    setPhase('idle')
    setStatus('')
  }

  const recording = phase === 'recording' || phase === 'requesting_permission'
  const showResult = phase === 'confirming' || phase === 'submitting_chat'
  const pendingLive = currentPending()
  const canRetryTranslate = phase === 'translation_error' && retryAllowed && Boolean(pendingLive)

  return (
    <section className="bahnar-speech" aria-label="Nói Bahnar">
      <div className="bahnar-speech-toolbar" role="group" aria-label="Chọn cách nhập Bahnar">
        <button
          type="button"
          className={mode === 'mic' ? 'hint active' : 'hint'}
          disabled={disabled || phase === 'encoding' || phase === 'submitting_chat'}
          onClick={() => openMode(mode === 'mic' ? 'closed' : 'mic')}
        >
          Ghi tiếng Bahnar
        </button>
        <button
          type="button"
          className={mode === 'upload' ? 'hint active' : 'hint'}
          disabled={disabled || phase === 'encoding' || phase === 'submitting_chat'}
          onClick={() => openMode(mode === 'upload' ? 'closed' : 'upload')}
        >
          Tải tệp âm thanh
        </button>
      </div>

      {mode === 'closed' ? (
        <p className="bahnar-speech-hint">{idleHint}</p>
      ) : null}

      {mode === 'mic' ? (
        <div className="bahnar-speech-body">
          <p className="bahnar-speech-hint">
            Âm thanh chỉ được gửi tới máy chủ sau khi bạn bấm Dừng và dịch.
          </p>
          <div className="bahnar-speech-actions">
            <button
              type="button"
              className={`voice-mic${recording ? ' listening' : ''}`}
              disabled={disabled || (busy && !recording)}
              onClick={beginRecord}
            >
              Bắt đầu ghi âm
            </button>
            <button
              type="button"
              disabled={disabled || phase !== 'recording'}
              onClick={() => void finalizeRecord()}
            >
              Dừng và dịch
            </button>
            <button
              type="button"
              className="hint"
              disabled={disabled || (!recording && phase !== 'encoding')}
              onClick={() => void cancelRecord()}
            >
              Hủy ghi âm
            </button>
          </div>
          {recording ? (
            <p className="bahnar-speech-live" role="status" aria-live="polite">
              Đang ghi {formatClock(elapsedMs)} / {SPEECH_MAX_AUDIO_SECONDS} giây — chưa gửi máy chủ
            </p>
          ) : null}
        </div>
      ) : null}

      {mode === 'upload' ? (
        <div className="bahnar-speech-body">
          <p className="bahnar-speech-hint">
            Chỉ nhận tệp âm thanh đơn kênh, tối đa {SPEECH_MAX_AUDIO_SECONDS} giây /{' '}
            {formatBytes(SPEECH_MAX_WAV_BYTES)}. Tệp chỉ được gửi khi bạn yêu cầu dịch.
          </p>
          <div className="bahnar-speech-actions">
            <label className="hint bahnar-file">
              Chọn tệp âm thanh
              <input
                ref={fileRef}
                type="file"
                accept=".wav,audio/wav,audio/x-wav"
                disabled={disabled || (busy && phase !== 'reading_file')}
                onChange={onPickFile}
              />
            </label>
            <button
              type="button"
              disabled={disabled || busy || phase !== 'upload_ready' || !pendingLive}
              onClick={() => void translateNow()}
            >
              Dịch âm thanh
            </button>
            <button
              type="button"
              className="hint"
              disabled={disabled || (busy && phase !== 'reading_file') || !fileMeta}
              onClick={clearFile}
            >
              Gỡ tệp
            </button>
          </div>
          {fileMeta ? (
            <p className="bahnar-speech-meta">
              {fileMeta.name} · {formatBytes(fileMeta.size)}
              {fileMeta.duration ? ` · ${fileMeta.duration.toFixed(1)} giây` : ''}
            </p>
          ) : null}
        </div>
      ) : null}

      {phase === 'upload_ready' && mode === 'mic' && pendingLive ? (
        <div className="bahnar-speech-actions">
          <button type="button" disabled={disabled || busy} onClick={() => void translateNow()}>
            Dịch âm thanh
          </button>
        </div>
      ) : null}

      {canRetryTranslate ? (
        <div className="bahnar-speech-actions">
          <button type="button" disabled={disabled || busy} onClick={() => void translateNow()}>
            Thử dịch lại
          </button>
        </div>
      ) : null}

      {phase === 'translating' ? (
        <div className="bahnar-speech-actions">
          <button type="button" className="hint" onClick={cancelTranslate}>
            Hủy yêu cầu dịch
          </button>
        </div>
      ) : null}

      {status ? (
        <p className="bahnar-speech-status" role="status">
          {status}
        </p>
      ) : null}
      {error ? <p className="form-error">{error}</p> : null}

      {showResult ? (
        <div className="bahnar-result">
          <label>
            Văn bản Bahnar
            <textarea value={asr} readOnly rows={2} aria-label="Văn bản Bahnar" />
          </label>
          <label>
            Bản dịch tiếng Việt
            <textarea
              value={vi}
              rows={3}
              disabled={phase === 'submitting_chat'}
              onChange={(e) => setVi(e.target.value)}
              aria-label="Bản dịch tiếng Việt"
            />
          </label>
          <div className="bahnar-speech-actions">
            <button
              type="button"
              className="composer-send"
              disabled={disabled || phase === 'submitting_chat' || !vi.trim()}
              onClick={() => void confirm()}
            >
              Xác nhận &amp; Hỏi trợ lý
            </button>
            <button type="button" className="hint" disabled={disabled || phase === 'submitting_chat'} onClick={retryFromResult}>
              Dịch lại / Ghi âm lại
            </button>
            <button
              type="button"
              className="hint"
              disabled={disabled || phase === 'submitting_chat'}
              onClick={() => {
                bumpOp()
                resetPending()
                resetResult()
                setPhase('idle')
                setMode('closed')
                setError('')
                setStatus('')
              }}
            >
              Hủy
            </button>
          </div>
        </div>
      ) : null}
    </section>
  )
}

export function fileDurationLabel(bytes) {
  const check = validateBahnarWav(bytes)
  if (!check.ok) return ''
  return `${wavDurationSeconds(check.parsed).toFixed(1)}s`
}
