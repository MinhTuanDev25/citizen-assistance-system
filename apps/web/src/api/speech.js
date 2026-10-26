import { loadAuth } from '../auth/auth.js'
import {
  SPEECH_FIELD,
  SPEECH_MODEL_ID,
  SPEECH_PATH,
  SPEECH_TIMEOUT_MS,
} from '../config/speech.js'
import { newTurnRequestId } from './session.js'

let inflight = null

export function speechErrorMessage(err) {
  const status = err?.status
  const code = err?.code
  if (err?.name === 'AbortError' || code === 'SPEECH_CANCELLED') {
    return 'Đã hủy yêu cầu dịch.'
  }
  switch (code) {
    case 'UNAUTHORIZED':
      return 'Thiếu hoặc sai thông tin đăng nhập phiên chat.'
    case 'FORBIDDEN':
      return 'Phiên chat này không thuộc về bạn.'
    case 'NOT_FOUND':
      return 'Không tìm thấy phiên chat. Thử hội thoại mới.'
    case 'CONFLICT':
      return 'Phiên chat đã đóng. Tạo hội thoại mới.'
    case 'PAYLOAD_TOO_LARGE':
      return 'File âm thanh vượt quá dung lượng cho phép.'
    case 'VALIDATION':
      return 'File âm thanh không hợp lệ. Chỉ nhận WAV PCM 16 kHz mono.'
    case 'SPEECH_DISABLED':
      return 'Dịch giọng Bahnar đang tắt trên máy chủ.'
    case 'SPEECH_OVERLOADED':
    case 'SPEECH_NOT_READY':
      return 'Máy dịch đang bận hoặc chưa sẵn sàng. Thử lại sau.'
    case 'SPEECH_TIMEOUT':
      return 'Dịch giọng quá lâu. Thử bản ghi ngắn hơn.'
    case 'SPEECH_NO_SPEECH':
      return 'Không nhận ra tiếng nói. Hãy ghi âm lại, nói gần micro hơn, rồi bấm Dịch âm thanh.'
    case 'SPEECH_UNAVAILABLE':
    case 'SPEECH_UPSTREAM_AUTH':
    case 'SPEECH_INVALID_UPSTREAM':
      return 'Không gọi được máy dịch Bahnar. Kiểm tra API và RunPod.'
    default:
      break
  }
  switch (status) {
    case 401:
      return 'Thiếu hoặc sai thông tin đăng nhập phiên chat.'
    case 403:
      return 'Phiên chat này không thuộc về bạn.'
    case 404:
      return 'Không tìm thấy phiên chat. Thử hội thoại mới.'
    case 413:
      return 'File âm thanh vượt quá dung lượng cho phép.'
    case 415:
      return 'Định dạng âm thanh không được hỗ trợ.'
    case 429:
      return 'Quá nhiều yêu cầu. Đợi rồi thử lại.'
    default:
      break
  }
  if (status >= 500) {
    return 'Máy chủ gặp lỗi khi dịch giọng. Thử lại sau.'
  }
  return err?.message || 'Không dịch được giọng nói. Thử lại.'
}

/** Retry the same WAV after transient failures. Invalid audio or session must be replaced. */
export function isRetryableSpeechError(err) {
  const code = err?.code
  const status = err?.status
  if (
    code === 'VALIDATION' ||
    code === 'PAYLOAD_TOO_LARGE' ||
    code === 'UNSUPPORTED' ||
    code === 'UNAUTHORIZED' ||
    code === 'FORBIDDEN' ||
    code === 'NOT_FOUND' ||
    code === 'CONFLICT' ||
    code === 'SPEECH_NO_SPEECH' ||
    status === 401 ||
    status === 403 ||
    status === 404 ||
    status === 409 ||
    status === 413 ||
    status === 415 ||
    status === 422
  ) {
    return false
  }
  return true
}

export function parseSpeechResult(payload) {
  const data = payload?.data && typeof payload.data === 'object' ? payload.data : payload
  if (!data || typeof data !== 'object') {
    throw Object.assign(new Error('Phản hồi dịch giọng không hợp lệ.'), {
      code: 'SPEECH_INVALID_UPSTREAM',
    })
  }
  const asr = typeof data.asr_text_bahnar === 'string' ? data.asr_text_bahnar.trim() : ''
  const vi = typeof data.translated_text_vi === 'string' ? data.translated_text_vi.trim() : ''
  if (data.model_id !== SPEECH_MODEL_ID || !asr || !vi) {
    throw Object.assign(new Error('Phản hồi dịch giọng thiếu nội dung hoặc sai model.'), {
      code: 'SPEECH_INVALID_UPSTREAM',
    })
  }
  if (
    (data.duration_ms != null && !(Number(data.duration_ms) >= 0)) ||
    (data.latency_ms != null && !(Number(data.latency_ms) >= 0))
  ) {
    throw Object.assign(new Error('Phản hồi dịch giọng không hợp lệ.'), {
      code: 'SPEECH_INVALID_UPSTREAM',
    })
  }
  return {
    requestId: data.request_id || payload?.request_id || '',
    modelId: data.model_id,
    asrTextBahnar: asr,
    translatedTextVi: vi,
    durationMs: data.duration_ms,
    latencyMs: data.latency_ms,
  }
}

export async function translateSpeech({
  sessionId,
  guestToken,
  audio,
  filename = 'bahnar.wav',
  signal,
  timeoutMs = SPEECH_TIMEOUT_MS,
  fetchImpl = fetch,
}) {
  if (!sessionId) {
    throw new Error('Thiếu phiên chat.')
  }
  if (!audio || !(audio.byteLength || audio.size || audio.length)) {
    throw new Error('Không có dữ liệu âm thanh.')
  }
  if (inflight) {
    throw Object.assign(new Error('Đang dịch một bản ghi khác. Đợi xong rồi thử lại.'), {
      code: 'SPEECH_OVERLOADED',
    })
  }

  const body = new FormData()
  const blob =
    audio instanceof Blob
      ? audio
      : new Blob([audio], { type: 'audio/wav' })
  body.append(SPEECH_FIELD, blob, filename)

  const headers = { Accept: 'application/json' }
  const auth = loadAuth()
  if (auth?.accessToken) {
    headers.Authorization = `Bearer ${auth.accessToken}`
  }
  if (guestToken) {
    headers['X-Guest-Token'] = guestToken
  }
  const requestId = newTurnRequestId()
  headers['X-Request-ID'] = requestId

  const controller = new AbortController()
  let timedOut = false
  const onAbort = () => controller.abort()
  if (signal?.aborted) {
    throw Object.assign(new Error(speechErrorMessage({ code: 'SPEECH_CANCELLED' })), {
      code: 'SPEECH_CANCELLED',
    })
  }
  if (signal) {
    signal.addEventListener('abort', onAbort, { once: true })
  }
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)
  inflight = controller
  try {
    const res = await fetchImpl(SPEECH_PATH(sessionId), {
      method: 'POST',
      headers,
      body,
      signal: controller.signal,
    })
    const payload = await res.json().catch(() => ({}))
    if (!res.ok) {
      const err = new Error(payload?.error?.message || `HTTP ${res.status}`)
      err.status = res.status
      err.code = payload?.error?.code
      err.payload = payload
      err.message = speechErrorMessage(err)
      throw err
    }
    return parseSpeechResult(payload)
  } catch (err) {
    if (err?.name === 'AbortError') {
      const cancelled = Boolean(signal?.aborted) && !timedOut
      const code = cancelled ? 'SPEECH_CANCELLED' : 'SPEECH_TIMEOUT'
      throw Object.assign(new Error(speechErrorMessage({ code })), {
        code,
        name: 'AbortError',
      })
    }
    if (err?.code || err?.status) throw err
    const net = Object.assign(new Error('Mất kết nối tới máy chủ. Kiểm tra API đang chạy.'), {
      code: 'NETWORK',
    })
    throw net
  } finally {
    clearTimeout(timer)
    if (signal) signal.removeEventListener('abort', onAbort)
    inflight = null
  }
}

export function resetSpeechInflight() {
  inflight = null
}
