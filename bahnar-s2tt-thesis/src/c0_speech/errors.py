"""HTTP-safe Speech API errors. Never include filesystem paths or traces."""
from __future__ import annotations


class SpeechAPIError(Exception):
    def __init__(self, code: str, message: str, status: int) -> None:
        self.code = code
        self.message = message
        self.status = status
        super().__init__(message)


class SpeechAudioError(SpeechAPIError):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(code, message, status)


class SpeechConfigError(SpeechAPIError):
    def __init__(self, code: str, message: str, status: int = 500) -> None:
        super().__init__(code, message, status)


PUBLIC_ERROR_MESSAGES = {
    "audio_missing": "Audio field is required",
    "audio_empty": "Audio is empty",
    "audio_invalid": "Audio could not be decoded",
    "audio_unsupported": "Only 16 kHz mono WAV is supported",
    "audio_sample_rate": "Audio sample rate must be 16000 Hz",
    "audio_channels": "Audio must be mono",
    "audio_too_long": "Audio exceeds maximum duration",
    "payload_too_large": "Upload exceeds maximum size",
    "unauthorized": "Authorization is required",
    "not_ready": "Model runtime is not ready",
    "overloaded": "Inference capacity exceeded",
    "shutting_down": "Service is shutting down",
    "inference_timeout": "Inference timed out",
    "empty_asr": "ASR produced empty transcript",
    "empty_mt": "MT produced empty translation",
    "inference_failed": "Inference could not be completed",
    "config_error": "Speech service configuration is invalid",
    "tokenizer_mismatch": "Tokenizer validation failed",
    "tokenizer_not_verified": "Tokenizer parity is not VERIFIED",
    "generation_not_rq1": "Generation settings do not match the RQ1 contract",
    "mt_contract_missing": "MT training contract path is missing or not a file",
    "shutdown_timeout": "In-flight inference did not finish before the shutdown grace period",
    "checkpoint_checksum_mismatch": "Checkpoint validation failed",
    "model_load_failed": "Model runtime could not be loaded",
}


def public_error_body(code: str, *, request_id: str) -> dict[str, object]:
    message = PUBLIC_ERROR_MESSAGES.get(code, PUBLIC_ERROR_MESSAGES["inference_failed"])
    return {"error": {"code": code, "message": message}, "request_id": request_id}


def map_c0_error_code(code: str) -> tuple[int, str]:
    table = {
        "audio_invalid": (400, "audio_invalid"),
        "empty_asr": (502, "empty_asr"),
        "empty_mt": (502, "empty_mt"),
        "tokenizer_mismatch": (503, "tokenizer_mismatch"),
        "tokenizer_not_verified": (503, "tokenizer_not_verified"),
        "generation_not_rq1": (503, "generation_not_rq1"),
        "checkpoint_checksum_mismatch": (503, "checkpoint_checksum_mismatch"),
        "checkpoint_weights_missing": (503, "checkpoint_checksum_mismatch"),
        "checkpoint_weights_unreadable": (503, "checkpoint_checksum_mismatch"),
        "model_load_failed": (503, "model_load_failed"),
        "dependency_missing": (503, "model_load_failed"),
        "config_error": (503, "config_error"),
    }
    return table.get(str(code or ""), (500, "inference_failed"))
