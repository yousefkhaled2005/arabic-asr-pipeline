from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    transcription: str
    confidence: float
    audio_duration_sec: float
    processing_time_sec: float
    rtf: float
