from typing import Dict, Any


class AudioPreprocessor:
    """Prepares audio streams for Wav2Vec acoustic model input."""

    TARGET_SAMPLE_RATE = 16000

    @staticmethod
    def inspect_audio(duration_sec: float, sample_rate: int) -> Dict[str, Any]:
        return {
            "duration": round(duration_sec, 2),
            "target_sample_rate": AudioPreprocessor.TARGET_SAMPLE_RATE,
            "resampling_required": sample_rate != AudioPreprocessor.TARGET_SAMPLE_RATE,
            "status": "normalized"
        }
