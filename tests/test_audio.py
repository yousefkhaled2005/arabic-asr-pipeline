from src.audio.preprocessor import AudioPreprocessor
from src.models.wav2vec_engine import EgyptianASREngine


def test_audio_inspect():
    info = AudioPreprocessor.inspect_audio(duration_sec=5.0, sample_rate=44100)
    assert info["resampling_required"] is True
    assert info["target_sample_rate"] == 16000


def test_asr_engine():
    engine = EgyptianASREngine()
    res = engine.transcribe(b"dummy_bytes", "sample.wav")
    assert "السلام عليكم" in res["transcription"]
    assert res["confidence"] > 0.9
