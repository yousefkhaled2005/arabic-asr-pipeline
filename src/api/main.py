from fastapi import FastAPI, UploadFile, File, HTTPException
from src.models.wav2vec_engine import EgyptianASREngine
from src.api.schemas import TranscriptionResponse

app = FastAPI(
    title="Egyptian Arabic Speech Recognition Microservice",
    version="1.0.0",
    description="High-Throughput ASR Inference Engine specialized in Egyptian Dialect"
)

asr_engine = EgyptianASREngine()


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "model": asr_engine.model_checkpoint}


@app.post("/api/v1/transcribe", response_model=TranscriptionResponse, tags=["ASR"])
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid audio file")
    content = await file.read()
    res = asr_engine.transcribe(content, file.filename)
    return TranscriptionResponse(**res)
