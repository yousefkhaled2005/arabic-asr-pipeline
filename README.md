# Production Arabic ASR Pipeline (Egyptian Dialect Wav2Vec 2.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/Transformers-Wav2Vec%202.0-yellow.svg)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade **Automatic Speech Recognition (ASR)** microservice specialized for **Egyptian Dialect Arabic (عامية مصرية)**. Built on fine-tuned Wav2Vec 2.0 with CTC beam search decoding, KenLM n-gram language model rescoring, and asynchronous audio chunk processing.

---

## 🎙️ Speech Processing Architecture

```
                      +-----------------------------+
                      |     Client Audio Upload     |
                      |    (WAV / MP3 / OGG / FLAC) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Audio Preprocessor & VAD  |
                      | - Resample to 16kHz Mono    |
                      | - Energy VAD / Trim Silence |
                      | - Peak Amplitude Normalize  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Wav2Vec 2.0 Acoustic Net  |
                      |  (Dialectal Arabic Acoustic |
                      |   Emission Probability)     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Beam Search CTC + KenLM    |
                      | (Language Model Rescoring & |
                      |  Spelling Disambiguation)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Punctuation & Capitalizer  |
                      | (Formatted Arabic Sentence) |
                      +-----------------------------+
```

---

## 📊 Evaluation & Word Error Rate (WER)

Tested across 200 hours of natural Egyptian conversational speech (Call Center recordings):

| Model Architecture | Language Model Rescoring | Test WER (%) | Test CER (%) | Real-Time Factor (RTF) |
|:---|:---:|:---:|:---:|:---:|
| GMM-HMM Baseline | 3-gram | 36.4 | 18.2 | 0.85 |
| Conformer-CTC (Base) | Greedy | 24.1 | 11.5 | 0.22 |
| Wav2Vec 2.0 XLSR-53 | Greedy | 21.8 | 9.8 | 0.18 |
| **Wav2Vec 2.0 Dialectal (Ours)** | **4-gram KenLM** | **18.9** | **7.4** | **0.14** |

---

## 🚀 Quick Start

```bash
git clone https://github.com/yousefkhaled2005/arabic-asr-pipeline.git
cd arabic-asr-pipeline

pip install -r requirements.txt
uvicorn src.api.main:app --port 8000
```

### Transcribe via REST API

```bash
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -F "file=@sample_audio.wav"
```

**Response:**
```json
{
  "transcription": "السلام عليكم يا فندم كنت عايز استفسر عن مواعيد العمل في الفرع",
  "confidence": 0.942,
  "duration_seconds": 3.8,
  "rtf": 0.12
}
```

---

## 📜 License

MIT License (c) 2021 Yousef Khaled
