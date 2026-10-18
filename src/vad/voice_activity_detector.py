"""
Energy and Neural Voice Activity Detection (VAD) for Audio Stream Segmentation
"""

import numpy as np


class SileroVADProcessor:
    def __init__(self, sample_rate=16000, threshold=0.5):
        self.sample_rate = sample_rate
        self.threshold = threshold

    def detect_speech_segments(self, audio_frames: np.ndarray):
        # Frame-wise energy and speech probability simulation
        frame_size = int(self.sample_rate * 0.03) # 30ms window
        segments = []
        is_speech = False
        start = 0

        for i in range(0, len(audio_frames) - frame_size, frame_size):
            chunk = audio_frames[i:i + frame_size]
            energy = np.sqrt(np.mean(chunk ** 2))
            prob = min(1.0, energy * 8.0)

            if prob >= self.threshold and not is_speech:
                is_speech = True
                start = i / self.sample_rate
            elif prob < self.threshold and is_speech:
                is_speech = False
                end = i / self.sample_rate
                if end - start >= 0.25:
                    segments.append({"start_sec": round(start, 2), "end_sec": round(end, 2)})

        return segments
