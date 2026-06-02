import numpy as np
import librosa
import sounddevice as sd
import wave
from threading import Thread
import json
from datetime import datetime

class NoiseDetector:
    def __init__(self, threshold_db=60, sample_rate=44100, chunk=2048):
        """
        Inisialisasi Noise Detector
        threshold_db: Batas kebisingan (dB) - default 60dB untuk perpustakaan
        """
        self.threshold_db = threshold_db
        self.sample_rate = sample_rate
        self.chunk = chunk
        self.is_recording = False
        self.noise_history = []
        
    def get_loudness_db(self, audio_chunk):
        """Menghitung loudness dalam dB"""
        rms = np.sqrt(np.mean(np.square(audio_chunk)))
        if rms > 0:
            db = 20 * np.log10(rms)
        else:
            db = -np.inf
        return db
    
    def analyze_frequency(self, audio_chunk):
        """Analisis frekuensi untuk deteksi tipe suara"""
        D = librosa.stft(audio_chunk)
        S = np.abs(D) ** 2
        freq = librosa.fft_frequencies(
    sr=self.sample_rate,
    n_fft=D.shape[0] * 2 - 2
)
        
        # Cari frekuensi dominan
        power = np.sum(S, axis=1)
        dominant_freq = freq[np.argmax(power)]
        
        return dominant_freq
    
    def detect_noise_event(self, audio_chunk):
        """Deteksi dan klasifikasi event kebisingan"""
        loudness = self.get_loudness_db(audio_chunk)
        dominant_freq = self.analyze_frequency(audio_chunk)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "loudness_db": round(loudness, 2),
            "dominant_frequency_hz": round(dominant_freq, 2),
            "is_excessive_noise": loudness > self.threshold_db,
            "alert_level": self._classify_alert(loudness)
        }
        
        return result
    
    def _classify_alert(self, db):
        """Klasifikasi tingkat alert"""
        if db < 40:
            return "NORMAL"
        elif db < 60:
            return "WARNING"
        else:
            return "CRITICAL"
    def start_realtime_detection(self):

        self.is_recording = True
        print("🎤 Recording dimulai...")

        def audio_callback(indata, frames, time, status):

            if status:
                print(status)

            audio_chunk = indata[:, 0]

            result = self.detect_noise_event(audio_chunk)

            self.noise_history.append(result)

            print(
                f"Suara: {result['loudness_db']} dB | "
                f"Freq: {result['dominant_frequency_hz']} Hz"
            )

            if result["is_excessive_noise"]:

                print(
                    f"⚠️ ALERT: "
                    f"{result['alert_level']} - "
                    f"{result['loudness_db']} dB"
                )

        with sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk,
            callback=audio_callback
        ):

            while self.is_recording:
                sd.sleep(100)
                
    def stop_detection(self):
        """Stop deteksi real-time"""
        self.is_recording = False
    
    def get_statistics(self):
        """Dapatkan statistik kebisingan"""
        if not self.noise_history:
            return None
        
        loudness_values = [h["loudness_db"] for h in self.noise_history]
        
        return {
            "total_samples": len(self.noise_history),
            "average_db": round(np.mean(loudness_values), 2),
            "max_db": round(np.max(loudness_values), 2),
            "min_db": round(np.min(loudness_values), 2),
            "excessive_noise_events": sum(1 for h in self.noise_history if h["is_excessive_noise"]),
            "noise_percentage": round((sum(1 for h in self.noise_history if h["is_excessive_noise"]) / len(self.noise_history)) * 100, 2)
        }
detector = NoiseDetector()

detector.start_realtime_detection()