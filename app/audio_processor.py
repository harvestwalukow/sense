"""
Audio Processing Pipeline for NLP Analysis
Based on SENSE App Jupyter Notebook
Includes: Transcription (Whisper), Emotion Detection (Wav2Vec2), Summarization (LexRank)
"""
from __future__ import annotations

import os
import re
import torch
import whisper
import numpy as np
import soundfile as sf
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import librosa
from pydub import AudioSegment
from scipy.ndimage import uniform_filter1d
from transformers import WhisperFeatureExtractor, AutoModelForAudioClassification
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Import progress tracker
try:
    from .progress_tracker import get_tracker
    HAS_TRACKER = True
except ImportError:
    HAS_TRACKER = False

def log_progress(message: str):
    """Log message to both console and progress tracker"""
    # Always print to console first
    print(message)
    
    # Try to log to tracker
    if HAS_TRACKER:
        try:
            tracker = get_tracker()
            with tracker._lock:
                tracker._logs.append(message)
        except Exception as e:
            # Silently fail if tracker not available
            pass


class AudioProcessor:
    """Audio processing pipeline for transcription, emotion detection, and summarization"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_whisper = None
        self.emotion_extractor = None
        self.emotion_model = None
        self.summarizer = LexRankSummarizer()
        self._models_loaded = False
        
    def load_models(self):
        """Load all required models"""
        if self._models_loaded:
            log_progress("Model sudah dimuat sebelumnya")
            return
        
        log_progress("Memulai inisialisasi model AI...")
        log_progress(f"Device: {self.device.upper()}")
        
        if self.device == "cpu":
            log_progress("INFO: Menggunakan CPU (proses akan lebih lambat)")
        else:
            log_progress("INFO: Menggunakan GPU (proses akan lebih cepat)")
        
        # Load Whisper model
        log_progress("Memuat model Whisper untuk transkripsi...")
        
        try:
            self.model_whisper = whisper.load_model("large-v3", device=self.device)
            log_progress("✓ Model Whisper berhasil dimuat")
        except Exception as e:
            log_progress(f"Gagal memuat large-v3, menggunakan model base...")
            self.model_whisper = whisper.load_model("base", device=self.device)
            log_progress("✓ Model Whisper base berhasil dimuat")
        
        # Load MERaLiON-SER Emotion model
        log_progress("Memuat model analisis emosi MERaLiON-SER-v1...")
        log_progress("INFO: Model berukuran ~3GB, sedang mengunduh dari Hugging Face...")
        log_progress("INFO: Proses ini hanya terjadi sekali saat pertama kali. Mohon tunggu...")
        
        emotion_model_name = "MERaLiON/MERaLiON-SER-v1"
        try:
            self.emotion_extractor = WhisperFeatureExtractor.from_pretrained(emotion_model_name)
            log_progress("Feature extractor berhasil dimuat, melanjutkan download model...")
            self.emotion_model = AutoModelForAudioClassification.from_pretrained(
                emotion_model_name,
                trust_remote_code=True
            )
        except Exception as e:
            log_progress(f"Error saat memuat model: {e}")
            log_progress("Mencoba lagi...")
            raise
        self.emotion_model.eval()
        
        # MERaLiON model labels: ["neutral", "happy", "angry", "sad"]
        self.emotion_labels = ["neutral", "happy", "angry", "sad"]
        
        if self.device == "cuda":
            self.emotion_model = self.emotion_model.to("cuda")
        
        log_progress("✓ Model emosi MERaLiON-SER-v1 berhasil dimuat")
        
        self._models_loaded = True
        
        log_progress("Semua model siap digunakan!")
    
    def clean_hallucinations_heavy(self, text: str) -> str:
        """Remove hallucinations and unwanted patterns from transcript"""
        text = re.sub(r'[^\x00-\x7FáéíóúÁÉÍÓÚñÑ]+', '', text)
        
        blacklist = [
            "subs by", "subtitle", "watching", "fabulous", "ketchup", "wicked",
            "residwed", "antonio", "copyright", "amara", "org", "www"
        ]
        
        text_lower = text.lower()
        for bad_word in blacklist:
            if bad_word in text_lower:
                text = re.sub(r'(?i)\b' + re.escape(bad_word) + r'\b', '', text)
        
        return re.sub(r'\s+', ' ', text).strip()
    
    def transcribe_raw_text(self, audio_path: str) -> str:
        """Transcribe audio to text using Whisper"""
        if not self._models_loaded:
            self.load_models()
        
        if HAS_TRACKER:
            tracker = get_tracker()
            tracker.set_step("Memulai transkripsi...")
        
        log_progress("Mengonversi audio menjadi teks...")
        
        if self.device == "cpu":
            log_progress("INFO: Proses transkripsi memakan waktu, mohon bersabar")
        
        # Custom progress callback
        class TranscriptionCallback:
            def __init__(self):
                self.last_text = ""
                
            def __call__(self, segment):
                text = segment.get('text', '').strip()
                if text and text != self.last_text:
                    self.last_text = text
                    # Show transcribed text
                    log_progress(f"[{segment.get('start', 0):.1f}s] {text}")
        
        result = self.model_whisper.transcribe(
            audio_path,
            language="id",
            task="transcribe",
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            verbose=None  # We'll handle progress ourselves
        )
        
        # Show all transcribed segments
        if 'segments' in result:
            for segment in result['segments']:
                text = segment.get('text', '').strip()
                if text:
                    start = segment.get('start', 0)
                    log_progress(f"[{start:.1f}s] {text}")
        
        log_progress("Membersihkan teks hasil transkripsi...")
        cleaned_text = self.clean_hallucinations_heavy(result['text'])
        log_progress(f"✓ Transkripsi selesai: {len(cleaned_text.split())} kata")
        
        return cleaned_text
    
    def is_question_context(self, words: List[str], index: int, check_range: int = 6) -> bool:
        """
        Check if the context indicates a question/topic change
        LOGIKA PENGAWAL SUPER KETAT:
        1. Cek Blacklist Frasa (Intro/Basa-basi)
        2. Cek Whitelist Frasa (Pertanyaan Beneran)
        """
        # Ambil potongan kalimat sebelumnya (konteks)
        start = max(0, index - check_range)
        prev_words = words[start:index]
        prev_text = " ".join(prev_words).lower()
        
        # --- A. BLACKLIST: Jika ada kata ini di depan, JANGAN POTONG! ---
        intro_markers = [
            "terkait", "mengetahui", "apa saja sih", "apa saja",
            "tentang apa", "coping", "judul", "tema", "yaitu"
        ]
        
        for marker in intro_markers:
            if marker in prev_text:
                return False  # BATALKAN SEGMEN BARU (Ini cuma intro)
        
        # --- B. WHITELIST: Harus ada kata tanya yang jelas ---
        valid_prefixes = [
            "bagaimana", "gimana", "jelaskan", "sebutkan", "adakah",
            "selanjutnya", "berikutnya", "terus", "lalu", "kemudian", "nah", "jadi",
            "mengenai", "soal", "pindah", "masuk", "kriteria",
            "kalau", "jika", "misalnya", "misal", "berarti"
        ]
        
        # Cek kata per kata
        for w in prev_words:
            clean_w = w.lower().strip(".,?!")
            if clean_w in valid_prefixes:
                return True
        
        return False
    
    def check_phrase_match(self, words: List[str], current_index: int, keyword: str) -> bool:
        """Check if a phrase matches at the current position"""
        keyword_parts = keyword.split()
        kw_len = len(keyword_parts)
        
        if current_index + kw_len > len(words):
            return False
        
        text_slice = words[current_index:current_index + kw_len]
        text_phrase = " ".join(text_slice).lower()
        text_phrase = re.sub(r'[^\w\s]', '', text_phrase)
        target_phrase = keyword.lower()
        
        if text_phrase == target_phrase:
            return True
        if kw_len == 1 and target_phrase in text_phrase:
            return True
        
        return False
    
    def segment_text_strict(self, full_text: str, buffer_words: int = 2) -> List[Tuple[str, str]]:
        """Segment text into topics: STRATEGI, TANTANGAN, HARAPAN, PREFERENSI DOSPEM"""
        words = full_text.split()
        
        triggers = {
            "TANTANGAN": ["tantangan", "hambatan", "kesulitan", "kendala", "masalah", "stuck"],
            "STRATEGI": ["strategi", "tips", "siasat", "mengatasi", "atasi", "mengatasinya", "coping"],
            "HARAPAN": ["harapan", "ke depan", "masa depan", "cita-cita"],
            "PREFERENSI DOSPEM": ["preferensi", "dospem", "kriteria", "memilih dosen", 
                                  "memilih pembimbing", "referensi", "referensinya"]
        }
        
        # Find all keyword matches
        all_matches = []
        for i in range(len(words)):
            for topic, kws in triggers.items():
                match_found = False
                for kw in kws:
                    if self.check_phrase_match(words, i, kw):
                        all_matches.append((i, topic))
                        match_found = True
                        break
                if match_found:
                    break
        
        # Filter valid cut points using strict logic
        valid_cut_points = []
        current_active_topic = None
        visited_topics = set()
        
        for idx, topic in all_matches:
            if topic in visited_topics:
                continue
            if topic != current_active_topic:
                # PANGGIL PENGAWAL KETAT (Range 6)
                if self.is_question_context(words, idx, check_range=6):
                    valid_cut_points.append((idx, topic))
                    current_active_topic = topic
                    visited_topics.add(topic)
        
        # Build segments
        segments_list = []
        current_idx = 0
        
        if valid_cut_points:
            first_idx = valid_cut_points[0][0]
            cut_point = max(0, first_idx - buffer_words)
            text_pendahuluan = " ".join(words[0:cut_point])
            segments_list.append(("PENDAHULUAN", text_pendahuluan))
            current_idx = cut_point
        else:
            segments_list.append(("FULL TRANSKRIP", full_text))
            return segments_list
        
        for k in range(len(valid_cut_points)):
            idx, topic = valid_cut_points[k]
            if k < len(valid_cut_points) - 1:
                next_idx = valid_cut_points[k + 1][0]
                end_point = max(0, next_idx - buffer_words)
            else:
                end_point = len(words)
            
            seg_text = " ".join(words[current_idx:end_point])
            segments_list.append((topic, seg_text))
            current_idx = end_point
        
        return segments_list
    
    def _preprocess_audio_for_emotion(self, audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """Preprocess audio for emotion analysis using MERaLiON"""
        # Load audio using pydub for format compatibility
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(target_sr)
        
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        
        # Normalize to [-1, 1]
        samples /= np.max(np.abs(samples)) + 1e-9
        
        # Silence removal using librosa
        samples, _ = librosa.effects.trim(samples, top_db=25)
        
        # Normalize amplitude
        samples = librosa.util.normalize(samples)
        
        return samples, target_sr
    
    def _segment_audio_whisper(self, y: np.ndarray, sr: int, seg_dur: float = 30.0, overlap: float = 0.5) -> List[np.ndarray]:
        """Segment audio for MERaLiON processing with overlap"""
        seg_len = int(seg_dur * sr)
        hop = int(seg_len * (1 - overlap))
        
        segments = []
        for i in range(0, len(y), hop):
            seg = y[i:i + seg_len]
            # Only include segments with at least 3 seconds of audio
            if len(seg) >= sr * 3:
                segments.append(seg)
        
        return segments
    
    def _predict_segment_emotion(self, segment: np.ndarray, sr: int) -> Optional[str]:
        """Predict emotion for a single audio segment using MERaLiON"""
        # Skip very short segments
        if len(segment) < sr * 1.5:
            return None
        
        inputs = self.emotion_extractor(
            segment,
            sampling_rate=sr,
            return_tensors="pt"
        )
        
        input_features = inputs["input_features"]
        
        # Pad if needed (minimum 3000 frames for Whisper-based model)
        if input_features.shape[-1] < 3000:
            pad_len = 3000 - input_features.shape[-1]
            input_features = torch.nn.functional.pad(
                input_features,
                (0, pad_len),
                value=0.0
            )
        
        if self.device == "cuda":
            input_features = input_features.to("cuda")
        
        with torch.no_grad():
            outputs = self.emotion_model(input_features=input_features)
        
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        
        # Map to emotion labels
        emotion_scores = {
            self.emotion_labels[i]: probs[i].item()
            for i in range(len(self.emotion_labels))
        }
        
        return max(emotion_scores, key=emotion_scores.get)
    
    def extract_emotion_complete(self, audio_path: str, segment_duration: float = 30.0) -> Dict:
        """Extract emotion from audio using MERaLiON-SER-v1"""
        if not self._models_loaded:
            self.load_models()
        
        if HAS_TRACKER:
            tracker = get_tracker()
            tracker.set_step("Menganalisis emosi...")
        
        log_progress("Memuat dan memproses file audio untuk analisis emosi...")
        
        try:
            # Preprocess audio
            y, sr = self._preprocess_audio_for_emotion(audio_path)
            
            log_progress(f"Durasi audio: {len(y)/sr:.1f} detik")
            
            # Segment audio with overlap
            segments = self._segment_audio_whisper(y, sr, seg_dur=segment_duration, overlap=0.5)
            total_segments = len(segments)
            
            log_progress(f"Memproses {total_segments} segmen dengan overlap...")
            
            emotions = []
            emotion_ids = []
            
            log_progress("Menganalisis emosi per segmen...")
            segment_count = 0
            
            for idx, seg in enumerate(segments):
                emotion = self._predict_segment_emotion(seg, sr)
                if emotion is not None:
                    emotions.append(emotion)
                    emotion_ids.append(self.emotion_labels.index(emotion))
                    segment_count += 1
                
                # Show progress every 3 segments
                if (idx + 1) % 3 == 0:
                    log_progress(f"  Segmen {idx + 1}/{total_segments} diproses...")
            
            if not emotions:
                return {
                    "dominant_emotion": "neutral",
                    "emotion_distribution": {},
                    "emotion_percentages": {},
                    "total_segments": 0,
                    "emotions_per_segment": []
                }
            
            log_progress(f"✓ Analisis emosi selesai untuk {segment_count} segmen")
            
            # Smooth emotions using moving average
            if len(emotion_ids) > 3:
                smoothed = uniform_filter1d(emotion_ids, size=3)
                emotions = [self.emotion_labels[int(round(x))] for x in smoothed]
            
            emotion_counts = Counter(emotions)
            total = len(emotions)
            
            # Map emotions to Indonesian for display
            emotion_map = {
                'neutral': 'Netral',
                'happy': 'Senang',
                'angry': 'Marah',
                'sad': 'Sedih'
            }
            
            log_progress("Distribusi emosi:")
            for emotion, count in emotion_counts.most_common():
                percentage = (count/total)*100
                emotion_id = emotion_map.get(emotion, emotion.capitalize())
                log_progress(f"  - {emotion_id}: {percentage:.1f}%")
            
            return {
                "emotions_per_segment": emotions,
                "dominant_emotion": emotion_counts.most_common(1)[0][0],
                "emotion_distribution": dict(emotion_counts),
                "emotion_percentages": {e: (c / total) * 100 for e, c in emotion_counts.items()},
                "total_segments": total
            }
        
        except Exception as e:
            log_progress(f"Error dalam analisis emosi: {e}")
            return {
                "dominant_emotion": "neutral",
                "emotion_distribution": {},
                "emotion_percentages": {},
                "total_segments": 0,
                "emotions_per_segment": []
            }
    
    def summarize_text(self, text: str, num_sentences: Optional[int] = None) -> str:
        """Summarize text using LexRank"""
        if len(text.split()) < 20:
            return text[:300] + "..." if len(text) > 300 else text
        
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            if num_sentences is None:
                num_sentences = max(3, min(10, int(len(text.split()) / 100)))
            
            summary_obj = self.summarizer(parser.document, num_sentences)
            summary = " ".join([str(s) for s in summary_obj])
            return summary
        except Exception as e:
            print(f"Summarization error: {e}")
            return text[:300] + "..."
    
    def summarize_chronological(self, segments_list: List[Tuple[str, str]]) -> str:
        """Generate chronological summary from segments"""
        full_summary_output = ""
        
        for i, (topic, text) in enumerate(segments_list):
            if len(text.split()) < 10:
                continue
            
            summary = self.summarize_text(text)
            full_summary_output += f"\n[{topic}]\n{summary}\n"
        
        return full_summary_output
    
    def analyze_audio_complete(self, audio_path: str) -> Dict:
        """
        Complete audio analysis pipeline
        Returns: transcript, segments, emotion, summary
        """
        import time
        start_time = time.time()
        
        log_progress(f"Memulai analisis: {Path(audio_path).name}")
        
        if not self._models_loaded:
            self.load_models()
        else:
            log_progress("Model sudah siap, melanjutkan proses...")
        
        # 1. Normalize audio
        if HAS_TRACKER:
            tracker = get_tracker()
            tracker.set_step("Normalisasi audio...")
        
        wav_path = "temp_proc.wav"
        try:
            log_progress("Mengonversi audio ke format WAV...")
            audio = AudioSegment.from_file(audio_path)
            duration_sec = len(audio) / 1000.0
            log_progress(f"Durasi: {duration_sec:.1f} detik ({duration_sec/60:.1f} menit)")
            
            audio = audio.set_frame_rate(16000).set_channels(1).normalize()
            audio.export(wav_path, format="wav")
            log_progress("✓ Normalisasi selesai")
        except Exception as e:
            log_progress(f"Peringatan: Menggunakan file asli")
            wav_path = audio_path
        
        # 2. Transcribe & Segment
        full_text_raw = self.transcribe_raw_text(wav_path)
        
        log_progress("Memisahkan transkrip ke dalam topik...")
        segments_list = self.segment_text_strict(full_text_raw, buffer_words=2)
        log_progress(f"✓ Ditemukan {len(segments_list)} segmen topik")
        for topic, _ in segments_list:
            log_progress(f"  - {topic}")
        
        # 3. Analyze Emotion
        emotion_result = self.extract_emotion_complete(wav_path, segment_duration=15.0)
        
        # 4. Generate Summary
        if HAS_TRACKER:
            tracker = get_tracker()
            tracker.set_step("Membuat ringkasan...")
        
        log_progress("Menghasilkan ringkasan untuk setiap topik...")
        final_summary = self.summarize_chronological(segments_list)
        log_progress("✓ Ringkasan berhasil dibuat")
        
        # Cleanup
        if os.path.exists("temp_proc.wav") and wav_path == "temp_proc.wav":
            try:
                os.remove("temp_proc.wav")
            except:
                pass
        
        total_time = time.time() - start_time
        log_progress(f"Total waktu proses: {total_time/60:.1f} menit")
        log_progress(f"Hasil: {len(full_text_raw.split())} kata berhasil ditranskripsi")
        
        # Format transcript
        formatted_transcript = ""
        for topic, content in segments_list:
            formatted_transcript += f"\n\n{'=' * 30}\n{topic}\n{'=' * 30}\n{content}"
        
        return {
            "transkripsi_bersih": formatted_transcript,
            "transkripsi_mentah": full_text_raw,
            "analisis_emosi": emotion_result,
            "ringkasan": final_summary,
            "segments": segments_list,
            "word_count": len(full_text_raw.split())
        }
    
    def aggregate_emotions(self, all_emotions: List[str]) -> Dict:
        """Aggregate emotions from multiple files"""
        if not all_emotions:
            return {
                "dominant_overall": "N/A",
                "avg_distribution": {},
                "total_segments": 0
            }
        
        emotion_counts = Counter(all_emotions)
        total = len(all_emotions)
        
        avg_distribution = {e: (c / total) * 100 for e, c in emotion_counts.items()}
        dominant = emotion_counts.most_common(1)[0][0]
        
        return {
            "dominant_overall": dominant,
            "avg_distribution": avg_distribution,
            "emotion_distribution": dict(emotion_counts),
            "total_segments": total
        }


# Singleton instance
_processor_instance = None

def get_processor() -> AudioProcessor:
    """Get or create the global audio processor instance"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = AudioProcessor()
    return _processor_instance

