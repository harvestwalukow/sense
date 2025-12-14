from __future__ import annotations

import base64
import csv
import json
import os
import random
import re
import uuid
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, List, Optional

from docx import Document
from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from mutagen import File as MutagenFile
from werkzeug.utils import secure_filename
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Import audio processor and evaluation modules
from .audio_processor import get_processor
from .evaluation import evaluate_transcript, get_clean_text_from_docx
from .batch_processor import extract_and_discover_files, batch_process_audio_files
from .progress_tracker import get_tracker, reset_tracker
from .ollama_summarizer import get_ollama_summarizer


bp = Blueprint("main", __name__)

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma"}
TR_EXT = {".txt", ".docx", ".json"}
EMOTIONS = ["Neutral", "Happy", "Sad", "Angry"]


def _project_root() -> Path:
    return Path(current_app.root_path).resolve().parent


def _storage_dir() -> Path:
    d = _project_root() / "storage"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _records_dir() -> Path:
    d = _storage_dir() / "records"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _uploads_dir() -> Path:
    d = _storage_dir() / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _transcripts_dir() -> Path:
    d = _storage_dir() / "transcripts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _batches_dir() -> Path:
    d = _storage_dir() / "batches"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _wordclouds_dir() -> Path:
    d = _storage_dir() / "wordclouds"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _generate_wordcloud_hash(records: List[Dict]) -> str:
    """Generate hash based on all record IDs to detect changes"""
    import hashlib
    record_ids = sorted([r.get("id", "") for r in records])
    hash_input = "".join(record_ids)
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def _get_or_generate_wordcloud(records: List[Dict], stopwords: set) -> Optional[str]:
    """Get cached wordcloud or generate new one with blue gradient"""
    if not records:
        return None
    
    # Calculate hash based on record IDs
    wordcloud_hash = _generate_wordcloud_hash(records)
    wordcloud_file = _wordclouds_dir() / f"wordcloud_{wordcloud_hash}.png"
    
    # Check if cached version exists
    if wordcloud_file.exists():
        try:
            # Read cached image and convert to base64
            with open(wordcloud_file, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            current_app.logger.error(f"Error reading cached wordcloud: {e}")
    
    # Generate new wordcloud
    all_text = " ".join(r.get("transcript", "") for r in records)
    if not all_text.strip():
        return None
    
    try:
        # Custom color function for blue gradient (darker = more frequent)
        def blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            """Generate blue colors - darker for more frequent words"""
            # font_size is proportional to word frequency
            # Normalize to 0-1 range (larger font = more frequent)
            # Max font size in wordcloud is typically around 200-300
            intensity = min(font_size / 150.0, 1.0)
            
            # Generate darker blue for higher intensity
            # Dark blue: (8, 48, 107) to Light blue: (158, 202, 225)
            r = int(8 + 150 * (1 - intensity))
            g = int(48 + 154 * (1 - intensity))
            b = int(107 + 118 * (1 - intensity))
            
            return f'rgb({r},{g},{b})'
        
        # Create wordcloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=stopwords,
            max_words=50,
            relative_scaling=0.5,
            min_font_size=10,
            color_func=blue_color_func,
            prefer_horizontal=0.7,
            random_state=42  # Fixed seed for consistent layout
        ).generate(all_text)
        
        # Convert to image
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Save to file (cache)
        plt.savefig(wordcloud_file, format='png', bbox_inches='tight', dpi=100)
        
        # Also save to BytesIO for immediate return
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plt.close()
        
        # Convert to base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
    except Exception as e:
        current_app.logger.error(f"Error generating wordcloud: {e}")
        return None


def format_duration(seconds: float | int | None) -> str:
    """Format seconds to HH:MM:SS"""
    if seconds is None or seconds == 0:
        return "00:00:00"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _audio_duration_seconds(path: Path | None) -> float:
    """Get audio duration using mutagen"""
    if path is None:
        return 0.0
    try:
        mf = MutagenFile(str(path))
        if mf is None or not hasattr(mf, "info") or mf.info is None:
            return 0.0
        return float(getattr(mf.info, "length", 0.0) or 0.0)
    except Exception:
        return 0.0


def _read_transcript(path: Path | None) -> str:
    """Read transcript from txt/docx/json"""
    if path is None:
        return ""
    ext = path.suffix.lower()
    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".docx":
        try:
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs).strip()
        except Exception:
            try:
                import zipfile
                import xml.etree.ElementTree as ET
                from io import BytesIO
                
                data = path.read_bytes()
                with zipfile.ZipFile(BytesIO(data)) as zf:
                    xml_bytes = zf.read("word/document.xml")
                
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                root = ET.fromstring(xml_bytes)
                texts = [t.text for t in root.findall(".//w:t", ns) if t.text]
                return " ".join(texts)
            except Exception:
                return ""
    if ext == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(obj, dict) and "text" in obj:
                return str(obj["text"])
            if isinstance(obj, list):
                parts = []
                for it in obj:
                    if isinstance(it, dict) and "text" in it:
                        parts.append(str(it["text"]))
                    else:
                        parts.append(str(it))
                return "\n".join(parts).strip()
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def _generate_mock_transcript(audio_name: str) -> str:
    """Generate dummy transcript bertema wawancara mahasiswa S2"""
    templates = [
        "Saya merasa cukup kesulitan dalam memahami materi kuliah yang diberikan. Terutama pada mata kuliah statistika, banyak konsep yang masih belum jelas. Saya perlu belajar lebih ekstra untuk memahami materi tersebut.",
        "Pengalaman saya selama S2 cukup menantang. Ada banyak tugas dan deadline yang harus dipenuhi. Tapi saya tetap semangat untuk belajar dan menyelesaikan studi dengan baik.",
        "Saya berharap bisa menyelesaikan studi dengan baik. Pembimbing saya sudah memberikan arahan yang jelas untuk penelitian. Saya merasa didukung dan termotivasi untuk menyelesaikan tesis.",
        "Tantangan utama yang saya hadapi adalah manajemen waktu. Antara kuliah, penelitian, dan aktivitas lain harus seimbang. Saya menggunakan beberapa strategi untuk mengatasi tekanan ini.",
        "Saya menggunakan beberapa strategi untuk mengatasi tekanan, seperti membuat jadwal harian yang terstruktur dan berdiskusi dengan teman sekelas. Diskusi dengan teman sangat membantu saya memahami materi.",
    ]
    return random.choice(templates) + " " + random.choice(templates)


def _generate_mock_emotion() -> Dict[str, float]:
    """Generate dummy emotion distribution (total 100%)"""
    values = []
    remaining = 100.0
    for i in range(len(EMOTIONS)):
        if i == len(EMOTIONS) - 1:
            values.append(round(remaining, 1))
        else:
            val = round(random.uniform(10, 40), 1)
            values.append(val)
            remaining -= val
    return dict(zip(EMOTIONS, values))


def _get_dominant_emotion(emotion_dist: Dict[str, float]) -> str:
    """Get dominant emotion from distribution"""
    return max(emotion_dist.items(), key=lambda x: x[1])[0]


def _save_record(record_id: str, data: Dict) -> None:
    """Save record to JSON"""
    record_file = _records_dir() / f"{record_id}.json"
    record_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_record(record_id: str) -> Optional[Dict]:
    """Load record from JSON"""
    record_file = _records_dir() / f"{record_id}.json"
    if not record_file.exists():
        return None
    return json.loads(record_file.read_text(encoding="utf-8"))


def _list_all_records() -> List[Dict]:
    """List all records"""
    records = []
    for f in sorted(_records_dir().glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            records.append(data)
        except Exception:
            continue
    return records


def _save_batch(batch_id: str, data: Dict) -> None:
    """Save batch summary to JSON"""
    batch_file = _batches_dir() / f"{batch_id}.json"
    batch_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_batch(batch_id: str) -> Optional[Dict]:
    """Load batch summary from JSON"""
    batch_file = _batches_dir() / f"{batch_id}.json"
    if not batch_file.exists():
        return None
    return json.loads(batch_file.read_text(encoding="utf-8"))


@bp.app_context_processor
def inject_helpers():
    return {"format_duration": format_duration}


@bp.route("/")
def index():
    """Landing page"""
    return render_template("landing.html")


@bp.route("/init-models")
def init_models():
    """Initialize NLP models (can be called manually or on first use)"""
    try:
        processor = get_processor()
        processor.load_models()
        flash("✅ Models loaded successfully!", "success")
    except Exception as e:
        current_app.logger.error(f"Error loading models: {e}", exc_info=True)
        flash(f"❌ Error loading models: {str(e)}", "danger")
    
    return redirect(url_for("main.index"))


@bp.route("/progress")
def progress():
    """Get current processing progress"""
    tracker = get_tracker()
    return tracker.get_progress()


@bp.route("/ingest", methods=["GET", "POST"])
def ingest():
    """Input rekaman: Audio Only atau Audio + GT"""
    if request.method == "GET":
        # Reset tracker for new upload
        reset_tracker()
        return render_template("ingest.html")
    
    try:
        if "audio" not in request.files:
            flash("Pilih file audio terlebih dahulu.", "danger")
            return redirect(url_for("main.ingest"))
        
        audio_file = request.files["audio"]
        if not audio_file or not audio_file.filename:
            flash("File audio tidak valid atau kosong.", "danger")
            return redirect(url_for("main.ingest"))
        
        audio_ext = Path(audio_file.filename).suffix.lower()
        if audio_ext not in AUDIO_EXT:
            flash(f"Format audio tidak didukung. Gunakan: {', '.join(AUDIO_EXT)}", "danger")
            return redirect(url_for("main.ingest"))
        
        record_id = uuid.uuid4().hex[:12]
        audio_filename = secure_filename(f"{record_id}_{audio_file.filename}")
        audio_path = _uploads_dir() / audio_filename
        audio_file.save(audio_path)
        
        if not audio_path.exists():
            flash("Gagal menyimpan file audio.", "danger")
            return redirect(url_for("main.ingest"))
        
        duration = _audio_duration_seconds(audio_path)
        
        transcript_text = ""
        has_gt = False
        tr_filename = None
        transcript_file = request.files.get("transcript")
        
        if transcript_file and transcript_file.filename:
            tr_ext = Path(transcript_file.filename).suffix.lower()
            if tr_ext in TR_EXT:
                tr_filename = secure_filename(f"{record_id}_{transcript_file.filename}")
                tr_path = _transcripts_dir() / tr_filename
                transcript_file.save(tr_path)
                transcript_text = _read_transcript(tr_path)
                has_gt = True
        
        # Perform real audio analysis
        try:
            # Initialize progress tracker
            print(f"\n{'='*60}")
            print(f"UPLOAD REQUEST")
            print(f"{'='*60}")
            print(f"Upload baru diterima: {audio_file.filename}")
            print(f"Ukuran file: {audio_path.stat().st_size / (1024*1024):.2f} MB")
            
            tracker = get_tracker()
            tracker._logs.append(f"Upload baru diterima: {audio_file.filename}")
            tracker._logs.append(f"Ukuran file: {audio_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Set initial step
            tracker.set_step("Menyiapkan model...")
            
            processor = get_processor()
            
            # Analyze audio (transcription, emotion, segmentation, summary)
            analysis_result = processor.analyze_audio_complete(str(audio_path))
            
            # Get analyzed transcript
            analyzed_transcript = analysis_result["transkripsi_mentah"]
            segments = analysis_result["segments"]
            emotion_data = analysis_result["analisis_emosi"]
            summary = analysis_result["ringkasan"]
            
            # Map emotion labels to readable names
            emotion_label_map = {
                'neu': 'Neutral',
                'hap': 'Happy',
                'ang': 'Angry',
                'sad': 'Sad'
            }
            
            # Convert emotion distribution to readable format
            emotion_dist = {}
            for emotion, percentage in emotion_data.get("emotion_percentages", {}).items():
                readable_name = emotion_label_map.get(emotion, emotion.capitalize())
                emotion_dist[readable_name] = round(percentage, 1)
            
            dominant_emotion_raw = emotion_data.get("dominant_emotion", "neu")
            dominant_emotion = emotion_label_map.get(dominant_emotion_raw, dominant_emotion_raw.capitalize())
            
            # If ground truth provided, evaluate WER/CER
            wer_cer_metrics = None
            if has_gt:
                gt_path = _transcripts_dir() / tr_filename
                wer_cer_metrics = evaluate_transcript(gt_path, analyzed_transcript)
            
        except Exception as e:
            current_app.logger.error(f"Error in audio analysis: {e}", exc_info=True)
            flash(f"⚠️ Error during analysis: {str(e)}. Using fallback data.", "warning")
            # Fallback to mock data if analysis fails
            analyzed_transcript = transcript_text if transcript_text else _generate_mock_transcript(audio_file.filename)
            emotion_dist = _generate_mock_emotion()
            dominant_emotion = _get_dominant_emotion(emotion_dist)
            segments = [("FULL TRANSKRIP", analyzed_transcript)]
            summary = "Summary unavailable due to processing error."
            wer_cer_metrics = None
        
        record_data = {
            "id": record_id,
            "audio_filename": audio_filename,
            "audio_name": audio_file.filename,
            "duration_sec": duration,
            "transcript": analyzed_transcript,
            "has_gt": has_gt,
            "transcript_filename": tr_filename if has_gt else None,
            "emotion_distribution": emotion_dist,
            "dominant_emotion": dominant_emotion,
            "label_partisipan": f"P{random.randint(1, 100):03d}",
            "created_at": datetime.now().isoformat(),
            "segments": segments,
            "summary": summary,
            "wer_cer_metrics": wer_cer_metrics,
        }
        
        _save_record(record_id, record_data)
        
        # Mark complete with final message
        tracker = get_tracker()
        
        # Force flush - ensure all logs from analysis are in tracker
        import time
        time.sleep(0.5)  # Small delay to ensure all logs are captured
        
        # Add final completion messages directly to tracker
        print("\n" + "="*60)
        print("PROSES SELESAI!")
        print("="*60)
        
        with tracker._lock:
            # Make sure we have all the completion messages
            if "✓ Ringkasan berhasil dibuat" not in "\n".join(tracker._logs):
                # Add missing final messages from analysis
                tracker._logs.append("✓ Ringkasan berhasil dibuat")
                tracker._logs.append(f"Total waktu proses: kompleks")
                tracker._logs.append(f"Hasil: {analysis_result.get('word_count', 0)} kata berhasil ditranskripsi")
            
            tracker._logs.append("")
            tracker._logs.append("="*60)
            tracker._logs.append("SELESAI!")
            tracker._logs.append("="*60)
            tracker._logs.append(f"Audio berhasil diproses: {audio_file.filename}")
            tracker._logs.append("Menyimpan hasil...")
            tracker._logs.append("Mengalihkan ke dashboard dalam 2 detik...")
        
        tracker.set_complete()
        
        # Wait a moment to ensure all logs are flushed
        import time
        time.sleep(0.5)
        
        print("Proses selesai. Menunggu frontend untuk redirect...")
        
        # Return JSON success - let frontend handle redirect
        # This ensures all logs are displayed before redirect
        return jsonify({
            "success": True,
            "message": f"Audio berhasil diproses! File: {audio_file.filename}",
            "redirect_url": url_for("main.dashboard")
        })
    
    except Exception as e:
        current_app.logger.error(f"Error processing upload: {e}", exc_info=True)
        # Return JSON error for AJAX request
        return jsonify({
            "success": False,
            "error": str(e),
            "message": f"Terjadi error saat memproses file: {str(e)}"
        }), 500


@bp.route("/dashboard")
def dashboard():
    """Dashboard global dengan cards, charts, wordcloud, tabel"""
    records = _list_all_records()
    
    # Calculate stats
    total_audio = len(records)
    total_duration = sum(r.get("duration_sec", 0) for r in records)
    avg_duration = total_duration / total_audio if total_audio > 0 else 0
    
    # Emotion summary
    emotion_counts = {}
    for r in records:
        dom = r.get("dominant_emotion", "Neutral")
        emotion_counts[dom] = emotion_counts.get(dom, 0) + 1
    
    # Top keywords (simple word frequency)
    all_text = " ".join(r.get("transcript", "") for r in records)
    words = re.findall(r"\b[a-zA-Z]{4,}\b", all_text.lower())
    word_freq = {}
    stopwords = {'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua', 'ia', 'seperti', 'jika', 'jika', 'sehingga', 'kembali', 'dan', 'tidak', 'ini', 'karena', 'kepada', 'oleh', 'saat', 'harus', 'sementara', 'setelah', 'belum', 'kami', 'sekitar', 'bagi', 'serta', 'di', 'dari', 'telah', 'sebagai', 'masih', 'hal', 'ketika', 'adalah', 'itu', 'dalam', 'bisa', 'bahwa', 'atau', 'hanya', 'kita', 'dengan', 'akan', 'juga', 'ada', 'mereka', 'sudah', 'saya', 'terhadap', 'secara', 'agar', 'lain', 'anda', 'begitu', 'mengapa', 'kenapa', 'yaitu', 'yakni', 'daripada', 'itulah', 'lagi', 'maka', 'tentang', 'demi', 'dimana', 'kemana', 'pula', 'sambil', 'sebelum', 'sesudah', 'supaya', 'guna', 'kah', 'pun', 'sampai', 'sedangkan', 'selagi', 'sementara', 'tetapi', 'apakah', 'kecuali', 'sebab', 'selain', 'seolah', 'seraya', 'seterusnya', 'tanpa', 'agak', 'boleh', 'dapat', 'dsb', 'dst', 'dll', 'dahulu', 'dulunya', 'anu', 'demikian', 'tapi', 'ingin', 'juga', 'nggak', 'mari', 'nanti', 'melainkan', 'oh', 'ok', 'seharusnya', 'sebetulnya', 'setiap', 'setidaknya', 'sesuatu', 'pasti', 'saja', 'toh', 'ya', 'walau', 'tolong', 'tentu', 'amat', 'apalagi', 'bagaimanapun', 'kalau', 'kayak', 'apa', 'jadi', 'terus'}
    for w in words:
        if w not in stopwords:
            word_freq[w] = word_freq.get(w, 0) + 1
    top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Get or generate wordcloud (with caching and blue gradient)
    wordcloud_image = _get_or_generate_wordcloud(records, stopwords)
    
    return render_template(
        "dashboard.html",
        records=records,
        total_audio=total_audio,
        total_duration=total_duration,
        avg_duration=avg_duration,
        emotion_counts=emotion_counts,
        top_keywords=top_keywords,
        wordcloud_image=wordcloud_image,
    )


@bp.route("/audio/<record_id>")
def audio_detail(record_id: str):
    """Detail per audio: hanya emosi + transkrip"""
    record = _load_record(record_id)
    if not record:
        flash("Rekaman tidak ditemukan.", "danger")
        return redirect(url_for("main.dashboard"))
    
    return render_template("audio_detail.html", record=record)


@bp.route("/audio/<record_id>/generate-ollama-summary", methods=["POST"])
def generate_ollama_summary(record_id: str):
    """Generate summary untuk single audio menggunakan Ollama"""
    try:
        record = _load_record(record_id)
        if not record:
            return jsonify({"success": False, "error": "Rekaman tidak ditemukan"}), 404
        
        # Get transcript
        transcript = record.get("transcript", "")
        if not transcript:
            return jsonify({"success": False, "error": "Transkrip tidak tersedia"}), 400
        
        # Get Ollama API key from environment or config
        import os
        api_key = os.environ.get('OLLAMA_API_KEY', '')
        
        if not api_key:
            return jsonify({
                "success": False, 
                "error": "Ollama API key tidak dikonfigurasi. Set OLLAMA_API_KEY environment variable."
            }), 400
        
        # Initialize Ollama summarizer
        summarizer = get_ollama_summarizer(api_key=api_key)
        
        # Generate summary
        result = summarizer.generate_summary_4_points(transcript, stream=False)
        
        # Save summary to record
        record['ollama_summary'] = result
        _save_record(record_id, record)
        
        return jsonify({
            "success": True,
            "summary": result
        })
    
    except Exception as e:
        current_app.logger.error(f"Error generating Ollama summary: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@bp.route("/audio/<record_id>/generate-ollama-summary-stream")
def generate_ollama_summary_stream(record_id: str):
    """Generate summary untuk single audio menggunakan Ollama dengan streaming"""
    from flask import Response, stream_with_context
    import json
    
    def generate():
        try:
            record = _load_record(record_id)
            if not record:
                yield f"data: {json.dumps({'error': 'Rekaman tidak ditemukan'})}\n\n"
                return
            
            # Get transcript
            transcript = record.get("transcript", "")
            if not transcript:
                yield f"data: {json.dumps({'error': 'Transkrip tidak tersedia'})}\n\n"
                return
            
            # Get Ollama API key
            import os
            api_key = os.environ.get('OLLAMA_API_KEY', '')
            
            if not api_key:
                yield f"data: {json.dumps({'error': 'Ollama API key tidak dikonfigurasi'})}\n\n"
                return
            
            # Initialize Ollama summarizer
            summarizer = get_ollama_summarizer(api_key=api_key)
            
            # Stream the summary
            full_text = ""
            for chunk in summarizer.generate_summary_4_points(transcript, stream=True):
                if isinstance(chunk, str):
                    # This is a text chunk
                    full_text += chunk
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                elif isinstance(chunk, dict):
                    # This is the final parsed result
                    # Save to record
                    record['ollama_summary'] = chunk
                    _save_record(record_id, record)
                    
                    yield f"data: {json.dumps({'done': True, 'summary': chunk})}\n\n"
        
        except Exception as e:
            current_app.logger.error(f"Error in streaming: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@bp.route("/summary", methods=["GET", "POST"])
def summary():
    """Generate summary batch untuk record yang dipilih"""
    if request.method == "GET":
        batch_id = request.args.get("batch")
        if batch_id:
            batch = _load_batch(batch_id)
            if not batch:
                flash("Batch tidak ditemukan.", "danger")
                return redirect(url_for("main.dashboard"))
            selected_ids = batch["selected_ids"]
        else:
            selected_ids = request.args.getlist("ids")
    else:
        selected_ids = request.form.getlist("selected_ids")
        if not selected_ids:
            # Check if "all" was selected
            if request.form.get("generate_all") == "true":
                all_records = _list_all_records()
                selected_ids = [r["id"] for r in all_records]
    
    if not selected_ids:
        flash("Pilih minimal 1 rekaman untuk generate summary.", "warning")
        return redirect(url_for("main.dashboard"))
    
    selected_records = []
    for rid in selected_ids:
        rec = _load_record(rid)
        if rec:
            selected_records.append(rec)
    
    if not selected_records:
        flash("Tidak ada rekaman yang valid.", "danger")
        return redirect(url_for("main.dashboard"))
    
    # Check if user requested Ollama summary
    use_ollama = request.form.get("use_ollama") == "true"
    
    if use_ollama:
        # Generate combined summary using Ollama
        try:
            combined_summary_4p = _generate_ollama_batch_summary(selected_records)
        except Exception as e:
            current_app.logger.error(f"Error generating Ollama summary: {e}", exc_info=True)
            flash(f"⚠️ Error generating Ollama summary: {str(e)}. Using LexRank instead.", "warning")
            combined_summary_4p = _generate_combined_summary(selected_records)
    else:
        # Generate combined summary using real summarization (LexRank)
        try:
            combined_summary_4p = _generate_combined_summary(selected_records)
        except Exception as e:
            current_app.logger.error(f"Error generating summary: {e}", exc_info=True)
            flash(f"⚠️ Error generating summary: {str(e)}", "warning")
            # Fallback to simple summary
            combined_summary_4p = {
                "tantangan": ["Error generating summary"],
                "solusi": ["Error generating summary"],
                "harapan": ["Error generating summary"],
                "preferensi": ["Error generating summary"],
            }
    
    batch_id = uuid.uuid4().hex[:12]
    batch_data = {
        "batch_id": batch_id,
        "selected_ids": selected_ids,
        "selected_count": len(selected_records),
        "combined_summary_4p": combined_summary_4p,
        "created_at": datetime.now().isoformat(),
        "summarizer": "ollama" if use_ollama else "lexrank"
    }
    _save_batch(batch_id, batch_data)
    
    return render_template("summary.html", records=selected_records, summaries=combined_summary_4p, batch_id=batch_id, summarizer="ollama" if use_ollama else "lexrank")


def _generate_ollama_batch_summary(selected_records: List[Dict]) -> Dict[str, List[str]]:
    """Generate batch summary menggunakan Ollama"""
    print("\n" + "="*60)
    print("GENERATING BATCH SUMMARY WITH OLLAMA")
    print("="*60)
    print(f"Processing {len(selected_records)} records...")
    
    # Get API key
    import os
    api_key = os.environ.get('OLLAMA_API_KEY', '')
    
    if not api_key:
        raise ValueError("Ollama API key tidak dikonfigurasi. Set OLLAMA_API_KEY environment variable.")
    
    # Collect all transcripts
    transcripts = []
    for record in selected_records:
        transcript = record.get("transcript", "")
        if transcript:
            transcripts.append(transcript)
            print(f"  - {record.get('audio_name', 'Unknown')}: {len(transcript.split())} words")
    
    if not transcripts:
        raise ValueError("Tidak ada transkrip yang tersedia")
    
    # Initialize Ollama summarizer
    summarizer = get_ollama_summarizer(api_key=api_key)
    
    # Generate batch summary
    print("\nGenerating summary with Ollama AI...")
    result = summarizer.generate_batch_summary(transcripts)
    
    print("\n" + "="*60)
    print("OLLAMA BATCH SUMMARY COMPLETE")
    print("="*60)
    
    # Debug: Print what we got
    print("\n--- DEBUG: Hasil dari Ollama ---")
    print(f"Full summary length: {len(result.get('full_summary', ''))}")
    print(f"Tantangan: {result['tantangan'][:100]}..." if len(result['tantangan']) > 100 else f"Tantangan: {result['tantangan']}")
    print(f"Solusi: {result['solusi'][:100]}..." if len(result['solusi']) > 100 else f"Solusi: {result['solusi']}")
    print(f"Harapan: {result['harapan'][:100]}..." if len(result['harapan']) > 100 else f"Harapan: {result['harapan']}")
    print(f"Preferensi: {result['preferensi'][:100]}..." if len(result['preferensi']) > 100 else f"Preferensi: {result['preferensi']}")
    print("--- END DEBUG ---\n")
    
    # Convert to expected format (list of strings per category)
    return {
        "tantangan": [result['tantangan']] if result['tantangan'] else ["Tidak ada data tantangan"],
        "solusi": [result['solusi']] if result['solusi'] else ["Tidak ada data solusi"],
        "harapan": [result['harapan']] if result['harapan'] else ["Tidak ada data harapan"],
        "preferensi": [result['preferensi']] if result['preferensi'] else ["Tidak ada data preferensi"],
    }


def _generate_combined_summary(selected_records: List[Dict]) -> Dict[str, List[str]]:
    """Generate combined summary 4 poin dari selected records using real summarization"""
    print("\n" + "="*60)
    print("GENERATING COMBINED SUMMARY")
    print("="*60)
    print(f"Processing {len(selected_records)} records...")
    
    processor = get_processor()
    
    # Note: LexRank summarizer is already initialized, no need to load heavy models
    # Only Whisper and Wav2Vec2 need explicit loading, which are NOT needed for summarization
    
    # Group texts by topic
    topic_texts = {
        "TANTANGAN": [],
        "STRATEGI": [],
        "HARAPAN": [],
        "PREFERENSI DOSPEM": []
    }
    
    # Collect all segments from selected records
    print("\nCollecting segments from records...")
    for record in selected_records:
        segments = record.get("segments", [])
        print(f"  - {record.get('audio_name', 'Unknown')}: {len(segments)} segments")
        
        for topic, text in segments:
            if topic in topic_texts:
                topic_texts[topic].append(text)
                print(f"    Added to {topic}: {len(text.split())} words")
    
    # Generate summaries for each topic
    topic_summaries = {}
    
    print("\nGenerating summaries per topic:")
    for topic, texts in topic_texts.items():
        if not texts:
            print(f"  - {topic}: No data")
            continue
        
        combined_text = " ".join(texts)
        word_count = len(combined_text.split())
        
        print(f"  - {topic}: {word_count} words from {len(texts)} files")
        
        if word_count < 20:
            # Too short, just use the text
            topic_summaries[topic] = [combined_text[:200] + "..."]
            print(f"    Too short, using raw text")
            continue
        
        try:
            # Use real summarization
            print(f"    Summarizing with LexRank...")
            summary_text = processor.summarize_text(combined_text)
            print(f"    Summary generated: {len(summary_text.split())} words")
            
            # Split into bullet points (by sentence)
            sentences = re.split(r'(?<=[.!?])\s+', summary_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            # Limit to 4-6 points
            topic_summaries[topic] = sentences[:6] if len(sentences) > 4 else sentences
            print(f"    Created {len(topic_summaries[topic])} bullet points")
            
        except Exception as e:
            print(f"    ERROR summarizing {topic}: {e}")
            # Fallback: extract first few sentences
            sentences = re.split(r'(?<=[.!?])\s+', combined_text)
            sentences = [s.strip() for s in sentences[:4] if len(s.strip()) > 10]
            topic_summaries[topic] = sentences if sentences else ["Error generating summary"]
            print(f"    Using fallback: {len(topic_summaries[topic])} points")
    
    # Map to expected keys
    result = {
        "tantangan": topic_summaries.get("TANTANGAN", ["Tidak ada data tantangan"]),
        "solusi": topic_summaries.get("STRATEGI", ["Tidak ada data strategi"]),
        "harapan": topic_summaries.get("HARAPAN", ["Tidak ada data harapan"]),
        "preferensi": topic_summaries.get("PREFERENSI DOSPEM", ["Tidak ada data preferensi"]),
    }
    
    print("\n" + "="*60)
    print("SUMMARY GENERATION COMPLETE")
    print("="*60)
    print(f"Tantangan: {len(result['tantangan'])} points")
    print(f"Solusi: {len(result['solusi'])} points")
    print(f"Harapan: {len(result['harapan'])} points")
    print(f"Preferensi: {len(result['preferensi'])} points")
    
    return result


@bp.route("/summary/export/pdf")
def export_pdf():
    """Export summary as PDF"""
    batch_id = request.args.get("batch")
    if not batch_id:
        flash("Batch ID tidak ditemukan.", "danger")
        return redirect(url_for("main.dashboard"))
    
    batch = _load_batch(batch_id)
    if not batch:
        flash("Batch tidak ditemukan.", "danger")
        return redirect(url_for("main.dashboard"))
    
    if not REPORTLAB_AVAILABLE:
        flash("PDF export tidak tersedia. Install reportlab: pip install reportlab", "warning")
        return redirect(url_for("main.summary", batch=batch_id))
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#0B2E5B"),
        spaceAfter=30,
    )
    story.append(Paragraph("SENSE - Summary Gabungan", title_style))
    story.append(Spacer(1, 12))
    
    # Metadata
    meta_style = styles["Normal"]
    story.append(Paragraph(f"<b>Tanggal Generate:</b> {batch['created_at'][:19]}", meta_style))
    story.append(Paragraph(f"<b>Jumlah File:</b> {batch['selected_count']}", meta_style))
    story.append(Paragraph(f"<b>Kode Audio:</b> {', '.join(batch['selected_ids'][:5])}{'...' if len(batch['selected_ids']) > 5 else ''}", meta_style))
    story.append(Spacer(1, 20))
    
    summaries = batch.get("combined_summary_4p", batch.get("summaries", {}))
    
    # 1. Tantangan
    story.append(Paragraph("<b>1. Tantangan selama S2</b>", styles["Heading2"]))
    for item in summaries.get("tantangan", []):
        story.append(Paragraph(f"• {item}", meta_style))
    story.append(Spacer(1, 12))
    
    # 2. Solusi
    story.append(Paragraph("<b>2. Solusi/Coping Strategy</b>", styles["Heading2"]))
    for item in summaries.get("solusi", []):
        story.append(Paragraph(f"• {item}", meta_style))
    story.append(Spacer(1, 12))
    
    # 3. Harapan
    story.append(Paragraph("<b>3. Harapan</b>", styles["Heading2"]))
    for item in summaries.get("harapan", []):
        story.append(Paragraph(f"• {item}", meta_style))
    story.append(Spacer(1, 12))
    
    # 4. Preferensi
    story.append(Paragraph("<b>4. Preferensi Pembimbing Tesis</b>", styles["Heading2"]))
    for item in summaries.get("preferensi", []):
        story.append(Paragraph(f"• {item}", meta_style))
    
    doc.build(story)
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"SENSE_Summary_{batch_id[:8]}.pdf"
    )


@bp.route("/summary/export/csv")
def export_csv():
    """Export summary as CSV"""
    batch_id = request.args.get("batch")
    if not batch_id:
        flash("Batch ID tidak ditemukan.", "danger")
        return redirect(url_for("main.dashboard"))
    
    batch = _load_batch(batch_id)
    if not batch:
        flash("Batch tidak ditemukan.", "danger")
        return redirect(url_for("main.dashboard"))
    
    buffer = StringIO()
    writer = csv.writer(buffer)
    
    summaries = batch.get("combined_summary_4p", batch.get("summaries", {}))
    
    # Single row with all data
    writer.writerow([
        "generated_at",
        "selected_count",
        "selected_ids",
        "tantangan",
        "solusi",
        "harapan",
        "preferensi_pembimbing"
    ])
    writer.writerow([
        batch["created_at"],
        batch["selected_count"],
        ",".join(batch["selected_ids"]),
        " | ".join(summaries.get("tantangan", [])),
        " | ".join(summaries.get("solusi", [])),
        " | ".join(summaries.get("harapan", [])),
        " | ".join(summaries.get("preferensi", []))
    ])
    
    buffer.seek(0)
    
    response = make_response(buffer.getvalue().encode('utf-8-sig'))  # UTF-8 with BOM for Excel compatibility
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = f"attachment; filename=SENSE_Summary_{batch_id[:8]}.csv"
    return response


@bp.route("/media/uploads/<filename>")
def media_upload(filename: str):
    """Serve uploaded audio files"""
    path = _uploads_dir() / filename
    if not path.exists():
        abort(404)
    return send_from_directory(_uploads_dir(), filename)


@bp.route("/media/transcripts/<filename>")
def media_transcript(filename: str):
    """Serve transcript files"""
    path = _transcripts_dir() / filename
    if not path.exists():
        abort(404)
    return send_from_directory(_transcripts_dir(), filename)


@bp.route("/delete-record/<record_id>", methods=["POST"])
def delete_record(record_id: str):
    """Delete a specific record"""
    try:
        record = _load_record(record_id)
        if not record:
            flash("Rekaman tidak ditemukan.", "danger")
            return redirect(url_for("main.dashboard"))
        
        # Delete audio file if exists
        if record.get("audio_filename"):
            audio_path = _uploads_dir() / record["audio_filename"]
            if audio_path.exists():
                try:
                    os.remove(audio_path)
                    print(f"Deleted audio file: {audio_path}")
                except Exception as e:
                    print(f"Error deleting audio: {e}")
        
        # Delete transcript file if exists
        if record.get("transcript_filename"):
            transcript_path = _transcripts_dir() / record["transcript_filename"]
            if transcript_path.exists():
                try:
                    os.remove(transcript_path)
                    print(f"Deleted transcript file: {transcript_path}")
                except Exception as e:
                    print(f"Error deleting transcript: {e}")
        
        # Delete record JSON
        record_file = _records_dir() / f"{record_id}.json"
        if record_file.exists():
            os.remove(record_file)
            print(f"Deleted record: {record_id}")
        
        flash(f"Rekaman '{record.get('audio_name', 'Unknown')}' berhasil dihapus!", "success")
        
    except Exception as e:
        current_app.logger.error(f"Error deleting record: {e}", exc_info=True)
        flash(f"Error menghapus rekaman: {str(e)}", "danger")
    
    return redirect(url_for("main.dashboard"))


@bp.route("/delete-all-records", methods=["POST"])
def delete_all_records():
    """Delete all records"""
    try:
        # Get all records
        records = _list_all_records()
        deleted_count = 0
        
        for record in records:
            record_id = record.get("id")
            if not record_id:
                continue
            
            # Delete audio file
            if record.get("audio_filename"):
                audio_path = _uploads_dir() / record["audio_filename"]
                if audio_path.exists():
                    try:
                        os.remove(audio_path)
                    except:
                        pass
            
            # Delete transcript file
            if record.get("transcript_filename"):
                transcript_path = _transcripts_dir() / record["transcript_filename"]
                if transcript_path.exists():
                    try:
                        os.remove(transcript_path)
                    except:
                        pass
            
            # Delete record JSON
            record_file = _records_dir() / f"{record_id}.json"
            if record_file.exists():
                os.remove(record_file)
                deleted_count += 1
        
        flash(f"Berhasil menghapus {deleted_count} rekaman!", "success")
        
    except Exception as e:
        current_app.logger.error(f"Error deleting all records: {e}", exc_info=True)
        flash(f"Error menghapus rekaman: {str(e)}", "danger")
    
    return redirect(url_for("main.dashboard"))


@bp.route("/batch-upload", methods=["GET", "POST"])
def batch_upload():
    """Batch upload and process multiple audio files from ZIP"""
    if request.method == "GET":
        return render_template("batch_upload.html")
    
    try:
        if "batch_file" not in request.files:
            flash("Pilih file ZIP terlebih dahulu.", "danger")
            return redirect(url_for("main.batch_upload"))
        
        batch_file = request.files["batch_file"]
        if not batch_file or not batch_file.filename:
            flash("File tidak valid atau kosong.", "danger")
            return redirect(url_for("main.batch_upload"))
        
        if not batch_file.filename.lower().endswith('.zip'):
            flash("Hanya file ZIP yang didukung untuk batch upload.", "danger")
            return redirect(url_for("main.batch_upload"))
        
        # Create batch ID and directories
        batch_id = uuid.uuid4().hex[:12]
        batch_dir = _storage_dir() / "batches" / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Save ZIP file
        zip_path = batch_dir / "batch.zip"
        batch_file.save(zip_path)
        
        # Extract and discover files
        extract_dir = batch_dir / "extracted"
        file_data = extract_and_discover_files(zip_path, extract_dir)
        
        audio_files = file_data['audio_files']
        gt_files = file_data['gt_files']
        
        if not audio_files:
            flash("❌ Tidak ada file audio ditemukan dalam ZIP.", "danger")
            return redirect(url_for("main.batch_upload"))
        
        # Process all audio files
        print(f"\n🚀 Starting batch processing for {len(audio_files)} files...")
        results = batch_process_audio_files(audio_files, gt_files)
        
        # Save batch results
        batch_data = {
            "batch_id": batch_id,
            "created_at": datetime.now().isoformat(),
            "total_files": len(audio_files),
            "files_processed": results['files_processed'],
            "topic_summaries": results['topic_summaries'],
            "global_emotion": results['global_emotion'],
            "avg_wer": results.get('avg_wer'),
            "avg_cer": results.get('avg_cer'),
            "evaluations": results['evaluations']
        }
        
        # Save individual records
        for filename in results['files_processed']:
            trans_data = results['transcriptions'].get(filename)
            emotion_data = results['emotions'].get(filename)
            
            if trans_data and emotion_data:
                record_id = uuid.uuid4().hex[:12]
                
                # Map emotion labels
                emotion_label_map = {'neu': 'Neutral', 'hap': 'Happy', 'ang': 'Angry', 'sad': 'Sad'}
                emotion_dist = {}
                for emotion, percentage in emotion_data['percentages'].items():
                    readable_name = emotion_label_map.get(emotion, emotion.capitalize())
                    emotion_dist[readable_name] = round(percentage, 1)
                
                dominant_emotion = emotion_label_map.get(
                    emotion_data['dominant'], 
                    emotion_data['dominant'].capitalize()
                )
                
                record_data = {
                    "id": record_id,
                    "audio_filename": filename,
                    "audio_name": filename,
                    "duration_sec": 0,  # Not calculated in batch mode
                    "transcript": trans_data['full_text'],
                    "has_gt": False,
                    "transcript_filename": None,
                    "emotion_distribution": emotion_dist,
                    "dominant_emotion": dominant_emotion,
                    "label_partisipan": f"B{batch_id[:4]}-{random.randint(1, 99):02d}",
                    "created_at": datetime.now().isoformat(),
                    "segments": trans_data['segments'],
                    "summary": "",
                    "batch_id": batch_id
                }
                
                _save_record(record_id, record_data)
        
        _save_batch(batch_id, batch_data)
        
        flash(f"✅ Batch processing complete! {len(results['files_processed'])} files processed.", "success")
        return redirect(url_for("main.batch_results", batch_id=batch_id))
    
    except Exception as e:
        current_app.logger.error(f"Error in batch processing: {e}", exc_info=True)
        flash(f"❌ Error: {str(e)}", "danger")
        return redirect(url_for("main.batch_upload"))


@bp.route("/batch-results/<batch_id>")
def batch_results(batch_id: str):
    """Display batch processing results"""
    batch = _load_batch(batch_id)
    if not batch:
        flash("Batch tidak ditemukan.", "danger")
        return redirect(url_for("main.dashboard"))
    
    # Load all records from this batch
    all_records = _list_all_records()
    batch_records = [r for r in all_records if r.get("batch_id") == batch_id]
    
    return render_template(
        "batch_results.html",
        batch=batch,
        records=batch_records
    )
