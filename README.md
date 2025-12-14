# SENSE - Speech Emotion & Natural language Summarization Engine

🎓 **Aplikasi Analisis Audio untuk Penelitian Mahasiswa Pascasarjana**

Sistem analisis audio cerdas yang menggunakan AI untuk transkripsi, deteksi emosi, dan summarisasi otomatis dari wawancara mahasiswa S2. Dibangun dengan Flask dan terintegrasi dengan model AI terkini.

## ✨ Fitur Utama

### 🎤 Audio Processing
- **Transkripsi Otomatis**: Menggunakan OpenAI Whisper Large-v3 untuk transkripsi Bahasa Indonesia
- **Deteksi Emosi**: Wav2Vec2 untuk analisis emosi (Netral, Senang, Sedih, Marah)
- **Multi-Format Support**: MP3, WAV, M4A, AAC, OGG, FLAC, WMA

### 📊 Analysis & Insights
- **Topic Segmentation**: Otomatis membagi transkrip ke topik (Tantangan, Strategi, Harapan, Preferensi)
- **Smart Summarization**: 
  - **LexRank**: Summarization lokal (cepat, gratis)
  - **Ollama AI**: Summarization dengan GPT-OSS model (lebih akurat, berbahasa natural)
- **Emotion Distribution**: Visualisasi distribusi emosi per segmen
- **WER/CER Evaluation**: Jika ground truth tersedia

### 🔄 Real-time Progress
- Live logging untuk setiap tahap pemrosesan
- Progress tracking untuk user experience yang lebih baik
- Tampilan real-time hasil transkripsi

### 📈 Dashboard & Reporting
- Dashboard global dengan statistik
- Export summary ke PDF/CSV
- History management dengan fitur delete
- Batch processing untuk multiple files

## 🛠️ Teknologi

### Backend
- **Flask**: Web framework
- **Whisper**: Speech-to-text (OpenAI)
- **Wav2Vec2**: Emotion recognition
- **LexRank**: Text summarization (ekstraktif)
- **Ollama**: AI summarization (abstraktif - optional)
- **PyTorch**: Deep learning framework

### Frontend
- **Bootstrap 5**: UI components
- **JavaScript**: Real-time updates via polling
- **Responsive Design**: Mobile-friendly

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/<username>/flask-nlp-dashboard.git
cd flask-nlp-dashboard
```

### 2. Setup Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download AI Models (First Time Only)
```bash
python setup_models.py
```

⏱️ **Note**: Model download memakan waktu ~5-10 menit (total ~3.5 GB)

### 5. (Optional) Setup Ollama AI for Advanced Summarization

Untuk menggunakan Ollama AI summarization:

```bash
# Copy environment template
cp env.example.txt .env

# Edit .env dan isi API key Anda
# OLLAMA_API_KEY=your_api_key_here
```

📖 **Panduan lengkap**: Lihat [QUICK_START_OLLAMA.md](QUICK_START_OLLAMA.md)

### 6. Run Application
```bash
python run.py
```

Buka browser: **http://127.0.0.1:5000**

## ⚙️ Konfigurasi

### Hardware Requirements

**Minimum:**
- RAM: 8 GB
- Storage: 5 GB free space
- CPU: 4 cores

**Recommended:**
- RAM: 16 GB+
- GPU: NVIDIA with CUDA support (5-10x faster)
- Storage: 10 GB free space

### Performance

| Hardware | Processing Speed |
|----------|------------------|
| **CPU** | ~30-60 detik per menit audio |
| **GPU (CUDA)** | ~5-10 detik per menit audio |

## 📁 Struktur Project

```
flask-nlp-dashboard/
├── app/
│   ├── __init__.py
│   ├── routes.py              # Flask routes
│   ├── audio_processor.py     # AI pipeline (Whisper, Wav2Vec2, LexRank)
│   ├── ollama_summarizer.py   # Ollama AI integration (optional)
│   ├── evaluation.py          # WER/CER metrics
│   ├── batch_processor.py     # Batch processing
│   ├── progress_tracker.py    # Real-time logging
│   ├── templates/             # HTML templates
│   └── static/                # CSS/JS
├── storage/                   # User data (excluded from git)
│   ├── uploads/              # Audio files
│   ├── transcripts/          # Ground truth files
│   ├── records/              # JSON records
│   └── batches/              # Batch results
├── requirements.txt          # Python dependencies
├── run.py                    # Application entry point
└── setup_models.py          # Model downloader
```

## 🎯 Use Cases

1. **Penelitian Mahasiswa**: Analisis wawancara mahasiswa S2
2. **Sentiment Analysis**: Deteksi emosi dalam percakapan
3. **Automatic Transcription**: Konversi audio ke teks Bahasa Indonesia
4. **Topic Modeling**: Segmentasi otomatis berdasarkan topik
5. **Summary Generation**: Ringkasan otomatis dari multiple interviews

## 🔧 Troubleshooting

### Error: MemoryError saat load model
**Solusi**: Gunakan model yang lebih kecil di `app/audio_processor.py`:
```python
self.model_whisper = whisper.load_model("base")  # Instead of "large-v3"
```

### Error: CUDA out of memory
**Solusi**: Process menggunakan CPU atau gunakan batch size lebih kecil

### Processing terlalu lambat
**Solusi**: 
- Gunakan GPU dengan CUDA
- Switch ke model Whisper yang lebih kecil (base/small)
- Pastikan tidak ada aplikasi lain yang menggunakan CPU/GPU

### Error: "Ollama API key tidak dikonfigurasi"
**Solusi**: 
- Buat file `.env` di root project
- Tambahkan `OLLAMA_API_KEY=your_api_key`
- Restart aplikasi Flask

**Note**: Ollama AI bersifat **opsional**. Anda tetap bisa menggunakan LexRank summarization tanpa API key.

## 📸 Screenshots

### Dashboard
Dashboard global dengan statistik, emotion distribution, dan keyword analysis.

### Audio Processing
Real-time progress dengan live transcription output dan step-by-step logging.

### Batch Summary
Generate ringkasan kombinasi dari multiple audio files dengan 4 kategori utama.

## 📝 License

MIT License - Feel free to use for academic or commercial purposes

## 👥 Contributors

Developed for academic research at **Universitas Airlangga**

## 🙏 Acknowledgments

- **OpenAI Whisper** for speech recognition
- **Hugging Face Transformers** for emotion detection
- **LexRank** for extractive text summarization
- **Ollama** for advanced AI summarization (optional)

---

**SENSE** - Speech Emotion & Natural language Summarization Engine  
*Teknologi Sains Data – Universitas Airlangga*
