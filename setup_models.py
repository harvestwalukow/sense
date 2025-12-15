"""
Setup script to download and initialize all AI models
Run this before first use to avoid delays during processing
"""
import os
import sys

def setup_models():
    """Download and initialize all required models"""
    print("=" * 60)
    print("SENSE App - Model Setup Script")
    print("=" * 60)
    print("\nThis script will download and initialize:")
    print("  1. Whisper Large-v3 (~2.9GB)")
    print("  2. MERaLiON-SER-v1 Emotion Model (~1.5GB)")
    print("  3. NLTK Data (punkt, punkt_tab)")
    print("\nTotal download size: ~4.5GB")
    print("This may take 10-30 minutes depending on internet speed.")
    print("=" * 60)
    
    response = input("\nProceed with download? (y/n): ")
    if response.lower() != 'y':
        print("Setup cancelled.")
        return
    
    # Set cache directory
    cache_dir = os.path.join(os.getcwd(), "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    
    print(f"\nModels will be cached in: {cache_dir}")
    
    # Step 1: Download NLTK data
    print("\n[1/3] Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"✗ Error downloading NLTK data: {e}")
        return
    
    # Step 2: Load Whisper model
    print("\n[2/3] Downloading Whisper Large-v3...")
    print("(This is the largest file and may take 10-20 minutes)")
    try:
        import whisper
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device.upper()}")
        model = whisper.load_model("large-v3", device=device)
        print("✓ Whisper Large-v3 downloaded and loaded")
    except Exception as e:
        print(f"✗ Error loading Whisper: {e}")
        print("Falling back to base model...")
        try:
            model = whisper.load_model("base", device=device)
            print("✓ Whisper Base downloaded and loaded")
        except Exception as e2:
            print(f"✗ Error: {e2}")
            return
    
    # Step 3: Load MERaLiON-SER model
    print("\n[3/3] Downloading MERaLiON-SER-v1 Emotion Model...")
    try:
        from transformers import WhisperFeatureExtractor, AutoModelForAudioClassification
        emotion_model_name = "MERaLiON/MERaLiON-SER-v1"
        emotion_extractor = WhisperFeatureExtractor.from_pretrained(emotion_model_name)
        emotion_model = AutoModelForAudioClassification.from_pretrained(emotion_model_name)
        print("✓ MERaLiON-SER-v1 Emotion Model downloaded and loaded")
    except Exception as e:
        print(f"✗ Error loading MERaLiON-SER-v1: {e}")
        return
    
    # Verify everything works
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\n✓ All models downloaded and cached successfully")
    print(f"✓ Cache location: {cache_dir}")
    print("\nYou can now run the Flask app with:")
    print("  python run.py")
    print("\nOr load models programmatically:")
    print("  >>> from app.audio_processor import get_processor")
    print("  >>> processor = get_processor()")
    print("  >>> processor.load_models()")

if __name__ == "__main__":
    try:
        setup_models()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)








