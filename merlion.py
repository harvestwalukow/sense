import os
from transformers import WhisperFeatureExtractor, AutoModelForAudioClassification

cache_dir = os.path.join(os.getcwd(), "model_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir

print("Downloading MERaLiON-SER-v1...")

model_name = "MERaLiON/MERaLiON-SER-v1"

feature_extractor = WhisperFeatureExtractor.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForAudioClassification.from_pretrained(
    model_name,
    trust_remote_code=True
)

print("✓ MERaLiON-SER-v1 downloaded successfully")
print(f"✓ Cached at: {cache_dir}")
