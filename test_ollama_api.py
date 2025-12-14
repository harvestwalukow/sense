"""
Test script untuk verifikasi Ollama API key
"""
import os
from dotenv import load_dotenv
from ollama import Client

# Load environment variables
load_dotenv()

def test_ollama_connection():
    """Test koneksi ke Ollama API"""
    
    # Get API key
    api_key = os.environ.get('OLLAMA_API_KEY', '')
    
    if not api_key:
        print("❌ ERROR: OLLAMA_API_KEY tidak ditemukan di environment variable")
        print("\nSolusi:")
        print("1. Buat file .env di root project")
        print("2. Tambahkan: OLLAMA_API_KEY=your_api_key_here")
        print("3. Restart aplikasi")
        return False
    
    print(f"✓ API Key ditemukan: {api_key[:20]}...{api_key[-10:]}")
    
    # Test connection
    try:
        print("\n⏳ Testing koneksi ke Ollama API...")
        
        client = Client(
            host="https://ollama.com",
            headers={'Authorization': f'Bearer {api_key}'}
        )
        
        # Simple test message
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one word."}
        ]
        
        response = client.chat('gpt-oss:120b', messages=messages, stream=False)
        
        print("✅ Koneksi BERHASIL!")
        print(f"Response: {response['message']['content']}")
        print("\n🎉 API Key valid dan siap digunakan!")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Koneksi GAGAL!")
        print(f"Error: {error_msg}")
        
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            print("\n🔍 Diagnosa: API Key TIDAK VALID")
            print("\nKemungkinan penyebab:")
            print("1. API key salah atau typo")
            print("2. API key expired")
            print("3. API key belum activated")
            print("\nSolusi:")
            print("1. Login ke https://ollama.com")
            print("2. Generate API key baru")
            print("3. Update file .env dengan API key yang baru")
            print("4. Restart script ini untuk test lagi")
        elif "404" in error_msg:
            print("\n🔍 Diagnosa: Model tidak ditemukan")
            print("Model 'gpt-oss:120b' mungkin tidak tersedia")
        else:
            print("\n🔍 Diagnosa: Error koneksi atau konfigurasi")
            print("Cek koneksi internet dan konfigurasi API")
        
        return False

if __name__ == "__main__":
    print("="*60)
    print("OLLAMA API KEY TESTER")
    print("="*60)
    
    test_ollama_connection()
    
    print("\n" + "="*60)
    print("Setelah fix, jalankan aplikasi Flask dengan: python run.py")
    print("="*60)
