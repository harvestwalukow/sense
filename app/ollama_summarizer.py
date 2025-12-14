"""
Ollama Integration untuk Generate Summary
Menggunakan Ollama API untuk meringkas transkrip wawancara
"""
from __future__ import annotations

import os
from typing import Dict, Optional
from ollama import Client


class OllamaSummarizer:
    """Generate summary menggunakan Ollama LLM"""
    
    def __init__(self, host: str = "https://ollama.com", api_key: Optional[str] = None):
        """
        Initialize Ollama client
        
        Args:
            host: Ollama API host URL
            api_key: API key untuk autentikasi (default dari environment variable)
        """
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.environ.get('OLLAMA_API_KEY', '')
        
        if not self.api_key:
            raise ValueError(
                "Ollama API key tidak ditemukan. "
                "Set OLLAMA_API_KEY environment variable atau pass api_key parameter."
            )
        
        self.client = Client(
            host=host,
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
        
        self.model = 'gpt-oss:120b'
    
    def generate_summary_4_points(self, transkrip: str, stream: bool = False) -> Dict[str, any]:
        """
        Generate summary dengan 4 poin dari transkrip wawancara
        
        Args:
            transkrip: Teks transkrip wawancara
            stream: Apakah menggunakan streaming response
        
        Returns:
            Dict dengan keys: 'full_summary', 'tantangan', 'solusi', 'harapan', 'preferensi'
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "Kamu adalah asisten yang bertugas meringkas transkrip wawancara. "
                    "Ringkas jawaban narasumber menjadi 4 poin berikut:\n"
                    "1. Tantangan selama S2\n"
                    "2. Solusi / Coping Strategy\n"
                    "3. Harapan setelah lulus\n"
                    "4. Preferensi Pembimbing Tesis\n\n"
                    "Gunakan bahasa Indonesia yang jelas, ringkas, dan berbentuk paragraf. "
                    "Format output harus jelas dengan menggunakan heading untuk setiap poin:\n"
                    "## 1. Tantangan selama S2\n[paragraf ringkasan]\n\n"
                    "## 2. Solusi / Coping Strategy\n[paragraf ringkasan]\n\n"
                    "## 3. Harapan setelah lulus\n[paragraf ringkasan]\n\n"
                    "## 4. Preferensi Pembimbing Tesis\n[paragraf ringkasan]"
                )
            },
            {
                "role": "user",
                "content": f"Berikut adalah transkrip wawancara yang perlu diringkas:\n\n{transkrip}"
            }
        ]
        
        if stream:
            # Streaming response
            full_text = ""
            for part in self.client.chat(self.model, messages=messages, stream=True):
                chunk = part['message']['content']
                full_text += chunk
                yield chunk
            
            # After streaming complete, parse the result
            result = self._parse_summary(full_text)
            result['full_summary'] = full_text
            yield result
        else:
            # Non-streaming response
            response = self.client.chat(self.model, messages=messages, stream=False)
            full_summary = response['message']['content']
            
            result = self._parse_summary(full_summary)
            result['full_summary'] = full_summary
            
            return result
    
    def _parse_summary(self, summary_text: str) -> Dict[str, str]:
        """
        Parse summary text menjadi 4 komponen terpisah
        
        Args:
            summary_text: Full summary text dari Ollama
        
        Returns:
            Dict dengan keys: tantangan, solusi, harapan, preferensi
        """
        import re
        
        result = {
            'tantangan': '',
            'solusi': '',
            'harapan': '',
            'preferensi': ''
        }
        
        # Pattern untuk extract setiap section - multiple variations
        patterns = {
            'tantangan': [
                r'##?\s*1\.?\s*Tantangan.*?\n+(.*?)(?=##?\s*2\.|$)',
                r'1\.?\s*Tantangan[^\n]*\n+(.*?)(?=2\.|$)',
                r'Tantangan[^\n]*:?\n+(.*?)(?=Solusi|2\.|$)',
            ],
            'solusi': [
                r'##?\s*2\.?\s*Solusi.*?\n+(.*?)(?=##?\s*3\.|$)',
                r'2\.?\s*Solusi[^\n]*\n+(.*?)(?=3\.|$)',
                r'Solusi[^\n]*:?\n+(.*?)(?=Harapan|3\.|$)',
            ],
            'harapan': [
                r'##?\s*3\.?\s*Harapan.*?\n+(.*?)(?=##?\s*4\.|$)',
                r'3\.?\s*Harapan[^\n]*\n+(.*?)(?=4\.|$)',
                r'Harapan[^\n]*:?\n+(.*?)(?=Preferensi|4\.|$)',
            ],
            'preferensi': [
                r'##?\s*4\.?\s*Preferensi.*?\n+(.*?)$',
                r'4\.?\s*Preferensi[^\n]*\n+(.*?)$',
                r'Preferensi[^\n]*:?\n+(.*?)$',
            ]
        }
        
        for key, pattern_list in patterns.items():
            found = False
            for pattern in pattern_list:
                match = re.search(pattern, summary_text, re.DOTALL | re.IGNORECASE)
                if match:
                    # Clean up the extracted text
                    text = match.group(1).strip()
                    # Remove extra newlines and spaces
                    text = re.sub(r'\n\s*\n', '\n\n', text)
                    if text and len(text) > 10:  # Make sure we got meaningful content
                        result[key] = text
                        found = True
                        break
            
            if not found:
                # Fallback: just use a portion of the full text
                result[key] = f"(Data tidak dapat diparsing dengan format standar. Lihat full summary.)"
        
        return result
    
    def generate_batch_summary(self, transcripts: list[str]) -> Dict[str, any]:
        """
        Generate summary untuk batch transkrip (combine multiple transcripts)
        
        Args:
            transcripts: List of transcript texts
        
        Returns:
            Dict dengan combined summary
        """
        # Combine all transcripts
        combined_transcript = "\n\n---\n\n".join(transcripts)
        
        # Add context to system message for batch processing
        messages = [
            {
                "role": "system",
                "content": (
                    "Kamu adalah asisten yang bertugas meringkas MULTIPLE transkrip wawancara. "
                    "Analisis SEMUA transkrip yang diberikan dan buat ringkasan GABUNGAN untuk 4 poin berikut:\n"
                    "1. Tantangan selama S2 (dari semua narasumber)\n"
                    "2. Solusi / Coping Strategy (dari semua narasumber)\n"
                    "3. Harapan setelah lulus (dari semua narasumber)\n"
                    "4. Preferensi Pembimbing Tesis (dari semua narasumber)\n\n"
                    "Gunakan bahasa Indonesia yang jelas, ringkas, dan berbentuk paragraf. "
                    "Identifikasi pola umum dan insight dari semua narasumber.\n"
                    "Format output harus jelas dengan menggunakan heading untuk setiap poin."
                )
            },
            {
                "role": "user",
                "content": f"Berikut adalah {len(transcripts)} transkrip wawancara yang perlu diringkas:\n\n{combined_transcript}"
            }
        ]
        
        response = self.client.chat(self.model, messages=messages, stream=False)
        full_summary = response['message']['content']
        
        result = self._parse_summary(full_summary)
        result['full_summary'] = full_summary
        result['total_transcripts'] = len(transcripts)
        
        return result


# Singleton instance
_summarizer_instance = None

def get_ollama_summarizer(host: str = "https://ollama.com", api_key: Optional[str] = None) -> OllamaSummarizer:
    """Get or create the global Ollama summarizer instance"""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = OllamaSummarizer(host=host, api_key=api_key)
    return _summarizer_instance
