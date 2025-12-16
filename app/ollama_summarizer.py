"""
Ollama GPT-OSS Summarizer untuk Generate Summary
Menggunakan Ollama API dengan model GPT-OSS untuk meringkas transkrip wawancara
"""
from __future__ import annotations

import os
import json
from typing import Dict, Optional, Generator
import ollama


class OllamaSummarizer:
    """Generate summary menggunakan Ollama GPT-OSS model"""
    
    def __init__(self, host: str = "https://ollama.com", api_key: Optional[str] = None):
        """
        Initialize Ollama client
        
        Args:
            host: Ollama API host URL
            api_key: API key for authentication
        """
        self.host = host
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        self.model = 'gpt-oss:120b'
        
        # Initialize Ollama client
        if self.api_key:
            self.client = ollama.Client(
                host=self.host,
                headers={'Authorization': f'Bearer {self.api_key}'}
            )
        else:
            self.client = None
    
    def is_configured(self) -> bool:
        """Check if API key is configured"""
        return self.api_key is not None and len(self.api_key) > 0
    
    def generate_summary_4_points(self, transkrip: str, stream: bool = False) -> Dict[str, any]:
        """
        Generate summary dengan 4 poin dari transkrip wawancara
        
        Args:
            transkrip: Teks transkrip wawancara
            stream: If True, return generator for streaming response
        
        Returns:
            Dict dengan keys: 'full_summary', 'tantangan', 'solusi', 'harapan', 'preferensi'
        """
        if not self.is_configured():
            raise ValueError("Ollama API key tidak dikonfigurasi. Set OLLAMA_API_KEY di environment.")
        
        system_prompt = """Kamu adalah asisten yang bertugas meringkas transkrip wawancara mahasiswa S2.

PENTING: Berikan ringkasan dalam PLAIN TEXT tanpa formatting markdown apapun.
JANGAN gunakan karakter berikut: **, *, -, #, bullet points, atau simbol formatting lainnya.
Tulis dalam paragraf biasa dengan kalimat lengkap.

Berikan ringkasan dalam 4 bagian:

1. Tantangan selama S2
Ringkas kesulitan dan tantangan yang dihadapi mahasiswa selama menempuh S2 dalam satu paragraf.

2. Solusi / Coping Strategy
Ringkas strategi atau solusi yang digunakan untuk mengatasi tantangan dalam satu paragraf.

3. Harapan setelah lulus
Ringkas harapan dan ekspektasi mahasiswa setelah menyelesaikan S2 dalam satu paragraf.

4. Preferensi Pembimbing Tesis
Ringkas kriteria atau preferensi mahasiswa dalam memilih dosen pembimbing dalam satu paragraf.

Berikan ringkasan yang padat dan informatif dalam Bahasa Indonesia. Gunakan kalimat yang natural dan mudah dibaca."""
        
        user_prompt = f"""Berikut adalah transkrip wawancara yang perlu diringkas:

{transkrip}

Tolong ringkas transkrip di atas sesuai format yang diminta. Ingat: gunakan plain text saja tanpa formatting markdown."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            if stream:
                return self._generate_stream(messages)
            else:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    stream=False
                )
                
                full_summary = response['message']['content']
                parsed = self._parse_summary(full_summary)
                
                return {
                    'full_summary': full_summary,
                    'tantangan': parsed.get('tantangan', ''),
                    'solusi': parsed.get('solusi', ''),
                    'harapan': parsed.get('harapan', ''),
                    'preferensi': parsed.get('preferensi', '')
                }
        
        except Exception as e:
            print(f"Error generating Ollama summary: {e}")
            raise
    
    def _generate_stream(self, messages: list) -> Generator[Dict, None, None]:
        """Generate streaming response"""
        try:
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True
            )
            
            full_text = ""
            for chunk in stream:
                content = chunk['message']['content']
                full_text += content
                yield {"chunk": content, "done": False}
            
            # Parse final result
            parsed = self._parse_summary(full_text)
            yield {
                "done": True,
                "summary": {
                    'full_summary': full_text,
                    'tantangan': parsed.get('tantangan', ''),
                    'solusi': parsed.get('solusi', ''),
                    'harapan': parsed.get('harapan', ''),
                    'preferensi': parsed.get('preferensi', '')
                }
            }
        
        except Exception as e:
            yield {"error": str(e), "done": True}
    
    def _parse_summary(self, full_summary: str) -> Dict[str, str]:
        """Parse full summary into 4 sections"""
        result = {
            'tantangan': '',
            'solusi': '',
            'harapan': '',
            'preferensi': ''
        }
        
        import re
        
        # Try to extract each section - updated patterns for plain text format
        patterns = {
            'tantangan': r'(?:##?\s*)?1\.?\s*Tantangan[^\n]*\n(.*?)(?=(?:##?\s*)?[234]\.|$)',
            'solusi': r'(?:##?\s*)?2\.?\s*(?:Solusi|Coping)[^\n]*\n(.*?)(?=(?:##?\s*)?[34]\.|$)',
            'harapan': r'(?:##?\s*)?3\.?\s*Harapan[^\n]*\n(.*?)(?=(?:##?\s*)?4\.|$)',
            'preferensi': r'(?:##?\s*)?4\.?\s*Preferensi[^\n]*\n(.*?)(?=$)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, full_summary, re.IGNORECASE | re.DOTALL)
            if match:
                # Clean up the extracted text
                text = match.group(1).strip() if match.lastindex else match.group(0).strip()
                # Remove any header remnants
                text = re.sub(r'^##?\s*\d\.?\s*[^\n]+\n?', '', text).strip()
                # Clean markdown characters
                text = self._clean_markdown_chars(text)
                result[key] = text
        
        return result
    
    def _clean_markdown_chars(self, text: str) -> str:
        """Remove markdown formatting characters from text"""
        import re
        
        # Remove ** and * for bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        # Remove leading - or * bullet points
        text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
        
        # Remove # headers
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # Remove remaining * and **
        text = text.replace('**', '')
        text = text.replace('*', '')
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text
    
    def generate_batch_summary(self, transcripts: list) -> Dict[str, any]:
        """
        Generate summary untuk batch transkrip (combine multiple transcripts)
        
        Args:
            transcripts: List of transcript texts
        
        Returns:
            Dict dengan combined summary
        """
        # Combine all transcripts
        combined_transcript = "\n\n---\n\n".join(transcripts)
        
        # Generate summary
        result = self.generate_summary_4_points(combined_transcript)
        result['total_transcripts'] = len(transcripts)
        
        return result
    
    def extract_identities(self, transkrip: str) -> list:
        """
        Ekstrak identitas partisipan dari transkrip wawancara.
        Mendukung FGD dengan multiple partisipan.
        
        Args:
            transkrip: Teks transkrip wawancara
        
        Returns:
            List of dict dengan keys: 'nama', 'umur', 'asal_s1'
        """
        if not self.is_configured():
            raise ValueError("Ollama API key tidak dikonfigurasi. Set OLLAMA_API_KEY di environment.")
        
        system_prompt = """Kamu adalah asisten yang bertugas mengekstrak informasi identitas NARASUMBER/PARTISIPAN yang DIWAWANCARAI dari transkrip wawancara.

TUGAS:
Ekstrak informasi berikut untuk SETIAP NARASUMBER yang diwawancarai (BUKAN pewawancara/interviewer):
1. Nama (nama lengkap atau nama panggilan narasumber)
2. Umur (dalam tahun)
3. Asal S1 (universitas atau institusi tempat narasumber menempuh S1)

ATURAN PENTING:
- HANYA ekstrak identitas NARASUMBER/PARTISIPAN yang sedang diwawancarai
- JANGAN masukkan identitas pewawancara/interviewer
- Jika ada beberapa narasumber (misalnya dalam FGD), ekstrak semua identitas mereka
- Jika informasi tidak disebutkan dalam transkrip, gunakan "-" sebagai nilai
- Jika transkrip tidak menyebutkan identitas narasumber apapun, kembalikan array kosong

FORMAT OUTPUT (JSON):
[
  {"nama": "...", "umur": "...", "asal_s1": "..."},
  {"nama": "...", "umur": "...", "asal_s1": "..."}
]

HANYA berikan output JSON, tanpa teks tambahan apapun."""

        user_prompt = f"""Ekstrak identitas partisipan dari transkrip berikut:

{transkrip}

Berikan output dalam format JSON array sesuai instruksi."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                stream=False
            )
            
            raw_response = response['message']['content'].strip()
            print(f"DEBUG - Raw identity response: {raw_response[:500]}...")
            
            # Parse JSON from response
            identities = self._parse_identities_json(raw_response)
            return identities
        
        except Exception as e:
            print(f"Error extracting identities: {e}")
            raise
    
    def _parse_identities_json(self, raw_response: str) -> list:
        """Parse JSON response untuk identitas"""
        import re
        
        # Try to find JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', raw_response)
        if json_match:
            try:
                identities = json.loads(json_match.group())
                # Validate structure
                validated = []
                for item in identities:
                    if isinstance(item, dict):
                        validated.append({
                            'nama': str(item.get('nama', '-')),
                            'umur': str(item.get('umur', '-')),
                            'asal_s1': str(item.get('asal_s1', '-'))
                        })
                return validated
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
        
        # If no valid JSON found, return empty list
        return []


# Singleton instance
_ollama_instance = None


def get_ollama_summarizer(api_key: Optional[str] = None) -> OllamaSummarizer:
    """Get or create the global Ollama summarizer instance"""
    global _ollama_instance
    if _ollama_instance is None:
        _ollama_instance = OllamaSummarizer(api_key=api_key)
    return _ollama_instance
