"""
LexRank Summarizer untuk Generate Summary
Menggunakan LexRank (sumy) untuk meringkas transkrip wawancara
Menggantikan Ollama API
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


class LexRankTextSummarizer:
    """Generate summary menggunakan LexRank Algorithm"""
    
    def __init__(self):
        """Initialize LexRank summarizer"""
        self.summarizer = LexRankSummarizer()
    
    def _summarize_text(self, text: str, num_sentences: Optional[int] = None) -> str:
        """
        Summarize a single text using LexRank
        
        Args:
            text: Text to summarize
            num_sentences: Number of sentences in summary (auto-calculated if None)
        
        Returns:
            Summary string
        """
        if not text or len(text.split()) < 20:
            return text[:300] + "..." if text and len(text) > 300 else (text or "")
        
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            
            if num_sentences is None:
                # Auto-calculate based on text length
                word_count = len(text.split())
                num_sentences = max(2, min(8, word_count // 50))
            
            summary_sentences = self.summarizer(parser.document, num_sentences)
            summary = " ".join([str(s) for s in summary_sentences])
            
            return summary if summary else text[:300] + "..."
        except Exception as e:
            print(f"LexRank summarization error: {e}")
            return text[:300] + "..."
    
    def _extract_topic_content(self, transcript: str, topic_keywords: List[str]) -> str:
        """
        Extract content related to a specific topic from transcript
        
        Args:
            transcript: Full transcript text
            topic_keywords: Keywords to identify the topic
        
        Returns:
            Extracted content for the topic
        """
        # Try to find section by keywords
        sentences = re.split(r'[.!?]+', transcript)
        relevant_sentences = []
        
        # Window approach: get sentences around keyword matches
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            for keyword in topic_keywords:
                if keyword.lower() in sentence_lower:
                    # Get context: previous sentence, current, and next 2
                    start = max(0, i - 1)
                    end = min(len(sentences), i + 3)
                    relevant_sentences.extend(sentences[start:end])
                    break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sentences = []
        for s in relevant_sentences:
            s_clean = s.strip()
            if s_clean and s_clean not in seen:
                seen.add(s_clean)
                unique_sentences.append(s_clean)
        
        return ". ".join(unique_sentences)
    
    def generate_summary_4_points(self, transkrip: str, stream: bool = False) -> Dict[str, any]:
        """
        Generate summary dengan 4 poin dari transkrip wawancara menggunakan LexRank
        
        Args:
            transkrip: Teks transkrip wawancara
            stream: Ignored (for API compatibility)
        
        Returns:
            Dict dengan keys: 'full_summary', 'tantangan', 'solusi', 'harapan', 'preferensi'
        """
        # Define keywords for each topic
        topics = {
            'tantangan': ['tantangan', 'hambatan', 'kesulitan', 'kendala', 'masalah', 'stuck', 'susah', 'sulit'],
            'solusi': ['strategi', 'solusi', 'tips', 'siasat', 'mengatasi', 'atasi', 'coping', 'cara'],
            'harapan': ['harapan', 'ke depan', 'masa depan', 'cita-cita', 'impian', 'berharap', 'ingin'],
            'preferensi': ['preferensi', 'dospem', 'kriteria', 'pembimbing', 'dosen', 'memilih', 'referensi']
        }
        
        result = {
            'full_summary': '',
            'tantangan': '',
            'solusi': '',
            'harapan': '',
            'preferensi': ''
        }
        
        full_summary_parts = []
        
        for topic_key, keywords in topics.items():
            # Extract relevant content
            topic_content = self._extract_topic_content(transkrip, keywords)
            
            if topic_content and len(topic_content.split()) > 10:
                # Summarize the extracted content
                summary = self._summarize_text(topic_content, num_sentences=3)
                result[topic_key] = summary
                
                # Build full summary
                topic_titles = {
                    'tantangan': '## 1. Tantangan selama S2',
                    'solusi': '## 2. Solusi / Coping Strategy',
                    'harapan': '## 3. Harapan setelah lulus',
                    'preferensi': '## 4. Preferensi Pembimbing Tesis'
                }
                full_summary_parts.append(f"{topic_titles[topic_key]}\n{summary}")
            else:
                # Fallback: use general summary from full transcript
                fallback = self._summarize_text(transkrip, num_sentences=2)
                result[topic_key] = f"(Tidak ditemukan informasi spesifik tentang {topic_key}. Ringkasan umum: {fallback[:200]}...)"
        
        result['full_summary'] = "\n\n".join(full_summary_parts) if full_summary_parts else self._summarize_text(transkrip)
        
        return result
    
    def generate_batch_summary(self, transcripts: List[str]) -> Dict[str, any]:
        """
        Generate summary untuk batch transkrip (combine multiple transcripts)
        
        Args:
            transcripts: List of transcript texts
        
        Returns:
            Dict dengan combined summary
        """
        # Combine all transcripts
        combined_transcript = "\n\n".join(transcripts)
        
        # Generate summary using the same 4-point approach
        result = self.generate_summary_4_points(combined_transcript)
        result['total_transcripts'] = len(transcripts)
        
        # Add note about aggregation
        if result['full_summary']:
            result['full_summary'] = f"**Ringkasan Gabungan dari {len(transcripts)} Transkrip**\n\n" + result['full_summary']
        
        return result
    
    def summarize_segments(self, segments_list: List[Tuple[str, str]]) -> str:
        """
        Generate chronological summary from segments (same as notebook)
        
        Args:
            segments_list: List of (topic, content) tuples
        
        Returns:
            Full summary text organized by topic
        """
        full_summary_output = ""
        
        for topic, text in segments_list:
            if len(text.split()) < 10:
                continue
            
            try:
                summary = self._summarize_text(text)
                full_summary_output += f"\n[{topic}]\n{summary}\n"
            except Exception:
                full_summary_output += f"\n[{topic}]\n{text[:200]}...\n"
        
        return full_summary_output


# Singleton instance
_summarizer_instance = None

def get_lexrank_summarizer() -> LexRankTextSummarizer:
    """Get or create the global LexRank summarizer instance"""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = LexRankTextSummarizer()
    return _summarizer_instance






