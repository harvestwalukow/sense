"""
Batch Audio Processing Module
Based on SENSE App Jupyter Notebook
Processes multiple audio files and generates global summary
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

from .audio_processor import get_processor
from .evaluation import match_audio_with_gt, get_clean_text_from_docx, calculate_wer_cer
from .utils import secure_extract_zip, AUDIO_EXTS


def extract_and_discover_files(folder_path: Path, extract_dir: Path) -> Dict:
    """
    Extract folder dan temukan semua file audio + ground truth
    
    Args:
        folder_path: Path ke folder ZIP atau direktori
        extract_dir: Directory to extract to
    
    Returns:
        dict: {'audio_files': [...], 'gt_files': {...}}
    """
    print("\n" + "=" * 60)
    print("📁 SCANNING FOLDER")
    print("=" * 60)
    
    # Handle ZIP extraction
    if folder_path.suffix.lower() == '.zip':
        print(f"Extracting ZIP: {folder_path}")
        secure_extract_zip(folder_path, extract_dir)
        scan_path = extract_dir
    else:
        scan_path = folder_path
    
    audio_files = []
    gt_files = {}
    
    # Scan files recursively
    for root, dirs, files in os.walk(scan_path):
        for file in files:
            file_path = Path(root) / file
            file_lower = file.lower()
            
            # Skip hidden/system files
            if file.startswith('.') or '__MACOSX' in str(file_path):
                continue
            
            # Detect audio files
            if file_path.suffix.lower() in AUDIO_EXTS:
                audio_files.append(file_path)
            
            # Detect ground truth
            elif file_lower.endswith('.docx'):
                folder_name = Path(root).name.lower()
                # Check if in GT folder or has gt_ prefix
                if any(kw in folder_name for kw in ['ground', 'gt', 'transkrip', 'truth']) or file_lower.startswith('gt_'):
                    key = file.replace('.docx', '').replace('gt_', '').replace('GT_', '').strip()
                    gt_files[key] = file_path
    
    print(f"✓ Found {len(audio_files)} audio files")
    print(f"✓ Found {len(gt_files)} ground truth files")
    
    if audio_files:
        print("\n📌 Audio Files Preview:")
        for f in audio_files[:3]:
            print(f"   - {f.name}")
        if len(audio_files) > 3:
            print(f"   ... and {len(audio_files) - 3} more")
    
    return {
        'audio_files': audio_files,
        'gt_files': gt_files
    }


def batch_process_audio_files(
    audio_files: List[Path],
    gt_files: Dict[str, Path],
    progress_callback=None
) -> Dict:
    """
    Process multiple audio files in batch
    
    Args:
        audio_files: List of audio file paths
        gt_files: Dictionary of ground truth files
        progress_callback: Optional callback function for progress updates
    
    Returns:
        Dictionary with all analysis results
    """
    processor = get_processor()
    
    results = {
        'transcriptions': {},
        'emotions': {},
        'evaluations': [],
        'global_emotion': {},
        'topic_summaries': {},
        'files_processed': []
    }
    
    print("\n" + "=" * 60)
    print("🎤 BATCH TRANSCRIPTION & ANALYSIS")
    print("=" * 60)
    
    # Process each audio file
    for idx, audio_path in enumerate(audio_files, 1):
        filename = audio_path.name
        print(f"\n[{idx}/{len(audio_files)}] Processing: {filename}")
        
        if progress_callback:
            progress_callback(idx, len(audio_files), filename)
        
        try:
            # Analyze audio
            analysis = processor.analyze_audio_complete(str(audio_path))
            
            results['transcriptions'][filename] = {
                'full_text': analysis['transkripsi_mentah'],
                'segments': analysis['segments'],
                'word_count': analysis['word_count']
            }
            
            results['emotions'][filename] = {
                'dominant': analysis['analisis_emosi']['dominant_emotion'],
                'distribution': analysis['analisis_emosi']['emotion_distribution'],
                'percentages': analysis['analisis_emosi']['emotion_percentages'],
                'total_segments': analysis['analisis_emosi']['total_segments']
            }
            
            results['files_processed'].append(filename)
            
            # WER/CER evaluation if ground truth available
            gt_path = match_audio_with_gt(audio_path, gt_files)
            if gt_path:
                print(f"  📄 Evaluating with ground truth: {gt_path.name}")
                gt_text = get_clean_text_from_docx(gt_path)
                
                if gt_text and len(gt_text.strip()) > 10:
                    metrics = calculate_wer_cer(gt_text, analysis['transkripsi_mentah'])
                    results['evaluations'].append({
                        'filename': filename,
                        'ground_truth': gt_path.name,
                        **metrics
                    })
                    print(f"  WER: {metrics['wer']:.2%} | CER: {metrics['cer']:.2%}")
            
            print(f"  ✓ Success ({analysis['word_count']} words)")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results['transcriptions'][filename] = None
    
    # Generate global summaries
    print("\n" + "=" * 60)
    print("📝 GENERATING GLOBAL SUMMARIES")
    print("=" * 60)
    
    topic_texts = defaultdict(list)
    all_emotions = []
    
    # Collect all segments and emotions
    for filename, trans_data in results['transcriptions'].items():
        if trans_data is None:
            continue
        
        # Collect segments by topic
        for topic, text in trans_data['segments']:
            if topic in ['TANTANGAN', 'STRATEGI', 'HARAPAN', 'PREFERENSI DOSPEM']:
                topic_texts[topic].append(text)
        
        # Collect emotions
        emotion_data = results['emotions'].get(filename)
        if emotion_data and 'distribution' in emotion_data:
            for emotion, count in emotion_data['distribution'].items():
                all_emotions.extend([emotion] * count)
    
    # Generate topic summaries
    for topic, texts in topic_texts.items():
        if texts:
            combined_text = " ".join(texts)
            print(f"\n📌 {topic}: {len(combined_text.split())} words from {len(texts)} files")
            
            try:
                summary = processor.summarize_text(combined_text)
                results['topic_summaries'][topic] = summary
                print(f"  ✓ Summarized to {len(summary.split())} words")
            except Exception as e:
                print(f"  ⚠ Summarization error: {e}")
                results['topic_summaries'][topic] = combined_text[:300] + "..."
    
    # Aggregate emotions
    if all_emotions:
        emotion_counts = Counter(all_emotions)
        total = len(all_emotions)
        
        results['global_emotion'] = {
            'dominant_overall': emotion_counts.most_common(1)[0][0],
            'distribution': dict(emotion_counts),
            'percentages': {e: (c / total) * 100 for e, c in emotion_counts.items()},
            'total_segments': total
        }
        
        print("\n" + "=" * 60)
        print("😊 GLOBAL EMOTION AGGREGATION")
        print("=" * 60)
        print(f"\nTotal Emotion Segments: {total}")
        print(f"Dominant Emotion: {results['global_emotion']['dominant_overall'].upper()}")
    
    # Calculate evaluation averages
    if results['evaluations']:
        avg_wer = sum(e['wer'] for e in results['evaluations']) / len(results['evaluations'])
        avg_cer = sum(e['cer'] for e in results['evaluations']) / len(results['evaluations'])
        
        results['avg_wer'] = avg_wer
        results['avg_cer'] = avg_cer
        
        print("\n" + "=" * 60)
        print("📊 EVALUATION METRICS")
        print("=" * 60)
        print(f"Average WER: {avg_wer:.2%}")
        print(f"Average CER: {avg_cer:.2%}")
        print(f"Files Evaluated: {len(results['evaluations'])}")
    
    return results


