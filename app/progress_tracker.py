"""
Progress Tracker for Real-time UI Updates
Stores processing logs and current step for frontend polling
"""
import json
import threading
from pathlib import Path
from typing import List, Optional

class ProgressTracker:
    """Thread-safe progress tracker"""
    
    def __init__(self):
        self._logs: List[str] = []
        self._current_step: str = ""
        self._complete: bool = False
        self._lock = threading.Lock()
        
    def log(self, message: str):
        """Add a log message"""
        with self._lock:
            self._logs.append(message)
            # Print to console as well (only if not already printed)
            # print(message)  # Disabled to avoid double printing
    
    def set_step(self, step: str):
        """Set current processing step"""
        with self._lock:
            self._current_step = step
            # Add to logs without double printing
            self._logs.append(f"\n{'='*60}")
            self._logs.append(step)
            self._logs.append(f"{'='*60}")
            # Print to console
            print(f"\n{'='*60}")
            print(step)
            print(f"{'='*60}")
    
    def set_complete(self):
        """Mark processing as complete"""
        with self._lock:
            self._complete = True
            self.log("\nSelesai! Proses analisis berhasil diselesaikan.")
    
    def get_progress(self) -> dict:
        """Get current progress state"""
        with self._lock:
            return {
                "logs": self._logs.copy(),
                "current_step": self._current_step,
                "complete": self._complete
            }
    
    def clear(self):
        """Clear all logs"""
        with self._lock:
            self._logs = []
            self._current_step = ""
            self._complete = False


# Global tracker instance
_global_tracker: Optional[ProgressTracker] = None

def get_tracker() -> ProgressTracker:
    """Get or create global progress tracker"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ProgressTracker()
    return _global_tracker

def reset_tracker():
    """Reset the global tracker"""
    global _global_tracker
    _global_tracker = ProgressTracker()

