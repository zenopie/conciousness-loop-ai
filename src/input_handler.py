"""
Input handling for consciousness loop.
Multiple ways to inject messages into the running loop.
"""

import threading
import queue
import sys
import os
from pathlib import Path


class InputHandler:
    """Base class for input injection."""
    
    def get_pending(self) -> str:
        """Return pending input, or empty string."""
        raise NotImplementedError


class FileInputHandler(InputHandler):
    """Watch a file for input. Write to file, loop picks it up."""
    
    def __init__(self, path: str = "input.txt"):
        self.path = Path(path)
        # Create empty file
        self.path.touch()
    
    def get_pending(self) -> str:
        """Read and clear the input file."""
        if not self.path.exists():
            return ""
        
        content = self.path.read_text().strip()
        if content:
            self.path.write_text("")  # Clear after reading
        return content


class StdinInputHandler(InputHandler):
    """Non-blocking stdin reader using a thread."""
    
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        
        # Start reader thread
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
    
    def _reader(self):
        """Background thread that reads stdin."""
        print("[Input thread started - type messages and press Enter]")
        while self.running:
            try:
                line = input()
                if line.strip():
                    self.queue.put(line.strip())
            except EOFError:
                break
    
    def get_pending(self) -> str:
        """Get all queued input, concatenated."""
        messages = []
        while not self.queue.empty():
            try:
                messages.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return "\n".join(messages)
    
    def stop(self):
        self.running = False


class CompositeInputHandler(InputHandler):
    """Check multiple sources."""
    
    def __init__(self, handlers: list):
        self.handlers = handlers
    
    def get_pending(self) -> str:
        """Collect from all handlers."""
        messages = []
        for handler in self.handlers:
            msg = handler.get_pending()
            if msg:
                messages.append(msg)
        return "\n".join(messages)
