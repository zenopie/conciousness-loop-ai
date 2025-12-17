"""
Executors - how the loop interfaces with reality.
Each executor takes an action string and returns an outcome.
"""

import subprocess
import requests
import re
from pathlib import Path
from typing import Optional


def extract_text_from_html(html: str, max_chars: int = 1000) -> str:
    """Extract readable text from HTML, stripped and truncated."""
    # Remove script and style elements
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common HTML entities
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Truncate
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


class Executor:
    """Base executor - override execute()"""
    
    def execute(self, action: str) -> str:
        raise NotImplementedError


class ShellExecutor(Executor):
    """Execute shell commands."""
    
    def __init__(self, allowed_commands: Optional[list] = None):
        # Safety: whitelist of allowed command prefixes
        self.allowed = allowed_commands or ["ls", "cat", "echo", "pwd", "find"]
    
    def execute(self, action: str) -> str:
        # Extract command from action
        # Naive: assume action is the command
        cmd = action.strip()
        
        # Safety check
        if not any(cmd.startswith(a) for a in self.allowed):
            return f"Blocked: {cmd} not in allowed commands"
        
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, 
                text=True, timeout=10
            )
            return result.stdout or result.stderr or "No output"
        except subprocess.TimeoutExpired:
            return "Command timed out"
        except Exception as e:
            return f"Error: {e}"


class FileExecutor(Executor):
    """Read and write files."""
    
    def __init__(self, base_path: str = "/home/claude/loop/workspace"):
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)
    
    def execute(self, action: str) -> str:
        # Parse action: READ path or WRITE path content
        parts = action.strip().split(" ", 2)
        
        if len(parts) < 2:
            return "Invalid file action"
        
        op = parts[0].upper()
        path = self.base / parts[1]
        
        # Safety: stay in workspace
        if not str(path.resolve()).startswith(str(self.base.resolve())):
            return "Path outside workspace"
        
        if op == "READ":
            if path.exists():
                return path.read_text()[:2000]  # Truncate
            return "File not found"
        
        elif op == "WRITE" and len(parts) == 3:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(parts[2])
            return f"Wrote {len(parts[2])} chars to {path.name}"
        
        return f"Unknown operation: {op}"


class WebExecutor(Executor):
    """Fetch URLs and extract readable text."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def execute(self, action: str) -> str:
        url = action.strip()

        if not url.startswith("http"):
            return "Invalid URL"

        try:
            resp = requests.get(url, timeout=self.timeout)
            # Extract readable text from HTML
            text = extract_text_from_html(resp.text, max_chars=1000)
            return f"Fetched {url}: {text}"
        except Exception as e:
            return f"Fetch error: {e}"


class ThinkExecutor(Executor):
    """No external action - just thinking."""
    
    def execute(self, action: str) -> str:
        return f"Thought: {action}"


class CompositeExecutor(Executor):
    """Route to appropriate executor based on action prefix."""
    
    def __init__(self):
        self.executors = {
            "SHELL": ShellExecutor(),
            "FILE": FileExecutor(),
            "WEB": WebExecutor(),
            "THINK": ThinkExecutor()
        }
    
    def execute(self, action: str) -> str:
        parts = action.strip().split(" ", 1)
        prefix = parts[0].upper()
        
        if prefix in self.executors:
            remainder = parts[1] if len(parts) > 1 else ""
            return self.executors[prefix].execute(remainder)
        
        # Default to thinking
        return self.executors["THINK"].execute(action)
