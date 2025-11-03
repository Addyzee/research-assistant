import os
import re
from pathlib import Path
from datetime import datetime
import threading
import time
import sys


def check_env(var: str):
    if not os.environ.get(var):
        raise ValueError(f"No env variable for {var}")

def save_markdown(topic: str, content: str):
    safe_filename = re.sub(r'[^\w\s-]', '', topic)  # Remove invalid chars
    safe_filename = re.sub(r'[-\s]+', '_', safe_filename)  # Replace spaces/hyphens with underscore
    safe_filename = safe_filename.strip('_').lower()  # Clean up and lowercase
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_path = output_dir / f"{safe_filename}_{timestamp}.md"
    
    output_path.write_text(content, encoding='utf-8')
    print(f"\nReport saved to: {output_path.absolute()}")


class Spinner:
    def __init__(self, message="Processing"):
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.message = message
        self.running = False
        self.thread = None
    
    def spin(self):
        idx = 0
        while self.running:
            sys.stdout.write(f'\r{self.message} {self.spinner_chars[idx % len(self.spinner_chars)]}')
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * (len(self.message) + 5) + '\r')  # Clear the line
        sys.stdout.flush()
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()


