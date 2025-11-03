import os
import re
from pathlib import Path
from datetime import datetime

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