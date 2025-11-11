import time
from pathlib import Path
from typing import List, Dict
from config.settings import SUM_DIR, LINKS_DIR

def timestamped_name(prefix: str, topic: str, ext: str):
    t = time.strftime('%Y%m%d_%H%M%S')
    safe_topic = ''.join(c for c in topic if c.isalnum() or c in ('_', '-')).lower()
    return f"{prefix}_{safe_topic}_{t}.{ext}"

def save_script_and_links(summaries: List[Dict], script: str, topic: str):
    script_path = SUM_DIR / timestamped_name('summary', topic, 'txt')
    links_path = LINKS_DIR / timestamped_name('links', topic, 'txt')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script)
    with open(links_path, 'w', encoding='utf-8') as f:
        for s in summaries:
            f.write(f"- {s.get('title','')}\n  {s.get('url','')}\n\n")
    return script_path, links_path
