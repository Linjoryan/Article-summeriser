import requests
from pathlib import Path
from config.settings import TELEGRAM_BOT_TOKEN

BASE = 'https://api.telegram.org/bot{token}/{method}'

def send_telegram_message(chat_id, text):
    url = BASE.format(token=TELEGRAM_BOT_TOKEN, method='sendMessage')
    data = {'chat_id': chat_id, 'text': text}
    resp = requests.post(url, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()

def send_telegram_audio(chat_id, audio_path: Path):
    url = BASE.format(token=TELEGRAM_BOT_TOKEN, method='sendAudio')
    with open(audio_path, 'rb') as f:
        files = {'audio': f}
        data = {'chat_id': chat_id}
        resp = requests.post(url, data=data, files=files, timeout=60)
    resp.raise_for_status()
    return resp.json()

def send_daily_brief(chat_id, audio_path: Path, links_path: Path):
    send_telegram_audio(chat_id, audio_path)
    with open(links_path, 'r', encoding='utf-8') as f:
        links = f.read()
    send_telegram_message(chat_id, f"📰 Here are your article links for today:\n\n{links}")
