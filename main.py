from modules.fetch_news import fetch_articles
from modules.summarize import summarize_articles
from modules.text_to_speech import articles_to_audio
from modules.utils import save_script_and_links
from modules.send_telegram import send_daily_brief
from pathlib import Path

def main_for_user(chat_id: int, topic: str, prefer_openai: bool = True):
    print(f"Running brief for {chat_id} - topic: {topic}")
    articles = fetch_articles(topic, count=5)
    if not articles:
        print('No fresh articles for', topic)
        return
    summaries, script = summarize_articles(articles, target_minutes=15, prefer_openai=prefer_openai)
    script_path, links_path = save_script_and_links(summaries, script, topic)
    audio_path = articles_to_audio(script)
    send_daily_brief(chat_id, audio_path, links_path)
    print('Delivered to', chat_id)
