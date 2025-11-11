import os, json, requests
from datetime import datetime, timedelta
from typing import List, Dict
from config.settings import NEWSAPI_KEY, SENT_DB

NEWSAPI_EVERYTHING = "https://newsapi.org/v2/everything"

def _load_sent_urls():
    if not SENT_DB.exists():
        return set()
    try:
        with open(SENT_DB, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get('urls', []))
    except Exception:
        return set()

def _save_sent_urls(urls_set):
    try:
        with open(SENT_DB, 'w', encoding='utf-8') as f:
            json.dump({"urls": list(urls_set)}, f, indent=2)
    except Exception as e:
        print('Failed saving sent DB:', e)

def fetch_articles(topic: str = 'technology', count: int = 5, max_age_days: int = 60) -> List[Dict]:
    """Fetch recent unique articles about `topic`. Skip previously sent and older than max_age_days."""
    if not NEWSAPI_KEY:
        raise RuntimeError('NEWSAPI_KEY not configured in .env')

    params = {
        'q': topic,
        'language': 'en',
        'pageSize': 30,
        'sortBy': 'publishedAt',
        'apiKey': NEWSAPI_KEY
    }
    resp = requests.get(NEWSAPI_EVERYTHING, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    articles = data.get('articles', [])

    sent_urls = _load_sent_urls()
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    fresh = []
    for a in articles:
        url = a.get('url')
        if not url or url in sent_urls:
            continue
        published = a.get('publishedAt')
        if not published:
            continue
        try:
            # publishedAt: '2025-11-07T12:34:56Z'
            dt = datetime.fromisoformat(published.replace('Z', '+00:00'))
        except Exception:
            continue
        if dt < cutoff:
            continue
        fresh.append(a)
        if len(fresh) >= count:
            break

    # update sent DB with freshly selected urls
    if fresh:
        new_urls = sent_urls.union({a.get('url') for a in fresh if a.get('url')})
        _save_sent_urls(new_urls)

    return fresh
