import math
from typing import List, Dict, Tuple
from tqdm import tqdm
from newspaper import Article
from config.settings import OPENAI_API_KEY

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

if OPENAI_AVAILABLE and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

def extract_full_text(url: str) -> str:
    try:
        art = Article(url)
        art.download()
        art.parse()
        return art.text or ''
    except Exception:
        return ''

def prepare_article(article: Dict) -> (str, str):
    title = article.get('title') or article.get('description') or '(untitled)'
    url = article.get('url')
    content = article.get('content') or ''
    full = ''
    if url:
        full = extract_full_text(url)
    if full and len(full) > 200:
        return title, full
    fallback = '\n\n'.join(filter(None, [article.get('description') or '', content]))
    return title, fallback or full or ''

def summarize_with_openai(text: str, max_tokens: int = 300) -> str:
    if not OPENAI_AVAILABLE:
        raise RuntimeError('OpenAI SDK not installed')
    prompt = (
        'You are an expert summarizer. Produce a concise, spoken-style summary of the article below. '
        'Keep it factual and suitable for a ~2-4 minute spoken segment.\n\n' + text
    )
    resp = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return resp['choices'][0]['message']['content'].strip()

def build_hf_summarizer():
    if not HF_AVAILABLE:
        raise RuntimeError('transformers not installed')
    return pipeline('summarization', model='facebook/bart-large-cnn')

def summarize_articles(articles: List[Dict], target_minutes: int = 15, prefer_openai: bool = True) -> Tuple[List[Dict], str]:
    total_words_target = target_minutes * 140
    per_article_words = max(80, total_words_target // max(1, len(articles)))
    summaries = []
    hf_summarizer = None
    for art in tqdm(articles, desc='Summarizing'):
        title, text = prepare_article(art)
        if not text:
            summary = '[Could not extract article text]'
        else:
            if prefer_openai and OPENAI_AVAILABLE and OPENAI_API_KEY:
                try:
                    summary = summarize_with_openai(text, max_tokens=math.ceil(per_article_words * 1.6))
                except Exception:
                    hf_summarizer = hf_summarizer or (build_hf_summarizer() if HF_AVAILABLE else None)
                    if hf_summarizer:
                        summary = hf_summarizer(text, max_length=min(300, per_article_words))[0]['summary_text']
                    else:
                        summary = '[Summarization failed]'
            else:
                hf_summarizer = hf_summarizer or (build_hf_summarizer() if HF_AVAILABLE else None)
                if hf_summarizer:
                    summary = hf_summarizer(text, max_length=min(300, per_article_words))[0]['summary_text']
                else:
                    summary = '[Summarization unavailable]'
        summaries.append({'title': title, 'url': art.get('url'), 'summary': summary})
    intro = 'Good morning — here are the top stories for today.'
    segments = [f"Story {i}: {s['title']}. {s['summary']}" for i, s in enumerate(summaries, start=1)]
    outro = "That's all for today's brief. To read the full articles, see the links provided. Have a great day."
    script = '\n\n'.join([intro] + segments + [outro])
    return summaries, script
