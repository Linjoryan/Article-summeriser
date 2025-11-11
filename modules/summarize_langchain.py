from typing import List, Dict, Tuple
from newspaper import Article
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

def extract_full_text(url: str) -> str:
    try:
        art = Article(url)
        art.download()
        art.parse()
        return art.text or ''
    except Exception:
        return ''

def summarize_articles_langchain(articles: List[Dict], target_minutes: int = 10) -> Tuple[List[Dict], str]:
    """Summarize articles using LangChain map-reduce summarization. target_minutes controls
    the approximate target spoken length (used to tune chunking/LLM params).

    Returns list of summaries and the combined script."""
    # approximate target words and per-article target
    words_target = target_minutes * 140
    per_article_words = max(80, words_target // max(1, len(articles)))

    llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.2, max_tokens=1500)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chain = load_summarize_chain(llm, chain_type='map_reduce')

    summaries = []
    for art in articles:
        title = art.get('title') or art.get('description') or 'Untitled'
        url = art.get('url')
        text = extract_full_text(url)
        if not text:
            summaries.append({'title': title, 'url': url, 'summary': '[Could not extract article text]'})
            continue
        docs = text_splitter.create_documents([text])
        # run chain to produce a summary
        try:
            summary = chain.run(docs)
        except Exception as e:
            summary = '[Summarization failed]'
        summaries.append({'title': title, 'url': url, 'summary': summary})

    # build script
    intro = 'Good morning — here are the top stories for today.'
    segments = [f"Story {i}: {s['title']}. {s['summary']}" for i, s in enumerate(summaries, start=1)]
    outro = "That's all for today's brief. To read the full articles, see the links provided. Have a great day."
    script = '\n\n'.join([intro] + segments + [outro])
    return summaries, script
