import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path.cwd()
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=str(ENV_PATH))
else:
    load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

OUTPUT_DIR = BASE_DIR / "outputs"
SUM_DIR = OUTPUT_DIR / "summaries"
AUDIO_DIR = OUTPUT_DIR / "audio"
LINKS_DIR = OUTPUT_DIR / "links"
for d in (SUM_DIR, AUDIO_DIR, LINKS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

USERS_DB = DATA_DIR / "users.json"
SENT_DB = DATA_DIR / "sent_articles.json"

DELIVERY_TIME = os.getenv("DELIVERY_TIME", "07:00")
