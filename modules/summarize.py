import os, json
from datetime import datetime
from pathlib import Path
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from config.settings import TELEGRAM_BOT_TOKEN, USERS_DB

USERS_DB = Path('data') / 'users.json'

def _load_users():
    if not USERS_DB.exists():
        return {'users': []}
    with open(USERS_DB, 'r', encoding='utf-8') as f:
        return json.load(f)

def _save_users(data):
    with open(USERS_DB, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to Daily Brief AI!\n\nPlease reply with your preferred topics (comma-separated, e.g. 'AI, health, business')."
    )
    context.user_data['awaiting_topics'] = True

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get('awaiting_topics'):
        topics = [t.strip() for t in update.message.text.split(',') if t.strip()]
        user_id = update.message.chat_id
        data = _load_users()
        # check existing
        for u in data['users']:
            if u.get('chat_id') == user_id:
                u['topics'] = topics
                u['registered_on'] = datetime.utcnow().isoformat()
                _save_users(data)
                await update.message.reply_text(f"✅ Preferences updated! You will receive briefs for: {', '.join(topics)}")
                context.user_data['awaiting_topics'] = False
                return
        # new user
        data['users'].append({'chat_id': user_id, 'topics': topics, 'registered_on': datetime.utcnow().isoformat()})
        _save_users(data)
        await update.message.reply_text(f"✅ Registered! You'll get daily briefs about: {', '.join(topics)}")
        context.user_data['awaiting_topics'] = False
    else:
        await update.message.reply_text("Type /start to register or update your preferences.")

def run_bot_in_thread():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    # run polling (blocking) - caller should run in a thread if they want the scheduler in same process
    app.run_polling()
