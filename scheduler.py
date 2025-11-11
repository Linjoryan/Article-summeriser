import os, time, json, threading
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from config.settings import DELIVERY_TIME, USERS_DB
from modules.telegram_bot import run_bot_in_thread
from main import main_for_user

def load_users():
    if not os.path.exists(USERS_DB):
        return {'users': []}
    with open(USERS_DB, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_for_all_users():
    data = load_users()
    for u in data.get('users', []):
        chat_id = u.get('chat_id')
        topics = u.get('topics', [])
        for topic in topics:
            try:
                main_for_user(chat_id, topic)
            except Exception as e:
                print('Error delivering to', chat_id, topic, e)

def start_scheduler_and_bot():
    # start bot in separate thread
    bot_thread = threading.Thread(target=run_bot_in_thread, daemon=True)
    bot_thread.start()
    print('Telegram bot started in background thread.')

    # schedule daily job
    hour, minute = map(int, DELIVERY_TIME.split(':'))
    sched = BlockingScheduler()
    trigger = CronTrigger(hour=hour, minute=minute)
    sched.add_job(run_for_all_users, trigger)
    print(f'Scheduler set for {DELIVERY_TIME} daily.')
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print('Shutting down scheduler...')

if __name__ == '__main__':
    start_scheduler_and_bot()
