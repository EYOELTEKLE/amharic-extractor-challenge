import os
import asyncio
from telethon import TelegramClient, events
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
from datetime import datetime
import pandas as pd
from PIL import Image

from io import BytesIO
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import dotenv
dotenv.load_dotenv()
from src.utils.preprocessing import preprocess_amharic_text


# ========== CONFIGURATION ==========
API_ID = os.getenv("TELEGRAM_API_ID")  # Set in .env
API_HASH = os.getenv("TELEGRAM_API_HASH")  # Set in .env
SESSION_NAME = os.getenv("TELEGRAM_SESSION", "amharic_ecom")

# List at least 5 Ethiopian e-commerce Telegram channels (usernames or links)
CHANNELS = [
    "@ZemenExpress",
    "@nevacomputer",
    "@meneshayeofficial",
    "@ethio_brand_collection",
    "@Leyueqa",
]

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
MEDIA_DIR = os.path.join(DATA_DIR, 'media')
OUTPUT_FILE = os.path.join(DATA_DIR, 'telegram_messages.jsonl')
os.makedirs(MEDIA_DIR, exist_ok=True)

# ========== INGESTION & HANDLING ==========
def save_media(message, media, media_type):
    if not media:
        return None
    file_name = f"{message.id}_{media_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    file_path = os.path.join(MEDIA_DIR, file_name)
    if isinstance(media, MessageMediaPhoto):
        # Download photo
        message.download_media(file_path)
    elif isinstance(media, MessageMediaDocument):
        message.download_media(file_path)
    return file_path

def save_message(message, sender, channel):
    # Preprocess text
    clean_text = preprocess_amharic_text(message.message or "")
    # Save media if exists
    media_path = None
    if message.media:
        if isinstance(message.media, MessageMediaPhoto):
            media_path = save_media(message, message.media, 'photo')
        elif isinstance(message.media, MessageMediaDocument):
            media_path = save_media(message, message.media, 'doc')
    # Structure data
    data = {
        "id": message.id,
        "channel": channel,
        "sender": sender.username if sender else None,
        "timestamp": message.date.isoformat(),
        "text": clean_text,
        "media_path": media_path,
        "raw_text": message.message,
    }
    # Append to JSONL
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

async def main():
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start()
    print("[INFO] Telegram client started.")
    
    async def handler(event):
        sender = await event.get_sender()
        channel = event.chat.username if event.chat else None
        save_message(event.message, sender, channel)
        print(f"[NEW] {event.message.id} from {channel}")

    for channel in CHANNELS:
        client.add_event_handler(handler, events.NewMessage(chats=channel))
    print(f"[INFO] Listening to channels: {CHANNELS}")
    await client.run_until_disconnected()

if __name__ == "__main__":
    asyncio.run(main())
