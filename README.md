# Week 4 Challenge

EthioMart has a vision to become the primary hub for all Telegram-based e-commerce activities in Ethiopia. With the increasing popularity of Telegram for business transactions, various independent e-commerce channels have emerged, each facilitating its own operations. However, this decentralization presents challenges for both vendors and customers who need to manage multiple channels for product discovery, order placement, and communication.

To solve this problem, EthioMart plans to create a single centralized platform that consolidates real-time data from multiple e-commerce Telegram channels into one unified channel. By doing this, they aim to provide a seamless experience for customers to explore and interact with multiple vendors in one place.

This project focuses on fine-tuning LLM’s for Amharic Named Entity Recognition (NER) system that extracts key business entities such as product names, prices, and Locations, from text, images, and documents shared across these Telegram channels. The extracted data will be used to populate EthioMart's centralised database, making it a comprehensive e-commerce hub.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Setup Telegram API Credentials

1. Go to [my.telegram.org](https://my.telegram.org) and sign in.
2. Create a new application to get your `API_ID` and `API_HASH`.
3. Add the following to your `.env` file in the project root:
   ```env
   TELEGRAM_API_ID=your_api_id
   TELEGRAM_API_HASH=your_api_hash
   TELEGRAM_SESSION=amharic_ecom
   ```

### 2. Select Channels

Edit the `CHANNELS` list in `src/services/telegram_ingest.py` to include at least 5 Ethiopian e-commerce Telegram channels (by username or link).

### 3. Run Data Ingestion

```bash
python src/services/telegram_ingest.py
```
This will start listening to the specified channels and save messages (text, images, documents) in real time.

### 4. Data Format
- Preprocessed data is saved in `data/telegram_messages.jsonl`.
- Media files (images, docs) are saved in `data/media/`.
- Each JSONL line includes: `id`, `channel`, `sender`, `timestamp`, `text`, `media_path`, `raw_text`.

### 5. Amharic Preprocessing
- Text is tokenized, normalized, and cleaned for Amharic-specific features using `src/utils/preprocessing.py`.
