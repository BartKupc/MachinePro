import requests
import time
import csv
import os
from datetime import datetime
from decimal import Decimal
import json

import logging
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add this to properly import from parent directory
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))
print(base_dir)

from utils.bitget_futures import BitgetFutures
bitget_client = BitgetFutures()

with open('config.json', 'r') as f:
    config = json.load(f)


ETHERSCAN_API_KEY = config['ethscan']['apiKey']
TELEGRAM_BOT_TOKEN = config['telegram']['botToken']
TELEGRAM_CHAT_ID = config['telegram']['chatId']


ETH_THRESHOLD = 500 # Alert if value > X ETH
USD_THRESHOLD = 1000000 # OR if value > $1M

with open('wallets.json', 'r') as f:
    wallets = json.load(f)
    
EXCHANGE_WALLETS = wallets['eth_wallets']  # For ETH tracking

ALERTED_TX_HASHES_FILE = "alerted_tx.txt"
LOG_FILE = "whale_alerts_log.csv"



# --- FUNCTIONS ---


def get_eth_price_usd():
    try:

        
        # Fetch ETH/USDT ticker
        ticker = bitget_client.fetch_ticker('ETH/USDT:USDT')
        return float(ticker['last'])  # Return the last price
        
    except Exception as e:
        print("Failed to fetch ETH price:", e)
        return None



def get_latest_transactions(address, last_block=None):
    try:
        # If we have a last block, get all transactions since then
        if last_block:
            url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock={last_block}&sort=desc&apikey={ETHERSCAN_API_KEY}"
        else:
            # First run - get last 1000 transactions
            url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&sort=desc&apikey={ETHERSCAN_API_KEY}"
        
        r = requests.get(url)
        response = r.json()
        
        # Check for API rate limit error
        if response.get('status') == '0' and 'rate limit' in response.get('message', '').lower():
            print(f"Rate limit hit for {address}. Waiting before retry...")
            time.sleep(1)  # Wait 1 second before retry
            return get_latest_transactions(address, last_block)  # Retry the request
            
        return response.get("result", [])
    except Exception as e:
        print(f"Error fetching transactions for {address}: {e}")
        return []

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=data)

def load_alerted_hashes():
    if not os.path.exists(ALERTED_TX_HASHES_FILE):
        return set()
    with open(ALERTED_TX_HASHES_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())

def save_alerted_hash(tx_hash):
    with open(ALERTED_TX_HASHES_FILE, "a") as f:
        f.write(tx_hash + "\n")

def log_to_csv(data):
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "tx_hash", "from", "to", "eth", "usd", "direction", "tag"])
        writer.writerow(data)

def check_transactions():
    eth_price = get_eth_price_usd()
    if not eth_price:
        return

    alerted_hashes = load_alerted_hashes()
    last_blocks = load_last_blocks()
    
    # Process wallets in batches of 5 (Etherscan's rate limit)
    wallet_items = list(EXCHANGE_WALLETS.items())
    for i in range(0, len(wallet_items), 5):
        batch = wallet_items[i:i+5]
        
        for wallet, tag in batch:
            last_block = last_blocks.get(wallet, None)
            txs = get_latest_transactions(wallet, last_block)
            
            if not txs:
                continue
                
            # Update last block for this wallet
            last_blocks[wallet] = txs[0]['blockNumber']
            
            for tx in txs:
                if tx['hash'] in alerted_hashes:
                    continue
                    
                value_eth = Decimal(tx['value']) / Decimal(10**18)
                value_usd = float(value_eth) * eth_price

                if value_eth < ETH_THRESHOLD and value_usd < USD_THRESHOLD:
                    continue

                # Tag direction
                from_addr = tx['from'].lower()
                to_addr = tx['to'].lower()

                if from_addr in EXCHANGE_WALLETS:
                    direction = "Outflow"
                    tag = EXCHANGE_WALLETS[from_addr]
                elif to_addr in EXCHANGE_WALLETS:
                    direction = "Inflow"
                    tag = EXCHANGE_WALLETS[to_addr]
                else:
                    direction = "Unknown"
                    tag = "?"

                message = f"""[Whale Alert - {direction}]
Tx: https://etherscan.io/tx/{tx['hash']}
From: {tx['from']}
To: {tx['to']}
Amount: {value_eth:.2f} ETH (${value_usd:,.0f})
Exchange: {tag}
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"""

                send_telegram_alert(message)
                save_alerted_hash(tx['hash'])
                log_to_csv([
                    datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    tx['hash'],
                    tx['from'],
                    tx['to'],
                    f"{value_eth:.2f}",
                    f"{value_usd:.0f}",
                    direction,
                    tag
                ])
        
        # If this isn't the last batch, wait 1 second before processing next batch
        if i + 5 < len(wallet_items):
            time.sleep(1)
    
    # Save updated last blocks
    save_last_blocks(last_blocks)

def load_last_blocks():
    """Load last processed block numbers for each wallet"""
    if not os.path.exists('last_blocks.json'):
        return {}
    with open('last_blocks.json', 'r') as f:
        return json.load(f)

def save_last_blocks(last_blocks):
    """Save last processed block numbers for each wallet"""
    with open('last_blocks.json', 'w') as f:
        json.dump(last_blocks, f)

# --- RUN LOOP ---
if __name__ == "__main__":
    print("Running whale alert bot...")
    while True:
        try:
            check_transactions()
        except Exception as e:
            print("Error:", e)
        time.sleep(60) # Poll every 1 minute