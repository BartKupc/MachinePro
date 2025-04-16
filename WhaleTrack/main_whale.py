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
from web3 import Web3
from decimal import Decimal
import sqlite3

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


ETH_THRESHOLD = 500 

with open('wallets.json', 'r') as f:
    wallets = json.load(f)
    
EXCHANGE_WALLETS = {k.lower(): v for k, v in wallets['eth_wallets'].items()}

ALERTED_TX_HASHES_FILE = "alerted_tx.txt"
LOG_FILE = "whale_alerts_log.csv"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "DB", "whales.db")

# --- FUNCTIONS ---

ALCHEMY_API = "https://eth-mainnet.g.alchemy.com/v2/-S2GIjK6gUSrTYKjHkIxdWU9u3QBAp_F"
web3 = Web3(Web3.HTTPProvider(ALCHEMY_API))

# --- FUNCTIONS ---


def get_eth_price_usd():
    try:

        
        # Fetch ETH/USDT ticker
        ticker = bitget_client.fetch_ticker('ETH/USDT:USDT')
        return float(ticker['last'])  # Return the last price
        
    except Exception as e:
        print("Failed to fetch ETH price:", e)
        return None
    

def to_eth(wei):
    return Decimal(wei) / Decimal(10**18)

LAST_BLOCK_FILE = "last_block.txt"

def load_last_block():
    if not os.path.exists(LAST_BLOCK_FILE):
        with open(LAST_BLOCK_FILE, 'w') as f:
            f.write(str(web3.eth.block_number))
        return web3.eth.block_number  # First run

    else:
        with open(LAST_BLOCK_FILE, 'r') as f:
            content = f.read().strip()
            if not content:  # If file is empty
                return web3.eth.block_number
            return int(content)  # Return the content we already read

def save_last_block(block_number):
    with open(LAST_BLOCK_FILE, 'w') as f:
        f.write(str(block_number))

def is_alerted(tx_hash):
    return tx_hash.hex() in load_alerted_hashes()

def scan_block_range(start_block, end_block, eth_price_usd):
    alerted_hashes = load_alerted_hashes()

    for block_number in range(start_block, end_block + 1):
        try:
            block = web3.eth.get_block(block_number, full_transactions=True)
            print(f" -> Block {block_number} with {len(block.transactions)} txs")

            for tx in block.transactions:
                tx_hash = tx.hash.hex()

                if tx_hash in alerted_hashes:
                    continue

                value_eth = to_eth(tx.value)
                if value_eth < ETH_THRESHOLD:
                    continue

                value_usd = float(value_eth) * eth_price_usd

                from_addr = tx['from'].lower()
                to_addr = tx['to'].lower()

                direction = "unknown"
                tag = "?"

                if from_addr in EXCHANGE_WALLETS:
                    direction = "outflow"
                    tag = EXCHANGE_WALLETS[from_addr]
                elif to_addr in EXCHANGE_WALLETS:
                    direction = "inflow"
                    tag = EXCHANGE_WALLETS[to_addr]

                message = f"""🐋 Whale Alert ({direction})
Tx: https://etherscan.io/tx/{tx_hash}
From: {tx['from']}
To: {tx['to']}
Amount: {value_eth:.2f} ETH (${value_usd:,.0f})
Tag: {tag}
Block: {block_number}"""

                send_telegram_alert(message)
                save_alerted_hash(tx_hash)
                if direction == "inflow":
                    update_wallet_balance(to_addr, value_eth, direction, tag)
                elif direction == "outflow":
                    update_wallet_balance(from_addr, value_eth, direction, tag)
                log_to_csv([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    tx_hash,
                    tx['from'],
                    tx['to'],
                    f"{value_eth:.2f}",
                    f"{value_usd:.0f}",
                    direction,
                    tag
                ])

        except Exception as e:
            logging.error(f"Error processing block {block_number}: {e}")

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


def update_wallet_balance(addr, value_eth, direction ,tag=""):
    if not tag or tag == "unknown":
        tag = "unknownWhale"
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO wallets(address, tag) VALUES (?, ?)", (addr, tag))

    # Convert Decimal to float for SQLite compatibility
    value_eth_float = float(value_eth)

    # Update balance stats
    if direction == "inflow":
        c.execute("UPDATE wallets SET last_balance_eth = last_balance_eth + ?, total_inflow_eth = total_inflow_eth + ? WHERE address = ?", (value_eth_float, value_eth_float, addr))
    elif direction == "outflow":
        c.execute("UPDATE wallets SET last_balance_eth = last_balance_eth - ?, total_outflow_eth = total_outflow_eth + ? WHERE address = ?", (value_eth_float, value_eth_float, addr))

    current_balance = float(get_wallet_balance(addr))  # Convert to float here as well
    c.execute("INSERT INTO snapshots(wallet_address, balance_eth, timestamp) VALUES(?,?,?)",(addr, current_balance, datetime.utcnow().isoformat()))

    conn.commit()
    conn.close()

def get_wallet_balance(address):
    try:
        # Convert address to checksum format
        checksum_address = web3.to_checksum_address(address)
        wei = web3.eth.get_balance(checksum_address)
        return to_eth(wei)
    except Exception as e:
        logging.error(f"Error getting wallet balance for {address}: {e}")
        return 0


if __name__ == "__main__":
    print("Running Alchemy Whale Tracker...")
    while True:

        try:
            eth_price= get_eth_price_usd()
            if not eth_price:
                continue
            last_block = load_last_block()
            latest_block = web3.eth.block_number
            print(f"Last block: {last_block}, Latest block: {latest_block}")  # Add this line
            
            if latest_block == last_block:
                scan_block_range(last_block, latest_block, eth_price)
                save_last_block(latest_block)
            elif latest_block > last_block:
                print(f"Scanning blocks {last_block + 1} to {latest_block}")
                scan_block_range(last_block +1, latest_block, eth_price)
                save_last_block(latest_block)
            else:
                print("no new blocks, yet")
        except Exception as e:
            logging.error(f"Error: {e}")
        time.sleep(20)

