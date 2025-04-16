import sqlite3
from datetime import datetime, timedelta
import argparse
from tabulate import tabulate


DB_PATH = "whales.db"

def get_wallet_summary(address, days=7):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    address = address.lower()

    c.execute("SELECT tag, last_balance_eth, total_inflow_eth, total_outflow_eth FROM wallets WHERE address = ?", (address,))
    wallet = c.fetchone()
    if not wallet:
        print("Wallet not found in database.")
        return

    tag, last_balance, inflow, outflow = wallet
    print(f"\nWallet Summary for {address}")
    print(f"Tag: {tag}")
    print(f"Current Balance: {last_balance:.2f} ETH")
    print(f"Total Inflow: {inflow:.2f} ETH")
    print(f"Total Outflow: {outflow:.2f} ETH")

    #Fetch recent snapshots
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    c.execute("""
              SELECT timestamp, balance_eth
              FROM snapshots
              WHERE wallet_address = ?
              AND timestamp >= ?
              ORDER BY timestamp ASC
              """, (address, since))
    
    snapshots = c.fetchall()

    conn.close()

    if snapshots:
        print(f"\nSnapshots in the last {days} day(s):")
        print(tabulate(snapshots, headers=["Timestamp", "Balance (ETH)"], tablefmt="fancy_grid"))
    else:
        print(f"\nNo snapshots data found for the last {days} day(s)")

def list_wallets():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT address, tag FROM wallets ORDER BY tag ASC")
    wallets = c.fetchall()
    conn.close()
    return wallets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View whale wallet summary")
    parser.add_argument("--address", help="Wallet address to view")
    parser.add_argument("--days", type=int, default=7, help="Number of days to look back (default: 7)")
    args = parser.parse_args()

    if args.address:
        get_wallet_summary(args.address, args.days)
    else:
        print("\n📚 No wallet address provided. Here are wallets in your database:\n")
        wallets = list_wallets()
        print(tabulate(wallets, headers=["Address", "Tag"], tablefmt="github"))
        
        choice = input("\n🔍 Enter wallet address to view: ").strip()
        days = input("🕒 Enter number of days to look back (default = 7): ").strip()
        try:
            days = int(days) if days else 7
        except:
            days = 7

        get_wallet_summary(choice, days)          

    
    
    
