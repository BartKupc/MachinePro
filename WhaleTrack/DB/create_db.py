import sqlite3

def init_db():
    conn = sqlite3.connect("whales.db")
    c = conn.cursor()

    #tables

    c.execute("""
    CREATE TABLE IF NOT EXISTS wallets (
              address TEXT PRIMARY KEY,
              tag TEXT,
              last_balance_eth REAL DEFAULT 0,
              total_inflow_eth REAL DEFAULT 0,
              total_outflow_eth REAL DEFAULT 0
    )     
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS snapshots(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              wallet_address TEXT,
              balance_eth REAL,
              timestamp TEXT,
              FOREIGN KEY(wallet_address) REFERENCES wallets(address)
    )
    """)
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()