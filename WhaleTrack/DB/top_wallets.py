import sqlite3
from tabulate import tabulate

DB = "whales.db"

def get_top_wallets(limit=20, direction="inflow", include_unknown=True):
    conn = sqlite3.connect(DB)
    c= conn.cursor()

    column = "total_inflow_eth" if direction == "inflow" else "total_outflow_eth"

    query = f"""
    SELECT address, tag, {column}
    FROM wallets
    WHERE {column} > 0
    """

    if not include_unknown:
        query += " AND tag != 'unknownWhale'"

    query += f" ORDER BY {column} DESC LIMIT ?"

    c.execute(query, (limit,))
    results = c.fetchall()
    conn.close()

    return results

if __name__ == "__main__":
    direction = input("choose direction [inflow/outflow] ").strip().lower()
    include_unknown = input("include unknown whales? [y/n]").strip().lower() == "y"
    limit = int(input("How many top wallets to show? : ").strip())

    data = get_top_wallets(limit=limit, direction=direction, include_unknown=include_unknown)
    print("\n Top Wallets by", direction.upper())
    print(tabulate(data, headers=["Address", "Tag", f"{direction.title()} (ETH)"], tablefmt="fancy_grid"))

