import pandas as pd
import logging
from datetime import datetime, timedelta
import sys
from pathlib import Path


# === CONFIG ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from utils.feature_calculator import calculate_all_features
from utils.bitget_futures import BitgetFutures


class OHLCVDownloader:
    def __init__(self, bitget_client):
        self.bitget_client = bitget_client
        self.symbol = 'ETH/USDT:USDT'
        self.timeframe = '1h'
        self.days_to_analyze = 450
        self.warmup_period = 4
        total_days_needed = self.days_to_analyze + self.warmup_period
        self.start_date = (datetime.now() - timedelta(days=total_days_needed)).strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.save_dir = base_dir / 'ohlcv' / f'ohlcv_from_{self.start_date}_to_{current_time}.csv'

    def fetch_data(self):
        try:
            total_days_needed = self.days_to_analyze + self.warmup_period
            logging.info(f"Fetching {self.timeframe} data from {self.start_date} for {self.symbol}")
            data = self.bitget_client.fetch_ohlcv(symbol=self.symbol, timeframe=self.timeframe, start_time=self.start_date)
            return data
        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")
            raise

    def save_data(self, data):
        data.to_csv(self.save_dir, index=False)

if __name__ == "__main__":
    bitget_client = BitgetFutures()
    ohlcv_downloader = OHLCVDownloader(bitget_client)
    data = ohlcv_downloader.fetch_data()
    ohlcv_downloader.save_data(data)


