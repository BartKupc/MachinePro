import numpy as np
import pandas as pd
import ta
import logging

def calculate_all_features(data, warmup_candles=110):
    """
    Calculate all technical indicators and features with preprocessing.
    
    Args:
        data: OHLCV data
        warmup_candles: Number of candles to skip for warmup (default 100 candles based on longest indicators:
                       SMA100 and ATR MA100 both need 100 candles)
        
    Returns:
        Tuple of (processed features DataFrame, processed price DataFrame)
    """
    # Convert input data to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    else:
        df = data.copy()
    
    features_df = df.copy()
    
    # === Price-based indicators ===
    # MACD
    features_df['macd'] = ta.trend.macd(features_df['close'])
    features_df['macd_signal'] = ta.trend.macd_signal(features_df['close'])
    features_df['macd_diff'] = features_df['macd'] - features_df['macd_signal']
    
    # Moving Averages
    features_df['ema9'] = ta.trend.ema_indicator(features_df['close'], window=9)
    features_df['sma20'] = ta.trend.sma_indicator(features_df['close'], window=20)
    features_df['sma50'] = ta.trend.sma_indicator(features_df['close'], window=50)
    features_df['sma100'] = ta.trend.sma_indicator(features_df['close'], window=100)
    
    # Price ratios & changes
    features_df['price_ema9_ratio'] = features_df['close'] / features_df['ema9'] - 1
    features_df['price_sma20_ratio'] = features_df['close'] / features_df['sma20'] - 1
    features_df['price_sma50_ratio'] = features_df['close'] / features_df['sma50'] - 1
    features_df['price_change_1'] = features_df['close'].pct_change(1)
    features_df['price_change_3'] = features_df['close'].pct_change(3)
    features_df['price_change_5'] = features_df['close'].pct_change(5)
    
    # Trend indicators
    adx_indicator = ta.trend.ADXIndicator(features_df['high'], features_df['low'], features_df['close'], window=14)
    features_df['adx'] = adx_indicator.adx()
    features_df['di_plus'] = adx_indicator.adx_pos()
    features_df['di_minus'] = adx_indicator.adx_neg()
    
    # Volatility indicators
    atr_indicator = ta.volatility.AverageTrueRange(features_df['high'], features_df['low'], features_df['close'], window=14)
    features_df['atr'] = atr_indicator.average_true_range()
    features_df['atr_pct'] = features_df['atr'] / features_df['close']
    features_df['atr_ma100'] = features_df['atr'].rolling(100).mean()
    features_df['atr_ratio'] = features_df['atr'] / features_df['atr_ma100'].replace(0, np.nan)
    
    # Bollinger Bands
    bb_indicator = ta.volatility.BollingerBands(features_df['close'], window=20)
    features_df['bb_upper'] = bb_indicator.bollinger_hband()
    features_df['bb_lower'] = bb_indicator.bollinger_lband()
    features_df['bb_width'] = (features_df['bb_upper'] - features_df['bb_lower']) / features_df['close']
    features_df['bb_pct'] = (features_df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
    
    # === Volume-based indicators ===
    # Volume moving averages
    features_df['volume_ma20'] = features_df['volume'].rolling(20).mean() 
    features_df['volume_ma50'] = features_df['volume'].rolling(50).mean()
    features_df['volume_ratio_20'] = features_df['volume'] / features_df['volume_ma20']
    features_df['volume_ratio_50'] = features_df['volume'] / features_df['volume_ma50']
    
    # Relative volume change
    features_df['volume_change_1'] = features_df['volume'].pct_change(1)
    features_df['volume_change_3'] = features_df['volume'].pct_change(3)
    
    # OBV (On-Balance Volume)
    features_df['obv'] = ta.volume.on_balance_volume(features_df['close'], features_df['volume'])
    features_df['obv_ma20'] = features_df['obv'].rolling(20).mean()
    features_df['obv_ratio'] = features_df['obv'] / features_df['obv_ma20']
    
    # === Momentum indicators ===
    # RSI
    features_df['rsi'] = ta.momentum.rsi(features_df['close'], window=14)
    features_df['rsi_ma5'] = features_df['rsi'].rolling(5).mean()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(features_df['high'], features_df['low'], features_df['close'], window=14, smooth_window=3)
    features_df['stoch_k'] = stoch.stoch()
    features_df['stoch_d'] = stoch.stoch_signal()
    features_df['stoch_diff'] = features_df['stoch_k'] - features_df['stoch_d']
    
    # Commodity Channel Index (CCI)
    features_df['cci'] = ta.trend.cci(features_df['high'], features_df['low'], features_df['close'], window=20)
    
    # Williams %R
    features_df['williams_r'] = ta.momentum.williams_r(features_df['high'], features_df['low'], features_df['close'], lbp=14)
    
    # Stochastic RSI
    stoch_rsi = ta.momentum.StochRSIIndicator(features_df['close'], window=14, smooth1=3, smooth2=3)
    features_df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
    features_df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()
    
    # Rate of Change - extended with short-term windows
    features_df['roc_3'] = ta.momentum.roc(features_df['close'], window=3)
    features_df['roc_5'] = ta.momentum.roc(features_df['close'], window=5)
    features_df['roc_10'] = ta.momentum.roc(features_df['close'], window=10)
    features_df['roc_20'] = ta.momentum.roc(features_df['close'], window=20)
    
    # === Pattern indicators ===
    # Donchian Channels
    features_df['donchian_high_20'] = features_df['high'].rolling(20).max()
    features_df['donchian_low_20'] = features_df['low'].rolling(20).min()
    features_df['donchian_width'] = (features_df['donchian_high_20'] - features_df['donchian_low_20']) / features_df['close']
    
    # Price vs key levels
    features_df['dist_from_high_20'] = features_df['close'] / features_df['donchian_high_20'] - 1
    features_df['dist_from_low_20'] = features_df['close'] / features_df['donchian_low_20'] - 1
    
    # VWAP-related
    # Simple approximation of daily VWAP using window
    features_df['vwap_24'] = (features_df['volume'] * features_df['close']).rolling(24).sum() / features_df['volume'].rolling(24).sum()
    features_df['vwap_ratio'] = features_df['close'] / features_df['vwap_24'] - 1
    
    # Ichimoku Cloud components
    high_9 = features_df['high'].rolling(9).max()
    low_9 = features_df['low'].rolling(9).min()
    features_df['tenkan_sen'] = (high_9 + low_9) / 2
    
    high_26 = features_df['high'].rolling(26).max()
    low_26 = features_df['low'].rolling(26).min()
    features_df['kijun_sen'] = (high_26 + low_26) / 2
    
    features_df['senkou_span_a'] = ((features_df['tenkan_sen'] + features_df['kijun_sen']) / 2).shift(26)
    features_df['price_vs_kumo'] = features_df['close'] - features_df['senkou_span_a']

    # === Trend indicators ===
    # Supertrend
    def calculate_supertrend(high, low, close, atr_period=14, multiplier=3):
        atr = ta.volatility.AverageTrueRange(high, low, close, window=atr_period).average_true_range()
        hl2 = (high + low) / 2
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        supertrend = pd.Series(index=close.index)
        direction = pd.Series(index=close.index)

        for i in range(len(close)):
            if i == 0:
                supertrend[i] = upperband[i]
                direction[i] = 1
            else:
                if close[i] > supertrend[i-1]:
                    supertrend[i] = lowerband[i]
                    direction[i] = 1
                elif close[i] < supertrend[i-1]:
                    supertrend[i] = upperband[i]
                    direction[i] = -1
                else:
                    supertrend[i] = supertrend[i-1]
                    direction[i] = direction[i-1]

        return supertrend, direction

    features_df['supertrend'], features_df['supertrend_direction'] = calculate_supertrend(features_df['high'], features_df['low'], features_df['close'])

    # === Volatility indicators ===
    # Historical Volatility (30)
    features_df['log_return'] = np.log(features_df['close'] / features_df['close'].shift(1))
    features_df['historical_volatility_30'] = features_df['log_return'].rolling(window=30).std() * np.sqrt(252)

    # === Volume-based indicators ===
    # Money Flow Index (MFI)
    features_df['mfi'] = ta.volume.money_flow_index(features_df['high'], features_df['low'], features_df['close'], features_df['volume'], window=14)
    
    # Chaikin Money Flow
    features_df['cmf'] = ta.volume.chaikin_money_flow(features_df['high'], features_df['low'], features_df['close'], features_df['volume'], window=20)
    
    # Calculate ema9_ratio (ema9 to closing price ratio)
    if 'ema9' in features_df.columns:
        # Create ema9_ratio = ema9/close
        # Values above 1 indicate price is below EMA (bearish)
        # Values below 1 indicate price is above EMA (bullish)
        features_df['ema9_ratio'] = features_df['ema9'] / features_df['close']
        
        # Subtract 1 and multiply by 100 to get percentage difference from price
        features_df['ema9_ratio'] = (features_df['ema9_ratio'] - 1) * 100
        
        logging.info(f"Created ema9_ratio showing EMA9 vs price: min={features_df['ema9_ratio'].min():.2f}%, max={features_df['ema9_ratio'].max():.2f}%")
    
    # Handle any NaN values - replace with 0
    features_df = features_df.fillna(0)
    
    # Skip warmup period to avoid initialization effects
    if len(features_df) > warmup_candles:
        original_len = len(features_df)
        features_df = features_df.iloc[warmup_candles:]
        df = df.iloc[warmup_candles:]
        logging.info(f"Skipped first {warmup_candles} candles as warmup period. Reduced dataset from {original_len} to {len(features_df)} rows.")
    else:
        logging.warning(f"Dataset too small to skip warmup period. Consider reducing warmup_candles (currently {warmup_candles}).")
    
    # Detect and report any potential initialization values that might still exist
    potential_init_values = (features_df == 0).sum() + (features_df == -100).sum()
    if potential_init_values.sum() > 0:
        logging.warning("Potential initialization values found in data after warmup period:")
        for col, count in potential_init_values.items():
            if count > 0:
                logging.warning(f"  - {col}: {count} potential initialization values")
    
    # Reset indices after warmup period
    df = df.reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    
    # Apply feature transformations
    if 'volume_ratio_50' in features_df.columns:
        features_df['volume_ratio_50'] = np.sign(features_df['volume_ratio_50']) * np.sqrt(np.abs(features_df['volume_ratio_50']))
        logging.info(f"Applied square root transformation to volume_ratio_50 to reduce extreme values while preserving more information")
    
    if 'supertrend' in features_df.columns:
        features_df['supertrend'] = features_df['supertrend'] / df['close'].values
        logging.info(f"Transformed supertrend to a ratio by dividing by closing price")
    
    if 'historical_volatility_30' in features_df.columns:
        mean_close = df['close'].mean()
        features_df['historical_volatility_30'] = features_df['historical_volatility_30'] / mean_close
        logging.info(f"Normalized historical_volatility_30 by dividing by mean closing price")
    
    if 'obv_ratio' in features_df.columns:
        features_df['obv_ratio'] = np.log1p(features_df['obv_ratio'])
        logging.info("Applied log transformation to obv_ratio to scale it down.")
    
    # Filter out initial rows where ATR is zero if present
    if 'atr_ratio' in features_df.columns:
        non_zero_atr_index = features_df['atr_ratio'].ne(0).idxmax()
        if non_zero_atr_index > 0:
            features_df = features_df.iloc[non_zero_atr_index:]
            df = df.iloc[non_zero_atr_index:].reset_index(drop=True)
            features_df = features_df.reset_index(drop=True)
    
    # Fill any remaining NaN values with 0
    features_df = features_df.fillna(0)
    logging.info("Filled NaN values in features_df with 0.")
    
    return features_df, df