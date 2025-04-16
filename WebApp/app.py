from flask import Flask, render_template, jsonify
import datetime
import logging
import boto3
import joblib
from sklearn.decomposition import PCA
import tarfile
import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Set up basic logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('web_indicators')

# Add parent directory to path for imports
base_dir = Path(__file__).resolve().parents[1]  # This points to MachinePro directory
sys.path.append(str(base_dir))

# Initialize Flask app
app = Flask(__name__)

# Empty placeholder data
current_data = {
    'formatted_price': '0.00',
    'last_updated': 'Never',
    'market_regime': 'Unknown',
    'trade_signal': 'Neutral',
    'pca_components': None,
    'has_error': False,
    'error_message': '',
    'pca_model_loaded': False,
    'history': [],
    'using_real_data': False,
    'rsi': 50,
    'macd_diff': 0,
    'atr_ratio': 1,
    'pca_interpretations': [],
    'market_description': None
}

def get_market_description(pc1, pc2, pc3):
    """Generate detailed market description based on PCA values"""
    descriptions = {
        'pc1': {
            'name': 'Trend Strength',
            'description': 'Measures the overall trend strength and momentum. High values indicate strong trend, low values suggest weak or reversing trend.',
            'interpretation': {
                '>2': 'Very Strong Trend',
                '>1': 'Strong Trend',
                '>0': 'Mild Trend',
                '>-1': 'Weak Trend',
                '>-2': 'Reversing Trend',
                '<=-2': 'Strong Reversal'
            }
        },
        'pc2': {
            'name': 'Volume & Short-Term Trend',
            'description': 'Combines volume analysis with short-term price action. High values suggest strong volume-backed moves, low values indicate distribution.',
            'interpretation': {
                '>2': 'Very High Volume',
                '>1': 'High Volume',
                '>0': 'Moderate Volume',
                '>-1': 'Low Volume',
                '>-2': 'Very Low Volume',
                '<=-2': 'Extreme Distribution'
            }
        },
        'pc3': {
            'name': 'Momentum Shift',
            'description': 'Captures momentum changes and potential reversals. High values suggest momentum building, low values indicate exhaustion.',
            'interpretation': {
                '>2': 'Momentum Spike',
                '>1': 'Building Momentum',
                '>0': 'Stable Momentum',
                '>-1': 'Fading Momentum',
                '>-2': 'Momentum Exhaustion',
                '<=-2': 'Reversal Signal'
            }
        }
    }

    def get_pc_description(value, pc):
        if value > 2:
            return pc['interpretation']['>2']
        elif value > 1:
            return pc['interpretation']['>1']
        elif value > 0:
            return pc['interpretation']['>0']
        elif value > -1:
            return pc['interpretation']['>-1']
        elif value > -2:
            return pc['interpretation']['>-2']
        else:
            return pc['interpretation']['<=-2']

    # Generate component descriptions
    pc1_desc = get_pc_description(pc1, descriptions['pc1'])
    pc2_desc = get_pc_description(pc2, descriptions['pc2'])
    pc3_desc = get_pc_description(pc3, descriptions['pc3'])

    # Generate overall market summary
    if pc1 > 2 and pc2 < -2 and pc3 < -2:
        summary = "Strong Bear Market - High trend strength with extreme distribution and reversal signals"
        recommendation = "Consider short positions with tight stops"
    elif pc1 > 2 and pc2 > 2 and pc3 > 0:
        summary = "Strong Bull Market - High trend strength with strong volume backing"
        recommendation = "Look for long entries on pullbacks"
    elif pc1 > 1 and pc2 > 0.5 and pc3 > 0:
        summary = "Emerging Bull Market - Building trend with moderate volume"
        recommendation = "Consider long positions with wider stops"
    elif pc1 < -1 and pc2 < -0.5 and pc3 < 0:
        summary = "Emerging Bear Market - Developing downtrend with distribution"
        recommendation = "Consider short positions with wider stops"
    elif pc1 > 2 and pc2 < -2:
        summary = "Trend Reversal Warning - Strong trend but extreme distribution"
        recommendation = "Consider taking profits on longs"
    elif pc1 < -2 and pc2 > 2:
        summary = "Downtrend Reversing - Strong distribution but potential reversal"
        recommendation = "Consider taking profits on shorts"
    elif abs(pc1) > 2 and abs(pc2) > 2:
        summary = "High Volatility - Extreme conditions in both trend and volume"
        recommendation = "Exercise caution and wait for clearer signals"
    else:
        summary = "Sideways Market - Mixed signals across components"
        recommendation = "Wait for clearer trend development"

    return {
        'components': [
            {
                'name': descriptions['pc1']['name'],
                'value': pc1,
                'description': descriptions['pc1']['description'],
                'interpretation': pc1_desc
            },
            {
                'name': descriptions['pc2']['name'],
                'value': pc2,
                'description': descriptions['pc2']['description'],
                'interpretation': pc2_desc
            },
            {
                'name': descriptions['pc3']['name'],
                'value': pc3,
                'description': descriptions['pc3']['description'],
                'interpretation': pc3_desc
            }
        ],
        'summary': summary,
        'recommendation': recommendation
    }

def load_pca_model():
    """Load the PCA model and get predictions"""
    global current_data
    try:
        # Import PCA Predict class
        from PCA.pca_predict import PCAPredict
        
        # Initialize Bitget client
        from utils.bitget_futures import BitgetFutures
        key_path = base_dir / 'config' / 'config.json'
        with open(key_path, "r") as f:
            api_setup = json.load(f)['bitget']
        bitget_client = BitgetFutures(api_setup)
        
        # Get current price from Bitget
        current_price = bitget_client.fetch_ticker('ETH/USDT:USDT')['last']
        current_data['formatted_price'] = f"{current_price:.2f}"
        
        # Initialize PCA Predict
        pca_predict = PCAPredict(bitget_client)
        
        # Get data and predictions
        data = pca_predict.fetch_data()
        features_df, df = pca_predict.calculate_features(data)
        df = pca_predict.filter_features(features_df)
        
        # Load model and get predictions
        pca_model, scaler = pca_predict.load_pca_model_and_scaler()
        pca1, pca2, pca3 = pca_predict.transform_features_and_scale(pca_model, scaler, df)
        
        # Get market description
        market_desc = get_market_description(pca1, pca2, pca3)
        current_data['market_description'] = market_desc
        
        # Get latest indicators from features_df
        latest_indicators = features_df.iloc[-1]
        current_data['rsi'] = latest_indicators.get('rsi', 50)
        current_data['macd_diff'] = latest_indicators.get('macd_diff', 0)
        current_data['atr_ratio'] = latest_indicators.get('atr_ratio', 1)
        current_data['obv'] = latest_indicators.get('obv', 0)
        current_data['volume_ratio'] = latest_indicators.get('volume_ratio_50', 1)
        
        # Update current_data with new interpretations
        current_data['pca_interpretations'] = [
            {
                'id': 1,
                'name': 'Trend & Momentum Strength',
                'current_value': float(pca1),
                'signal': 'Bullish' if pca1 > 1 else 'Bearish' if pca1 < -1 else 'Neutral',
                'variance': '44.06',
                'description': 'Heavily influenced by trend-following indicators and upper-bound levels. Represents bullish momentum or long-term trend strength.',
                'trading_guidance': 'Good for detecting breakouts or strong uptrends.'
            },
            {
                'id': 2,
                'name': 'Volume & Short-Term Trend',
                'current_value': float(pca2),
                'signal': 'Accumulation' if pca2 > 0.5 else 'Distribution' if pca2 < -0.5 else 'Neutral',
                'variance': '37.15',
                'description': 'Mix of volume (OBV), short-term EMAs, and trend signals. Represents short-term trend shifts and volume-backed movement.',
                'trading_guidance': 'Great for trend confirmation or volume-backed signals.'
            },
            {
                'id': 3,
                'name': 'Momentum Shift',
                'current_value': float(pca3),
                'signal': 'Momentum Up' if pca3 > 0 else 'Momentum Down' if pca3 < 0 else 'Neutral',
                'variance': '6.62',
                'description': 'Focuses on momentum and mean-reversion relationships. Captures divergence, momentum turning points, or how far price strays from moving averages.',
                'trading_guidance': 'Useful for detecting trend exhaustion or reversal setups.'
            }
        ]
        
        # Set market regime and trade signal based on PCA values
        if pca1 > 2 and pca2 < -2 and pca3 < -2:
            current_data['market_regime'] = 'Strong Bear'
            current_data['trade_signal'] = 'Go Short'
        elif pca1 > 2 and pca2 > 2 and pca3 > 0:
            current_data['market_regime'] = 'Strong Bull'
            current_data['trade_signal'] = 'Go Long'
        elif pca1 > 1 and pca2 > 0.5 and pca3 > 0:
            current_data['market_regime'] = 'Emerging Bull'
            current_data['trade_signal'] = 'Consider Long'
        elif pca1 < -1 and pca2 < -0.5 and pca3 < 0:
            current_data['market_regime'] = 'Emerging Bear'
            current_data['trade_signal'] = 'Consider Short'
        elif pca1 > 2 and pca2 < -2:
            current_data['market_regime'] = 'Trend Reversal'
            current_data['trade_signal'] = 'Exit Long'
        elif pca1 < -2 and pca2 > 2:
            current_data['market_regime'] = 'Downtrend Reversing'
            current_data['trade_signal'] = 'Exit Short'
        elif abs(pca1) > 2 and abs(pca2) > 2:
            current_data['market_regime'] = 'High Volatility'
            current_data['trade_signal'] = 'Caution'
        else:
            current_data['market_regime'] = 'Sideways'
            current_data['trade_signal'] = 'Hold'
            
        current_data['last_updated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_data['using_real_data'] = True
        current_data['pca_model_loaded'] = True
        
        return True
        
    except Exception as e:
        logger.error(f"Error in PCA analysis: {str(e)}")
        current_data['has_error'] = True
        current_data['error_message'] = str(e)
        return False

# Basic CSS helper
@app.template_filter('safe_width')
def safe_width(value):
    """Convert value to safe width percentage for CSS"""
    try:
        val = float(value)
        return max(0, min(100, val))
    except:
        return 0

# Add a safe round filter to handle undefined values
@app.template_filter('round')
def safe_round(value, precision=0):
    """Safely round a value, returns 0 if value is undefined or not a number"""
    try:
        return round(float(value), precision)
    except:
        return 0

# Basic routes
@app.route('/')
def index():
    return render_template('index.html', 
                          market_data=current_data,
                          now=datetime.datetime.now())

@app.route('/update')
def update():
    """Refresh the PCA model and market data when Update Now button is clicked"""
    try:
        # Store current data for history before updating
        if current_data['pca_model_loaded']:
            history_entry = {
                'market_regime': current_data['market_regime'],
                'trade_signal': current_data['trade_signal'],
                'pca1': current_data['pca_interpretations'][0]['current_value'],
                'pca2': current_data['pca_interpretations'][1]['current_value'],
                'pca3': current_data['pca_interpretations'][2]['current_value'],
                'last_updated': current_data['last_updated'],
                'formatted_price': current_data['formatted_price']
            }
            
            # Add to history (limit to last 50 entries)
            current_data['history'].append(history_entry)
            if len(current_data['history']) > 50:
                current_data['history'] = current_data['history'][-50:]
        
        # Reload the PCA model and market data
        logger.info("Manual update requested - refreshing PCA model and market data")
        success = load_pca_model()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Data updated successfully',
                'data': {
                    'market_regime': current_data['market_regime'],
                    'trade_signal': current_data['trade_signal'],
                    'formatted_price': current_data['formatted_price'],
                    'last_updated': current_data['last_updated'],
                    'pca_components': [
                        {
                            'value': comp['current_value'],
                            'name': comp['name'],
                            'signal': comp['signal']
                        } for comp in current_data['pca_interpretations']
                    ]
                }
            })
        else:
            return jsonify({'status': 'error', 'message': 'Failed to update PCA model'}), 500
            
    except Exception as e:
        logger.error(f"Error during manual update: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/pca-analysis')
def pca_analysis():
    return render_template('pca_analysis.html', 
                          market_data=current_data,
                          now=datetime.datetime.now())

@app.route('/history')
def history():
    return render_template('history.html', 
                          market_data=current_data,
                          now=datetime.datetime.now())

# Run the app
if __name__ == '__main__':
    # Load PCA model first before starting the app
    success = load_pca_model()
    counter = 0
    if not success:
        logger.error("Failed to load initial PCA model. Check configuration and data.")
        counter += 1
        if counter > 5:
            sys.exit(1)
        else:
            logger.info("Retrying PCA model loading...")
            success = load_pca_model()

    logger.info("Initial PCA model loaded successfully")
    logger.info("Starting Web Indicators app")
    app.run(debug=True, host='0.0.0.0', port=5000)
