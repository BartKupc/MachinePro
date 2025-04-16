import boto3
import logging
from pathlib import Path

# === CONFIG ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
base_dir = Path(__file__).resolve().parents[1]
BUCKET_NAME = "machinepro-models"

# Initialize S3 client
s3 = boto3.client('s3', region_name='eu-west-1')

def upload_model(model_name, local_dir):
    """
    Upload model to S3
    Args:
        model_name: Name of the model file
        local_dir: Directory where the model is located
    """
    try:
        local_path = Path(local_dir) / model_name
        
        if not local_path.exists():
            logging.error(f"Model not found at {local_path}")
            return False
            
        logging.info(f"Uploading {local_path} to S3 bucket: {BUCKET_NAME}")
        s3.upload_file(str(local_path), BUCKET_NAME, model_name)
        logging.info("✅ Upload successful")
        return True
    except Exception as e:
        logging.error(f"❌ Upload failed: {str(e)}")
        return False

def download_model(model_name):
    """Download model from S3 to live/models/"""
    try:
        # Always save to live/models directory
        save_dir = base_dir / 'live' / 'models'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / model_name
        logging.info(f"Downloading from {BUCKET_NAME} to {save_path}")
        s3.download_file(BUCKET_NAME, model_name, str(save_path))
        logging.info("✅ Download successful")
        return save_path
    except Exception as e:
        logging.error(f"❌ Download failed: {str(e)}")
        return None

if __name__ == "__main__":
    # Upload example - directory is configurable
    upload_dir = base_dir / 'XG' / 'model' / 'big'
    upload_model('xgb_regressor.pkl', upload_dir)
    
    # Download - always saves to live/models/
    download_model('xgb_regressor.pkl')
