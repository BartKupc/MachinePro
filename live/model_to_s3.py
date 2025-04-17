import boto3
import logging
from pathlib import Path

# === CONFIG ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
base_dir = Path(__file__).resolve().parents[1]
BUCKET_NAME = "machinepro-models"

# Initialize S3 client
s3 = boto3.client('s3', region_name='eu-west-1')

def upload_model(model_name, local_dir, s3_name=None):
    """
    Upload model to S3
    Args:
        model_name: Name of the model file locally
        local_dir: Directory where the model is located
        s3_name: Optional different name to use in S3 (if None, uses model_name)
    """
    try:
        local_path = Path(local_dir) / model_name
        
        if not local_path.exists():
            logging.error(f"Model not found at {local_path}")
            return False
        
        # Use s3_name if provided, otherwise use original model_name
        upload_name = s3_name if s3_name else model_name
            
        logging.info(f"Uploading {local_path} to S3 bucket: {BUCKET_NAME} as {upload_name}")
        s3.upload_file(str(local_path), BUCKET_NAME, upload_name)
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
    # Upload examples
    # upload_dir = base_dir / 'XG' / 'model' / 'small'
    
    # Upload with same name
    # upload_model('xgb_regressor.pkl', upload_dir)
    
    # Upload with different name
    # upload_model('xgb_regressor.pkl', upload_dir, s3_name='xgb_regressor_small.pkl')
    
    # Download - always saves to live/models/
    download_model('xgb_regressor_small.pkl')
