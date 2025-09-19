import os
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class MinIOSyncManager:
    """
    MinIO Sync Manager class for downloading files from MinIO
    Only keeps sync download functionality with credentials from environment variables
    """
    
    def __init__(self, config=None):
        """
        Initialize MinIO client with credentials from environment variables
        
        Args:
            config (dict): Configuration dictionary containing MinIO settings
        """
        # Get credentials from environment variables
        self.access_key = os.getenv('MINIO_ACCESS_KEY')
        self.secret_key = os.getenv('MINIO_SECRET_KEY')
        
        if not all([self.access_key, self.secret_key]):
            raise ValueError("MinIO credentials not found in environment variables. "
                           "Please set MINIO_ACCESS_KEY, and MINIO_SECRET_KEY")
        
        # Get other parameters from config
        if config and 'minio' in config:
            minio_config = config['minio']
            self.endpoint = minio_config.get('endpoint', '')
            self.bucket_name = minio_config.get('bucket_name', '')
            self.folder_prefix = minio_config.get('folder_prefix', '')
            self.secure = minio_config.get('secure', True)
            self.enabled = minio_config.get('enabled', True)
        else:
            raise ValueError("MinIO configuration not found in config")
        
        # Initialize MinIO client
        self.client = Minio(
            endpoint=self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )
    
    def _check_bucket_exists(self):
        """Check if the configured bucket exists"""
        try:
            return self.client.bucket_exists(self.bucket_name)
        except S3Error as e:
            logger.error(f"Erreur lors de la vérification du bucket: {e}")
            return False
    
    def download_missing_files(self, local_folder_path="./input", minio_folder_prefix=None):
        """
        Download only files that don't exist locally from MinIO
        
        Args:
            local_folder_path (str): Local folder path to download files to
            minio_folder_prefix (str): Optional override for folder prefix in MinIO
        
        Returns:
            dict: Download statistics (downloaded, skipped, errors, total_objects)
        """
        if minio_folder_prefix is None:
            minio_folder_prefix = self.folder_prefix
        
        stats = {
            'downloaded': 0,
            'skipped': 0,
            'errors': 0,
            'total_objects': 0
        }
        
        try:
            # Check if bucket exists
            if not self._check_bucket_exists():
                logger.error(f"Erreur: Le bucket '{self.bucket_name}' n'existe pas")
                return stats
            
            # Create local folder if it doesn't exist
            os.makedirs(local_folder_path, exist_ok=True)
            
            # List all objects in the bucket with the prefix
            objects = self.client.list_objects(self.bucket_name, prefix=minio_folder_prefix, recursive=True)
            
            logger.info(f"Vérification des fichiers à télécharger depuis MinIO...")
            
            for obj in objects:
                stats['total_objects'] += 1
                
                try:
                    # Create local path preserving structure
                    if minio_folder_prefix:
                        # Remove prefix from object path
                        relative_path = obj.object_name[len(minio_folder_prefix):].lstrip('/')
                    else:
                        relative_path = obj.object_name
                    
                    local_file_path = os.path.join(local_folder_path, relative_path)
                    
                    # Check if file already exists locally
                    if os.path.exists(local_file_path):
                        stats['skipped'] += 1
                        continue
                    
                    # Create parent directories if necessary
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    
                    # Download the file
                    self.client.fget_object(
                        bucket_name=self.bucket_name,
                        object_name=obj.object_name,
                        file_path=local_file_path
                    )
                    
                    # logger.info(f"Téléchargé: {obj.object_name} -> {local_file_path}")
                    stats['downloaded'] += 1
                    
                except S3Error as e:
                    logger.error(f"Erreur MinIO lors du téléchargement de {obj.object_name}: {e}")
                    stats['errors'] += 1
                except Exception as e:
                    logger.error(f"Erreur lors du téléchargement de {obj.object_name}: {e}")
                    stats['errors'] += 1
            
            
            return stats    
            
        except S3Error as e:
            logger.error(f"Erreur MinIO: {e}")
            stats['errors'] += 1
            return stats
        except Exception as e:
            logger.error(f"Erreur générale: {e}")
            stats['errors'] += 1
            return stats
    
    def sync_to_local(self, local_folder_path="./input"):
        """
        Synchronize files from MinIO to local folder
        
        Args:
            local_folder_path (str): Local destination folder
        
        Returns:
            dict: Synchronization statistics
        """
        logger.info(f"\n{'='*60}")
        logger.info("SYNCHRONISATION MINIO -> LOCAL")
        logger.info(f"{'='*60}")
        logger.info(f"Source: {self.bucket_name}/{self.folder_prefix}")
        logger.info(f"Destination: {local_folder_path}")
        logger.info(f"Endpoint: {self.endpoint}")
        
        # Use selective download function
        stats = self.download_missing_files(
            local_folder_path=local_folder_path,
            minio_folder_prefix=self.folder_prefix
        )
        
        logger.info(f"{'='*60}")
        
        return stats
 