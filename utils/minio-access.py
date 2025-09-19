import os
import argparse
import sys
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def upload_folder_to_minio(
    endpoint,
    access_key,
    secret_key,
    bucket_name,
    local_folder_path,
    minio_folder_prefix="",
    secure=False  # Changé à False par défaut
):
    """
    Upload un dossier complet vers MinIO
   
    Args:
        endpoint (str): URL du serveur MinIO (ex: 'localhost:9000')
        access_key (str): Votre access key MinIO
        secret_key (str): Votre secret key MinIO
        bucket_name (str): Nom du bucket de destination
        local_folder_path (str): Chemin vers le dossier local à uploader
        minio_folder_prefix (str): Préfixe pour organiser les fichiers dans MinIO
        secure (bool): Utiliser HTTPS (True) ou HTTP (False)
    """
   
    # Créer le client MinIO
    client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )
   
    try:
        # Vérifier si le bucket existe, sinon le créer
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' créé avec succès")
        else:
            print(f"Bucket '{bucket_name}' existe déjà")
       
        # Parcourir tous les fichiers du dossier
        uploaded_files = 0
        for root, dirs, files in os.walk(local_folder_path):
            for file in files:
                local_file_path = os.path.join(root, file)
               
                # Créer le chemin relatif pour MinIO
                relative_path = os.path.relpath(local_file_path, local_folder_path)
                minio_object_name = os.path.join(minio_folder_prefix, relative_path).replace(os.sep, '/')
               
                try:
                    # Uploader le fichier
                    client.fput_object(
                        bucket_name=bucket_name,
                        object_name=minio_object_name,
                        file_path=local_file_path
                    )
                    print(f"Uploaded: {local_file_path} -> {minio_object_name}")
                    uploaded_files += 1
                   
                except S3Error as e:
                    print(f"Erreur lors de l'upload de {local_file_path}: {e}")
       
        print(f"\n Upload terminé! {uploaded_files} fichiers uploadés avec succès")
       
    except S3Error as e:
        print(f"Erreur MinIO: {e}")
    except Exception as e:
        print(f"Erreur générale: {e}")


def download_from_minio(
    endpoint,
    access_key,
    secret_key,
    bucket_name,
    object_name,
    local_file_path,
    secure=False
):
    """
    Télécharge un fichier depuis MinIO
    
    Args:
        endpoint (str): URL du serveur MinIO (ex: 'localhost:9000')
        access_key (str): Votre access key MinIO
        secret_key (str): Votre secret key MinIO
        bucket_name (str): Nom du bucket source
        object_name (str): Nom de l'objet dans MinIO à télécharger
        local_file_path (str): Chemin local où sauvegarder le fichier
        secure (bool): Utiliser HTTPS (True) ou HTTP (False)
    
    Returns:
        bool: True si le téléchargement réussit, False sinon
    """
    
    # Créer le client MinIO
    client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )
    
    try:
        # Vérifier si le bucket existe
        if not client.bucket_exists(bucket_name):
            print(f"Erreur: Le bucket '{bucket_name}' n'existe pas")
            return False
        
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Télécharger le fichier
        client.fget_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=local_file_path
        )
        
        print(f"Téléchargé avec succès: {object_name} -> {local_file_path}")
        return True
        
    except S3Error as e:
        print(f"Erreur MinIO lors du téléchargement: {e}")
        return False
    except Exception as e:
        print(f"Erreur générale lors du téléchargement: {e}")
        return False


def download_folder_from_minio(
    endpoint,
    access_key,
    secret_key,
    bucket_name,
    minio_folder_prefix="",
    local_folder_path="./downloads",
    secure=False
):
    """
    Télécharge un dossier complet depuis MinIO
    
    Args:
        endpoint (str): URL du serveur MinIO (ex: 'localhost:9000')
        access_key (str): Votre access key MinIO
        secret_key (str): Votre secret key MinIO
        bucket_name (str): Nom du bucket source
        minio_folder_prefix (str): Préfixe du dossier dans MinIO à télécharger
        local_folder_path (str): Chemin local où sauvegarder les fichiers
        secure (bool): Utiliser HTTPS (True) ou HTTP (False)
    
    Returns:
        int: Nombre de fichiers téléchargés avec succès
    """
    
    # Créer le client MinIO
    client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )
    
    try:
        # Vérifier si le bucket existe
        if not client.bucket_exists(bucket_name):
            print(f"Erreur: Le bucket '{bucket_name}' n'existe pas")
            return 0
        
        # Créer le dossier local s'il n'existe pas
        os.makedirs(local_folder_path, exist_ok=True)
        
        downloaded_files = 0
        
        # Lister tous les objets dans le bucket avec le préfixe
        objects = client.list_objects(bucket_name, prefix=minio_folder_prefix, recursive=True)
        
        for obj in objects:
            try:
                # Créer le chemin local en préservant la structure
                if minio_folder_prefix:
                    # Enlever le préfixe du chemin de l'objet
                    relative_path = obj.object_name[len(minio_folder_prefix):].lstrip('/')
                else:
                    relative_path = obj.object_name
                
                local_file_path = os.path.join(local_folder_path, relative_path)
                
                # Créer les répertoires parents si nécessaire
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # Télécharger le fichier
                client.fget_object(
                    bucket_name=bucket_name,
                    object_name=obj.object_name,
                    file_path=local_file_path
                )
                
                print(f"Téléchargé: {obj.object_name} -> {local_file_path}")
                downloaded_files += 1
                
            except S3Error as e:
                print(f"Erreur lors du téléchargement de {obj.object_name}: {e}")
        
        print(f"\nTéléchargement terminé! {downloaded_files} fichiers téléchargés avec succès")
        return downloaded_files
        
    except S3Error as e:
        print(f"Erreur MinIO: {e}")
        return 0
    except Exception as e:
        print(f"Erreur générale: {e}")
        return 0


def list_objects_in_minio(
    endpoint,
    access_key,
    secret_key,
    bucket_name,
    prefix="",
    secure=False
):
    """
    Liste tous les objets dans un bucket MinIO
    
    Args:
        endpoint (str): URL du serveur MinIO (ex: 'localhost:9000')
        access_key (str): Votre access key MinIO
        secret_key (str): Votre secret key MinIO
        bucket_name (str): Nom du bucket
        prefix (str): Préfixe pour filtrer les objets
        secure (bool): Utiliser HTTPS (True) ou HTTP (False)
    
    Returns:
        list: Liste des noms d'objets
    """
    
    # Créer le client MinIO
    client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )
    
    try:
        # Vérifier si le bucket existe
        if not client.bucket_exists(bucket_name):
            print(f"Erreur: Le bucket '{bucket_name}' n'existe pas")
            return []
        
        # Lister les objets
        objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)
        object_names = []
        
        print(f"Objets dans le bucket '{bucket_name}' avec le préfixe '{prefix}':")
        for obj in objects:
            print(f"  - {obj.object_name} (Taille: {obj.size} bytes, Modifié: {obj.last_modified})")
            object_names.append(obj.object_name)
        
        return object_names
        
    except S3Error as e:
        print(f"Erreur MinIO: {e}")
        return []
    except Exception as e:
        print(f"Erreur générale: {e}")
        return []


def main():
    """
    Fonction principale avec gestion des arguments de ligne de commande
    """
    parser = argparse.ArgumentParser(
        description="Utilitaire MinIO pour uploader/télécharger des fichiers et dossiers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Lister les objets dans un bucket
  python minio-access.py list --bucket mon-bucket --prefix uploads/

  # Uploader un dossier
  python minio-access.py upload-folder --bucket mon-bucket --local-path ./input --prefix uploads/

  # Télécharger un fichier
  python minio-access.py download-file --bucket mon-bucket --object-name uploads/file.txt --local-path ./downloads/file.txt

  # Télécharger un dossier complet
  python minio-access.py download-folder --bucket mon-bucket --prefix uploads/ --local-path ./downloads

Configuration par défaut:
  - Endpoint: variable MINIO_ENDPOINT ou object.bdc.atlascs.ma
  - Access Key: variable MINIO_ACCESS_KEY
  - Secret Key: variable MINIO_SECRET_KEY
  - Bucket: variable MINIO_BUCKET
  - Secure: True (HTTPS)
  
Variables d'environnement:
  export MINIO_ENDPOINT="your_endpoint"
  export MINIO_ACCESS_KEY="your_access_key"
  export MINIO_SECRET_KEY="your_secret_key"
  export MINIO_BUCKET="your_bucket"
        """
    )
    
    # Arguments de configuration MinIO
    parser.add_argument("--endpoint", default=os.getenv("MINIO_ENDPOINT", "your_endpoint"), 
                       help="Endpoint MinIO (défaut: variable MINIO_ENDPOINT ou your_endpoint)")
    parser.add_argument("--access-key", default=os.getenv("MINIO_ACCESS_KEY"),
                       help="Access Key MinIO (défaut: variable MINIO_ACCESS_KEY)")
    parser.add_argument("--secret-key", default=os.getenv("MINIO_SECRET_KEY"),
                       help="Secret Key MinIO (défaut: variable MINIO_SECRET_KEY)")
    parser.add_argument("--bucket", default=os.getenv("MINIO_BUCKET"),
                       help="Nom du bucket MinIO (défaut: variable MINIO_BUCKET ou your_bucket)")
    parser.add_argument("--secure", action="store_true", default=True,
                       help="Utiliser HTTPS (défaut: True)")
    parser.add_argument("--no-secure", action="store_false", dest="secure",
                       help="Utiliser HTTP au lieu de HTTPS")
    
    # Sous-commandes
    subparsers = parser.add_subparsers(dest="operation", help="Opération à effectuer")
    
    # Commande list
    list_parser = subparsers.add_parser("list", help="Lister les objets dans le bucket")
    list_parser.add_argument("--prefix", default="", 
                           help="Préfixe pour filtrer les objets")
    
    # Commande upload-folder
    upload_parser = subparsers.add_parser("upload-folder", help="Uploader un dossier complet")
    upload_parser.add_argument("--local-path", required=True,
                             help="Chemin local du dossier à uploader")
    upload_parser.add_argument("--prefix", default="",
                             help="Préfixe dans MinIO pour organiser les fichiers")
    
    # Commande download-file
    download_file_parser = subparsers.add_parser("download-file", help="Télécharger un fichier")
    download_file_parser.add_argument("--object-name", required=True,
                                    help="Nom de l'objet dans MinIO")
    download_file_parser.add_argument("--local-path", required=True,
                                    help="Chemin local où sauvegarder le fichier")
    
    # Commande download-folder
    download_folder_parser = subparsers.add_parser("download-folder", help="Télécharger un dossier complet")
    download_folder_parser.add_argument("--prefix", default="",
                                      help="Préfixe du dossier dans MinIO")
    download_folder_parser.add_argument("--local-path", default="./downloads",
                                      help="Chemin local où sauvegarder les fichiers (défaut: ./downloads)")
    
    # Parser les arguments
    args = parser.parse_args()
    
    # Vérifier qu'une opération a été spécifiée
    if not args.operation:
        parser.print_help()
        sys.exit(1)
    
    # Vérifier que les credentials sont fournis
    if not args.access_key:
        print("Erreur: Access Key manquant. Utilisez --access-key ou définissez la variable MINIO_ACCESS_KEY")
        sys.exit(1)
    
    if not args.secret_key:
        print("Erreur: Secret Key manquant. Utilisez --secret-key ou définissez la variable MINIO_SECRET_KEY")
        sys.exit(1)
    
    if not args.bucket:
        print("Erreur: Bucket manquant. Utilisez --bucket ou définissez la variable MINIO_BUCKET")
        sys.exit(1)
    
    # Exécuter l'opération demandée
    try:
        if args.operation == "list":
            print(f"=== Liste des objets dans le bucket '{args.bucket}' ===")
            objects = list_objects_in_minio(
                endpoint=args.endpoint,
                access_key=args.access_key,
                secret_key=args.secret_key,
                bucket_name=args.bucket,
                prefix=args.prefix,
                secure=args.secure
            )
            if not objects:
                print("Aucun objet trouvé.")
                
        elif args.operation == "upload-folder":
            print(f"=== Upload du dossier '{args.local_path}' vers le bucket '{args.bucket}' ===")
            upload_folder_to_minio(
                endpoint=args.endpoint,
                access_key=args.access_key,
                secret_key=args.secret_key,
                bucket_name=args.bucket,
                local_folder_path=args.local_path,
                minio_folder_prefix=args.prefix,
                secure=args.secure
            )
            
        elif args.operation == "download-file":
            print(f"=== Téléchargement du fichier '{args.object_name}' ===")
            success = download_from_minio(
                endpoint=args.endpoint,
                access_key=args.access_key,
                secret_key=args.secret_key,
                bucket_name=args.bucket,
                object_name=args.object_name,
                local_file_path=args.local_path,
                secure=args.secure
            )
            if success:
                print("Téléchargement réussi!")
            else:
                print("Échec du téléchargement.")
                sys.exit(1)
                
        elif args.operation == "download-folder":
            print(f"=== Téléchargement du dossier avec préfixe '{args.prefix}' ===")
            count = download_folder_from_minio(
                endpoint=args.endpoint,
                access_key=args.access_key,
                secret_key=args.secret_key,
                bucket_name=args.bucket,
                minio_folder_prefix=args.prefix,
                local_folder_path=args.local_path,
                secure=args.secure
            )
            if count > 0:
                print(f"Téléchargement réussi! {count} fichiers téléchargés.")
            else:
                print("Aucun fichier téléchargé.")
                
    except KeyboardInterrupt:
        print("\nOpération interrompue par l'utilisateur.")
        sys.exit(1)
    except Exception as e:
        print(f"Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()