"""
Zero-shot Constrained Topic Classification with Pre-summarization
================================================================

This module provides topic classification for customer support transcriptions
using AWS Bedrock Claude models. It first summarizes the transcription in French,
then classifies it into predefined categories.
"""

import os
import json
from typing import Tuple
from pathlib import Path
import pandas as pd
import boto3
from tenacity import retry, stop_after_attempt, wait_exponential
import re
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)



class TopicClassifier:
    """Handles topic classification using AWS Bedrock"""
    
    def __init__(self,config: dict, business_type="B2C"):
        """Initialize topic classifier with AWS configuration"""
        # Check AWS configuration
        if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
            logger.warning("AWS credentials not configured - topic classification disabled")
            self.enabled = False
            return
            
        self.config = config
        self.enabled = True
        
        # Initialize Bedrock client
        self.bedrock = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_DEFAULT_REGION"),
            aws_access_key_id= os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        
        # Load topic catalogue
        self._load_topics(business_type)
        
        # Create prompts
        self._create_prompts()
        
        logger.info("Topic classifier initialized successfully")
    
    def _load_topics(self, business_type="B2C"):
        """Load topic catalogue from Excel file"""
        try:
            if business_type == "B2C":
                topics_path = Path(self.config.get('topics_glossary_b2c')) 
            elif business_type == "B2B":
                topics_path = Path(self.config.get('topics_glossary_b2b'))
            else:
                raise ValueError(f"Invalid business type: {business_type}")
            
            if not topics_path.exists():
                logger.error(f"Topics glossary not found: {topics_path}")
                self.enabled = False
                return
                
            df_topics = pd.read_excel(topics_path, sheet_name="explication").fillna("")
            
            # Create topic strings
            df_topics["topic_str"] = (
                df_topics["Catégorie"].str.strip()
                + " – "
                + df_topics["Type de spécialité"].str.strip()
                + " : "
                + df_topics["Explication"].str.strip()
            )
            
            # Create numbered topic lines
            self.topic_lines = [f"{i+1}. {t}" for i, t in enumerate(df_topics["topic_str"])]
            self.topic_lookup = dict(enumerate(df_topics["topic_str"], start=1))
            self.df_topics = df_topics
            
            logger.info(f"Loaded {len(self.topic_lines)} topic categories")
            
        except Exception as e:
            logger.error(f"Error loading topics: {e}")
            self.enabled = False
    
    def _create_prompts(self):
        """Create prompts for summarization and classification"""
        self.summary_prompt = (
            "Tu es un expert du service client télécom.\n"
            "Lis la transcription (en darija) et rédige un **résumé en français** "
            "de 120 mots maximum incluant :\n"
            "• la raison de l'appel\n"
            "• les actions demandées ou proposées\n"
            "• les offres/services mentionnés\n\n"
            "Transcription :\n{transcript}\n\nRésumé :"
        )
        
        self.classification_prompt_header = (
            "Tu es un analyste expert du service client télécom.\n"
            "Voici la liste complète des sujets possibles, chacun identifié par un numéro :\n"
            + "\n".join(self.topic_lines)
            + "\n\nD'après le texte ci-dessous, réponds STRICTEMENT par le numéro du sujet "
            + "le plus pertinent (un seul numéro, aucun autre texte).\n"
        )
    
    @retry(wait=wait_exponential(min=0, max=1), stop=stop_after_attempt(5), reraise=True)
    def _invoke_model(self, model_id: str, body: dict) -> dict:
        """Invoke Bedrock model with retry logic"""
        response = self.bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(body).encode("utf-8"),
            accept="application/json",
            contentType="application/json",
        )
        return json.loads(response["body"].read())
    
    def summarize(self, transcript: str) -> str:
        """
        Summarize Darija transcript in French
        
        Args:
            transcript: Darija transcription text
            
        Returns:
            French summary
        """
        if not self.enabled:
            return "Service de résumé non disponible"
            
        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config.get('max_tokens_summary'),
                "temperature": self.config.get('temperature'),
                "messages": [{
                    "role": "user",
                    "content": self.summary_prompt.format(transcript=transcript)
                }],
            }
            
            data = self._invoke_model(self.config.get('summary_model_id'), body)
            summary = data["content"][0]["text"].strip()
            
            logger.info(f"Generated summary: {len(summary)} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return "Erreur lors du résumé"
    
    def classify(self, summary_text: str) -> str:
        """
        Classify summary into topic category
        
        Args:
            summary_text: French summary text
            
        Returns:
            Topic index number as string
        """
        if not self.enabled:
            return "0"
            
        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config.get('max_tokens_class'),
                "temperature": self.config.get('temperature'),
                "messages": [{  
                    "role": "user",
                    "content": f"{self.classification_prompt_header}\nTexte :\n{summary_text}\nNuméro :"
                }],
            }
            
            data = self._invoke_model(self.config.get('class_model_id'), body)
            index = data["content"][0]["text"].strip()
            
            logger.info(f"Classified as topic index: {index}")
            return index
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "0"
    
    def map_index_to_category(self, idx_str: str) -> Tuple[str, str]:
        """
        Map topic index to category and type
        
        Args:
            idx_str: Topic index as string
            
        Returns:
            Tuple of (category, type)
        """
        try:
            idx = int(idx_str)
            if 1 <= idx <= len(self.df_topics):
                row = self.df_topics.iloc[idx - 1]
                return row["Catégorie"], row["Type de spécialité"]
            else:
                logger.warning(f"Invalid topic index: {idx}")
                return "UNKNOWN", "UNKNOWN"
        except (ValueError, IndexError) as e:
            logger.error(f"Error mapping index {idx_str}: {e}")
            return "UNKNOWN", "UNKNOWN"
    
    def clean_summary(self, summary: str) -> str:
        """
        Clean summary text by removing common prefixes
        
        Args:
            summary: Raw summary text
            
        Returns:
            Cleaned summary
        """
        # Remove common prefixes
        summary = re.sub(r"^[^:]*:\s*", "", summary, count=1)
        summary = re.sub(
            r"(?i)^résumé\s+en\s+français\s*\([^)]+\)\s*:\s*",
            "",
            summary,
            count=1
        )
        return summary.strip()
    
    def infer(self, transcription: str) -> Tuple[str, str, str]:
        """
        Complete inference pipeline: summarize and classify
        
        Args:
            transcription: Darija transcription
            
        Returns:
            Tuple of (summary, category, type)
        """
        if not self.enabled:
            logger.warning("Topic classifier not enabled")
            return "Service non disponible", "Appel blanc", "Non classifié"
        
        try:
            # Summarize
            summary = self.summarize(transcription)
            summary_cleaned = self.clean_summary(summary)
            
            # Classify
            idx = self.classify(summary)
            category, type_specialty = self.map_index_to_category(idx)
            
            logger.info(f"Inference complete: {category} - {type_specialty}")
            return summary_cleaned, category, type_specialty
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return "Erreur", "Appel blanc", "Erreur"


