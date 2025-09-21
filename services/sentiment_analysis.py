import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, List, Optional
import joblib
import torchaudio
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import json
from collections import Counter
from services.topics_inf import TopicClassifier
from services.speech_segment import DarijaFrenchConverter


class SentimentAnalyzer:
    """Main sentiment analyzer that coordinates text, acoustic, and fusion analysis"""
    
    def __init__(self, config: dict):
        self.config = config
        if torch.cuda.is_available():
            gpu_index = config.get('gpu_index', 0)
            self.device = torch.device(f"cuda:{gpu_index}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        self.logger = logging.getLogger(__name__)
        self.load_models()
        self.topic_classifier = TopicClassifier(config, business_type=config.get('business_type', 'B2C'))
        self.converter = DarijaFrenchConverter(config)
        # Parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4))
        
        # Database manager (optional)
        self.db_manager = None
        # Map raw labels to display variants
        self.sentiment_display: Dict[str, str] = {
            "content": "Content",
            "mécontent": "Mécontent",
            "mecontent": "Mécontent",
            "tres mecontent": "Très Mécontent",
            "très mécontent": "Très Mécontent",
            "neutre": "Neutre",
            "aggressive": "Agressif",
            "agressif": "Agressif",
            "sec": "Sec",
            "courtois": "Courtois",
        }
    
    def load_models(self):
        """Load all sentiment analysis models"""
        self.logger.info("Loading sentiment analysis models...")
        
        try:
            self.acoustic_analyzer = AcousticSentimentAnalyzer(self.config)
            self.text_analyzer = TextSentimentAnalyzer(self.config)
            self.late_fusion_analyzer = LateFusionSentimentAnalyzer(self.config)
            self.late_fusion_analyzer.agent_id2label = self.acoustic_analyzer.agent_acoustic_id2label
            self.late_fusion_analyzer.client_id2label = self.acoustic_analyzer.client_acoustic_id2label
            self.logger.info("All sentiment models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading sentiment models: {e}")
            raise

    def analyze_batch_sentiment(self, chunks: List[Dict]) -> List[Dict]:
        """Analyze sentiment for a batch of chunks in parallel"""
        if not chunks:
            return chunks        
        # Use true batch processing for larger batches
        try:
            self.logger.debug(f"Using batch processing for {len(chunks)} chunks)")
            sentiment_results = self._analyze_batch(chunks)
            
            # Update chunks with results
            for i, chunk in enumerate(chunks):
                if i < len(sentiment_results):
                    chunk.update(sentiment_results[i])
                else:
                    # Fallback for any missing results
                    chunk.update(self._get_empty_sentiment_result("batch_error"))
                    
        except Exception as e:
            self.logger.error(f"Error in batch sentiment analysis: {e}")
            # Fallback to individual processing
            return self._fallback_individual_processing(chunks)
                
        # Save chunks to database if database manager is available
        if self.db_manager:
            self._save_chunks_to_database(chunks)
            # Also update call-level aggregated emotions (client/agent) after chunks are processed
            try:
                self._update_calls_aggregated_emotions(chunks)
            except Exception as e:
                self.logger.error(f"Failed to update call-level emotions: {e}")
        
        return chunks
    
    def _analyze_batch(self, chunks: List[Dict]) -> List[Dict]:
        """Optimized batch processing for sentiment analysis"""
        # Separate chunks by speaker type for batch processing
        agent_texts = []
        client_texts = []
        agent_waveforms = []
        client_waveforms = []
        chunk_indices = []
        
        # Collect all data for batch processing
        for i, chunk in enumerate(chunks):
            agent_text = chunk.get('agent_transcription', '')
            client_text = chunk.get('client_transcription', '')
            agent_waveform = chunk.get('agent_waveform')
            client_waveform = chunk.get('client_waveform')
            
            agent_texts.append(agent_text)
            client_texts.append(client_text)
            agent_waveforms.append(agent_waveform)
            client_waveforms.append(client_waveform)
            chunk_indices.append(i)
        
        # Batch process each modality
        agent_text_results = self.text_analyzer.analyze_batch_sentiment(agent_texts, 'agent')
        client_text_results = self.text_analyzer.analyze_batch_sentiment(client_texts, 'client')
        
        agent_acoustic_results = self.acoustic_analyzer.analyze_batch_sentiment(agent_waveforms, self.config.get('target_sample_rate', 16000), 'agent')
        client_acoustic_results = self.acoustic_analyzer.analyze_batch_sentiment(client_waveforms, self.config.get('target_sample_rate', 16000), 'client')
        
        # Combine results for each chunk
        results = []
        for i in range(len(chunks)):
            chunk_result = {}
            
            # Agent results
            agent_text = agent_text_results[i] if i < len(agent_text_results) else {}
            agent_acoustic = agent_acoustic_results[i] if i < len(agent_acoustic_results) else {}
            
            chunk_result.update({
                'agent_text_sentiment': agent_text.get('prediction', ''),
                'agent_text_confidence': agent_text.get('confidence', 0.0),
                'agent_text_probabilities': agent_text.get('probabilities', []),
                'agent_acoustic_sentiment': agent_acoustic.get('prediction', '') if agent_text.get('prediction', '') != '' else '',
                'agent_acoustic_confidence': agent_acoustic.get('confidence', 0.0) if agent_text.get('prediction', '') != '' else 0.0,
                'agent_acoustic_probabilities': agent_acoustic.get('probabilities', []) if agent_text.get('prediction', '') != '' else []
            })
            
            # Client results
            client_text = client_text_results[i] if i < len(client_text_results) else {}
            client_acoustic = client_acoustic_results[i] if i < len(client_acoustic_results) else {}
            
            chunk_result.update({
                'client_text_sentiment': client_text.get('prediction', ''),
                'client_text_confidence': client_text.get('confidence', 0.0),
                'client_text_probabilities': client_text.get('probabilities', []),
                'client_acoustic_sentiment': client_acoustic.get('prediction', '') if client_text.get('prediction', '') != '' else '',
                'client_acoustic_confidence': client_acoustic.get('confidence', 0.0) if client_text.get('prediction', '') != '' else 0.0,
                'client_acoustic_probabilities': client_acoustic.get('probabilities', []) if client_text.get('prediction', '') != '' else []
            })
            
            # Late fusion for both speakers
            agent_fusion = self.late_fusion_analyzer.analyze_sentiment(chunk_result, 'agent')
            client_fusion = self.late_fusion_analyzer.analyze_sentiment(chunk_result, 'client')
            
            chunk_result.update({
                'agent_fusion_sentiment': agent_fusion.get('prediction', ''),
                'agent_fusion_confidence': agent_fusion.get('confidence', 0.0),
                'client_fusion_sentiment': client_fusion.get('prediction', ''),
                'client_fusion_confidence': client_fusion.get('confidence', 0.0)
            })
            
            results.append(chunk_result)
        
        return results
    
    def _save_chunks_to_database(self, chunks: List[Dict]):
        """Save processed chunks to database (call info should already exist)"""
        if not self.db_manager:
            return
        
        try:
            from datetime import date
            
            for chunk in chunks:
                # Extract file information
                filename_raw = chunk.get('file_name', '')
                if not filename_raw:
                    self.logger.warning("Chunk missing file_name, skipping database save")
                    continue
                
                # Normalize to match call.id_enregistrement (base name without extension)
                # Strip extension first
                base_without_ext = filename_raw.rsplit('.', 1)[0]
                # Remove trailing _{chunk_idx} if present
                idx = chunk.get('chunk_idx', chunk.get('chunk_index'))
                if isinstance(idx, int):
                    suffix = f"_{idx}"
                    if base_without_ext.endswith(suffix):
                        base_without_ext = base_without_ext[: -len(suffix)]
                call_id = base_without_ext
                
                # Ensure call exists; if not, create a minimal record
                try:
                    existing = self.db_manager.get_call_by_id_enregistrement(call_id)
                except Exception as e:
                    existing = None
                    self.logger.warning(f"Call lookup failed for {call_id}: {e}")
                if not existing:
                    minimal_call = {
                        'id_enregistrement': call_id,
                        'file': '',
                        'duration_seconds': None,
                        'date_appel': date.today().isoformat(),
                        'partenaire': self.config.get('partenaire', 'INWI'),
                        'login_conseiller': self.config.get('login_conseiller', 'system'),
                        'topics': '',
                        'emotion_client_globale': '',
                        'ton_agent_global': ''
                    }
                    try:
                        self.db_manager.insert_call(minimal_call)
                        self.logger.info(f"Created minimal call record for {call_id}")
                    except Exception as e:
                        self.logger.error(f"Failed to create call for {call_id}: {e}")
                        continue
                
                # Create chunk record only (call should already exist)
                chunk_data = {
                    'id_chunk': f"{chunk.get('chunk_idx', 0)}",
                    'id_enregistrement': call_id,
                    'transcription_chunk': chunk.get('transcription_chunk', ''),
                    'transcription_agent': chunk.get('agent_transcription', ''),
                    'transcription_client': chunk.get('client_transcription', ''),
                    'emotion_client': chunk.get('client_fusion_sentiment', ''),
                    'ton_agent': chunk.get('agent_fusion_sentiment', '')
                }
                # TODO: Uncomment this when the converter is needed
                # chunk_data['transcription_chunk'] = self.converter.convert_text(chunk_data['transcription_chunk'])
                # chunk_data['transcription_agent'] = self.converter.convert_text(chunk_data['transcription_agent'])
                # chunk_data['transcription_client'] = self.converter.convert_text(chunk_data['transcription_client'])
                self.db_manager.insert_chunk(chunk_data)
            
            self.logger.info(f"Saved {len(chunks)} chunks to database")
            
        except Exception as e:
            self.logger.error(f"Failed to save chunks to database: {e}")
            # Don't raise the exception to avoid interrupting sentiment analysis
    
    def set_database_manager(self, db_manager):
        """Set the database manager for saving chunks"""
        self.db_manager = db_manager
        self.logger.info("Database manager set for sentiment analyzer")
    
    def _fallback_individual_processing(self, chunks: List[Dict]) -> List[Dict]:
        """Fallback to original individual processing if batch processing fails"""
        self.logger.warning("Falling back to individual chunk processing")
        
        # Process chunks in parallel (original method)
        futures = []
        for chunk in chunks:
            future = self.executor.submit(self._analyze_single_chunk, chunk)
            futures.append((chunk, future))
        
        # Collect results
        for chunk, future in futures:
            try:
                sentiment_result = future.result(timeout=120)
                chunk.update(sentiment_result)
            except Exception as e:
                self.logger.error(f"Error analyzing sentiment for {chunk.get('file_name', 'unknown')}: {e}")
                chunk.update(self._get_empty_sentiment_result(str(e)))
        
        return chunks
    
    def _get_empty_sentiment_result(self, error_msg: str = "") -> Dict:
        """Get empty sentiment result structure"""
        return {
            'agent_text_sentiment': 'error',
            'agent_text_confidence': 0.0,
            'agent_acoustic_sentiment': 'error',
            'agent_acoustic_confidence': 0.0,
            'agent_fusion_sentiment': 'error',
            'agent_fusion_confidence': 0.0,
            'client_text_sentiment': 'error',
            'client_text_confidence': 0.0,
            'client_acoustic_sentiment': 'error',
            'client_acoustic_confidence': 0.0,
            'client_fusion_sentiment': 'error',
            'client_fusion_confidence': 0.0,
            'sentiment_error': error_msg
        }
    
    def _analyze_single_chunk(self, chunk: Dict) -> Dict:
        """Analyze sentiment for a single chunk"""
        try:
            # Extract data from chunk
            agent_text = chunk.get('agent_transcription', '')
            client_text = chunk.get('client_transcription', '')
            agent_waveform = chunk.get('agent_waveform')
            client_waveform = chunk.get('client_waveform')
            
            # Analyze agent sentiment
            agent_results = self._analyze_speaker_sentiment(
                agent_text, agent_waveform, 'agent'
            )
            
            # Analyze client sentiment
            client_results = self._analyze_speaker_sentiment(
                client_text, client_waveform, 'client'
            )
            
            # Combine results
            sentiment_results = {**agent_results, **client_results}
            
            return sentiment_results
            
        except Exception as e:
            self.logger.error(f"Error in single chunk sentiment analysis: {e}")
            return {
                'agent_text_sentiment': '',
                'agent_text_confidence': 0.0,
                'agent_acoustic_sentiment': '',
                'agent_acoustic_confidence': 0.0,
                'agent_fusion_sentiment': '',
                'agent_fusion_confidence': 0.0,
                'client_text_sentiment': '',
                'client_text_confidence': 0.0,
                'client_acoustic_sentiment': '',
                'client_acoustic_confidence': 0.0,
                'client_fusion_sentiment': '',
                'client_fusion_confidence': 0.0,
                'sentiment_error': str(e)
            }
    
    def _analyze_speaker_sentiment(self, text: str, waveform: torch.Tensor, speaker: str) -> Dict:
        """Analyze sentiment for a single speaker (text + acoustic + fusion)"""
        results = {}
        
        try:
            # Text sentiment analysis
            if text and text.strip():
                text_result = self.text_analyzer.analyze_sentiment(text, speaker)
                results[f'{speaker}_text_sentiment'] = text_result.get('prediction', '')
                results[f'{speaker}_text_confidence'] = text_result.get('confidence', 0.0)
                results[f'{speaker}_text_probabilities'] = text_result.get('probabilities', [])
            else:
                results[f'{speaker}_text_sentiment'] = ''
                results[f'{speaker}_text_confidence'] = 0.0
            
            # Acoustic sentiment analysis
            if waveform is not None and waveform.numel() > 0:
                acoustic_result = self.acoustic_analyzer.analyze_sentiment(waveform, 16000, speaker)
                results[f'{speaker}_acoustic_sentiment'] = acoustic_result.get('prediction', '')
                results[f'{speaker}_acoustic_confidence'] = acoustic_result.get('confidence', 0.0)
                results[f'{speaker}_acoustic_probabilities'] = acoustic_result.get('probabilities', [])
            else:
                results[f'{speaker}_acoustic_sentiment'] = ''
                results[f'{speaker}_acoustic_confidence'] = 0.0
                results[f'{speaker}_acoustic_probabilities'] = []
            
            # Late fusion sentiment analysis
            if text and text.strip() and waveform is not None and waveform.numel() > 0:
                fusion_result = self.late_fusion_analyzer.analyze_sentiment(results, speaker)
                results[f'{speaker}_fusion_sentiment'] = fusion_result.get('prediction', '')
                results[f'{speaker}_fusion_confidence'] = fusion_result.get('confidence', 0.0)
                
            else:
                results[f'{speaker}_fusion_sentiment'] = ''
                results[f'{speaker}_fusion_confidence'] = 0.0
                
        except Exception as e:
            self.logger.error(f"Error analyzing {speaker} sentiment: {e}")
            results[f'{speaker}_text_sentiment'] = ''
            results[f'{speaker}_text_confidence'] = 0.0
            results[f'{speaker}_acoustic_sentiment'] = ''
            results[f'{speaker}_acoustic_confidence'] = 0.0
            results[f'{speaker}_fusion_sentiment'] = ''
            results[f'{speaker}_fusion_confidence'] = 0.0
        
        return results
   
    def _update_calls_aggregated_emotions(self, chunks: List[Dict]):
        """Aggregate per-chunk fusion sentiments into call-level emotions and update call rows."""
        if not self.db_manager or not chunks:
            return
        from collections import defaultdict
        # Group by call_filename
        per_call = defaultdict(list)
        for chunk in chunks:
            filename_raw = chunk.get('file_name', '')
            if not filename_raw:
                continue
            base_without_ext = filename_raw.rsplit('.', 1)[0]
            idx = chunk.get('chunk_idx', chunk.get('chunk_index'))
            if isinstance(idx, int):
                suffix = f"_{idx}"
                if base_without_ext.endswith(suffix):
                    base_without_ext = base_without_ext[: -len(suffix)]
            call_filename = base_without_ext
            per_call[call_filename].append(chunk)

        # Use the provided business rules helpers for final sentiments
        for call_filename, items in per_call.items():
            client_list = [it.get('client_fusion_sentiment', '') for it in items]
            agent_list = [it.get('agent_fusion_sentiment', '') for it in items]
            client_emotion = self.sentiment_appel_client(client_list)
            agent_ton = self.sentiment_appel_agent(agent_list)
            topics = 'Vide' #self.sentiment_appel_topics(items)
            if client_emotion or agent_ton:
                try:
                    self.db_manager.update_call_sentiment(call_filename, client_emotion, agent_ton, topics)
                    # self.logger.info(f"Updated call emotions for {call_filename}: client={client_emotion}, agent={agent_ton}")
                except Exception as e:
                    self.logger.error(f"Call sentiment update failed for {call_filename}: {e}")

    def sentiment_appel_topics(self, items: List[Dict]) -> str:
        """Determine overall topics from list of chunks"""
        transcription_call = ''
        for item in items:
            transcription_call += item.get('transcription_chunk', '')

        _, cat, typ_cat = self.topic_classifier.infer(transcription_call)
        return f'{cat} - {typ_cat}'
    
    def pretty_sentiment(self, label: Optional[str]) -> str:
        """Convert sentiment label to display format"""
        if label is None:
            return "Vide"
        key = str(label).strip().lower()
        return self.sentiment_display.get(key, str(label).capitalize())
                 
    def sentiment_appel_client(self, sentiments: List[str]) -> str:
        """
        Determine overall client sentiment from list of sentiments
        
        Args:
            sentiments: List of sentiment labels
            
        Returns:
            Overall sentiment
        """
        if not sentiments:
            return "Inconnu"

        sentiments = [self.pretty_sentiment(s.strip()) for s in sentiments if s and s.strip()]
        if not sentiments:
            return "Inconnu"
        
        count = Counter(sentiments)
        total = len(sentiments)

        # Priority 1: Last emotion is "Content"
        if sentiments[-1] == "Content":
            return "Content"

        # Priority 2: Presence of "Très Mécontent"
        if "Très Mécontent" in count: 
            return "Très Mécontent"

        # Priority 3: Presence of "Mécontent"
        if "Mécontent" in count:
            return "Mécontent"

        # Priority 4: ≥50% "Neutre" AND no negative emotions
        if count.get("Neutre", 0) / total >= 0.5:
            return "Neutre"

        # Priority 5: Most common emotion
        candidates = ["Content", "Mécontent", "Très Mécontent", "Neutre"]
        dominant = max(candidates, key=lambda x: count.get(x, 0))
        
        self.logger.info(f"Client overall sentiment: {dominant} from {dict(count)}")
        return dominant
    
    def sentiment_appel_agent(self, sentiments: List[str]) -> str:
        """
        Determine overall agent sentiment from list of sentiments
        
        Args:
            sentiments: List of sentiment labels
            
        Returns:
            Overall sentiment
        """
        if not sentiments:
            return "Inconnu"

        sentiments = [self.pretty_sentiment(s.strip()) for s in sentiments if s and s.strip()]
        if not sentiments:
            return "Inconnu"
        
        count = Counter(sentiments)
        total = len(sentiments)

        # Priority 1: ≥1 occurrence of "Agressif"
        if "Agressif" in count:
            return "Agressif"

        # Priority 2: Last tone is "Sec" OR ≥30% "Sec"
        if sentiments[-1] == "Sec" or count.get("Sec", 0) / total >= 0.3:
            return "Sec"

        # Priority 3: Last tone is "Courtois" AND ≥50% "Courtois"
        if sentiments[-1] == "Courtois" and count.get("Courtois", 0) / total >= 0.5:
            return "Courtois"

        # Priority 4: Last tone is "Neutre" AND no "Sec"
        if sentiments[-1] == "Neutre" and "Sec" not in count:
            return "Neutre"

        # Priority 5: Most common tone
        candidates = ["Agressif", "Sec", "Courtois", "Neutre"]
        dominant = max(candidates, key=lambda x: count.get(x, 0))
        
        self.logger.info(f"Agent overall sentiment: {dominant} from {dict(count)}")
        return dominant


class AcousticSentimentAnalyzer:
    """Analyzes sentiment from acoustic features"""
    
    def __init__(self, config: dict):
        self.config = config
        if torch.cuda.is_available():
            gpu_index = config.get('gpu_index', 0)
            self.device = torch.device(f"cuda:{gpu_index}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        self.logger = logging.getLogger(__name__)
        
        # Model paths
        self.client_acoustic_model_path = config.get('client_acoustic_model_path')
        self.client_acoustic_scaler_path = config.get('client_acoustic_scaler_path')

        self.agent_acoustic_model_path = config.get('agent_acoustic_model_path')
        self.agent_acoustic_scaler_path = config.get('agent_acoustic_scaler_path')
        
        # Only load models if paths are provided
        if (self.client_acoustic_model_path and self.client_acoustic_scaler_path and 
            self.agent_acoustic_model_path and self.agent_acoustic_scaler_path):
            try:
                self.load_models()
            except Exception as e:
                self.logger.error(f"Failed to load acoustic models: {e}")
                self._set_models_unavailable()
        else:
            self.logger.warning("Acoustic model paths not found in config - using fallback mode")
            self._set_models_unavailable()
    
    def load_models(self):
        """Load acoustic sentiment models"""
        try:
            self.logger.info(f"Loading acoustic model from {self.client_acoustic_model_path}")
            self.client_acoustic_model = joblib.load(self.client_acoustic_model_path)
            self.client_acoustic_scaler = joblib.load(self.client_acoustic_scaler_path)
            self.client_acoustic_classes = self.client_acoustic_model.classes_
            self.client_acoustic_id2label = {i: label for i, label in enumerate(self.client_acoustic_classes)}
            self.client_acoustic_label2id = {label: i for i, label in enumerate(self.client_acoustic_classes)}
            # print(f"Client acoustic classes: {self.client_acoustic_classes}")
            self.logger.info(f"Loading agent acoustic model from {self.agent_acoustic_model_path}")
            self.agent_acoustic_model = joblib.load(self.agent_acoustic_model_path)
            self.agent_acoustic_scaler = joblib.load(self.agent_acoustic_scaler_path)
            self.agent_acoustic_classes = self.agent_acoustic_model.classes_
            self.agent_acoustic_id2label = {i: label for i, label in enumerate(self.agent_acoustic_classes)}
            self.agent_acoustic_label2id = {label: i for i, label in enumerate(self.agent_acoustic_classes)}
            # print(f"Agent acoustic classes: {self.agent_acoustic_classes}")
            self.logger.info(f"Acoustic model loaded successfully")
            self.logger.info(f"Client Acoustic model classes: {self.client_acoustic_classes}")
            self.logger.info(f"Agent Acoustic model classes: {self.agent_acoustic_classes}")
            
            # Validate models
            if not hasattr(self.client_acoustic_model, 'predict'):
                self.logger.error("Client acoustic model does not have predict method")
            if not hasattr(self.agent_acoustic_model, 'predict'):
                self.logger.error("Agent acoustic model does not have predict method")
            
            # Test prediction with dummy data
            try:
                # Get expected feature count from scaler
                client_feature_count = self.client_acoustic_scaler.n_features_in_
                agent_feature_count = self.agent_acoustic_scaler.n_features_in_
                
                dummy_features_client = [0.0] * client_feature_count
                dummy_features_agent = [0.0] * agent_feature_count
                
                client_pred = self.client_acoustic_model.predict([dummy_features_client])
                agent_pred = self.agent_acoustic_model.predict([dummy_features_agent])
                self.logger.info(f"Model validation successful - client: {client_pred[0]} ({client_feature_count} features), agent: {agent_pred[0]} ({agent_feature_count} features)")
            except Exception as e:
                self.logger.error(f"Model validation failed: {e}")
            
            # Mark models as available
            self.models_available = True

        except Exception as e:
            self.logger.error(f"Error loading acoustic models: {e}")
            raise
    
    def _set_models_unavailable(self):
        """Set models as unavailable and initialize fallback attributes"""
        self.client_acoustic_model = None
        self.client_acoustic_scaler = None
        self.agent_acoustic_model = None
        self.agent_acoustic_scaler = None
        self.models_available = False

    def analyze_sentiment(self, waveform: torch.Tensor, sample_rate: int, speaker: str) -> Dict[str, Any]:
        """Extract acoustic features and predict sentiment"""
        try:
            # Check if models are available
            if not getattr(self, 'models_available', False):
                return {'prediction': '', 'confidence': 0.0, 'probabilities': []}
            
            # Extract features
            features = self._extract_acoustic_features(waveform, sample_rate)
            if features is None:
                return {'prediction': '', 'confidence': 0.0, 'probabilities': []}
            
            if speaker == 'client':
                acoustic_model = self.client_acoustic_model
                acoustic_scaler = self.client_acoustic_scaler
            else:
                acoustic_model = self.agent_acoustic_model
                acoustic_scaler = self.agent_acoustic_scaler
                
            # Double check models exist
            if acoustic_model is None or acoustic_scaler is None:
                return {'prediction': '', 'confidence': 0.0, 'probabilities': []}

            # Debug features before scaling
            self.logger.debug(f"Raw features for {speaker}: {features}")
            
            # Get the expected number of features from the scaler
            expected_features = acoustic_scaler.n_features_in_
            self.logger.debug(f"Scaler expects {expected_features} features")
            
            if len(features) != expected_features:
                self.logger.warning(f"Expected {expected_features} features, got {len(features)}")
                # Pad or truncate features to match expected size
                feature_list = list(features.values())
                if len(feature_list) < expected_features:
                    feature_list.extend([0.0] * (expected_features - len(feature_list)))
                else:
                    feature_list = feature_list[:expected_features]
            else:
                feature_list = list(features.values())
            
            # Scale features
            features_scaled = acoustic_scaler.transform([feature_list])
            
            # Debug scaled features
            self.logger.debug(f"Scaled features for {speaker}: {features_scaled[0]}")
            self.logger.debug(f"Scaled features shape: {features_scaled.shape}")
            
            # Check if scaled features are all zeros or NaN
            if np.all(features_scaled == 0) or np.any(np.isnan(features_scaled)):
                self.logger.warning(f"Scaled features are all zeros or contain NaN for {speaker}")
                return {'prediction': '', 'confidence': 0.0}
            
            # Predict sentiment
            prediction = acoustic_model.predict(features_scaled)[0]
            probabilities = acoustic_model.predict_proba(features_scaled)
            confidence = probabilities.max()
            if speaker == 'client':
                prediction_label = self.client_acoustic_id2label.get(prediction, 'unknown')
            else:
                prediction_label = self.agent_acoustic_id2label.get(prediction, 'unknown')
            
            # Debug prediction results
            self.logger.debug(f"Acoustic prediction for {speaker}: {probabilities}, confidence: {confidence:.4f}")
            
            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'features': features,
                'probabilities': probabilities[0]
            }
                
        except Exception as e:
            self.logger.error(f"Acoustic sentiment analysis error: {e}")
            return {'prediction': '', 'confidence': 0.0}

    def analyze_batch_sentiment(self, waveforms: List[torch.Tensor], sample_rate: int, speaker: str) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of waveforms efficiently"""
        try:
            if not waveforms:
                return []
            
            # Check if models are available
            if not getattr(self, 'models_available', False):
                return [{'prediction': '', 'confidence': 0.0, 'probabilities': []} for _ in waveforms]
            
            # Filter out None/empty waveforms and track indices
            valid_waveforms = []
            valid_indices = []
            for i, waveform in enumerate(waveforms):
                if waveform is not None and waveform.numel() > 0:
                    valid_waveforms.append(waveform)
                    valid_indices.append(i)
            
            # If no valid waveforms, return empty results for all
            if not valid_waveforms:
                return [{'prediction': '', 'confidence': 0.0, 'probabilities': []} for _ in waveforms]
            
            # Select model and scaler based on speaker
            if speaker == 'client':
                acoustic_model = self.client_acoustic_model
                acoustic_scaler = self.client_acoustic_scaler
                id2label = self.client_acoustic_id2label
            else:
                acoustic_model = self.agent_acoustic_model
                acoustic_scaler = self.agent_acoustic_scaler
                id2label = self.agent_acoustic_id2label
                
            # Double check models exist
            if acoustic_model is None or acoustic_scaler is None:
                return [{'prediction': '', 'confidence': 0.0, 'probabilities': []} for _ in waveforms]
            
            # Extract features for all valid waveforms
            feature_lists = []
            expected_features = acoustic_scaler.n_features_in_
            
            for waveform in valid_waveforms:
                features = self._extract_acoustic_features(waveform, sample_rate)
                if features is None:
                    # Use zero features if extraction fails
                    features = self._get_zero_features()
                
                # Ensure correct number of features
                feature_list = list(features.values())
                if len(feature_list) != expected_features:
                    if len(feature_list) < expected_features:
                        feature_list.extend([0.0] * (expected_features - len(feature_list)))
                    else:
                        feature_list = feature_list[:expected_features]
                
                feature_lists.append(feature_list)
            
            # Batch scale features
            if feature_lists:
                features_scaled_batch = acoustic_scaler.transform(feature_lists)
                
                # Check for invalid features
                valid_feature_mask = []
                for i, features_scaled in enumerate(features_scaled_batch):
                    is_valid = not (np.all(features_scaled == 0) or np.any(np.isnan(features_scaled)))
                    valid_feature_mask.append(is_valid)
                
                # Batch predict sentiment
                predictions = acoustic_model.predict(features_scaled_batch)
                probabilities_batch = acoustic_model.predict_proba(features_scaled_batch)
                
                # Process results
                results = []
                for i in range(len(valid_waveforms)):
                    if valid_feature_mask[i]:
                        prediction = predictions[i]
                        probabilities = probabilities_batch[i]
                        confidence = probabilities.max()
                        
                        results.append({
                            'prediction': prediction,
                            'confidence': float(confidence),
                            'probabilities': probabilities.tolist()
                        })
                    else:
                        results.append({
                            'prediction': '',
                            'confidence': 0.0,
                            'probabilities': []
                        })
            else:
                results = []
            
            # Map results back to original waveform positions
            final_results = []
            result_idx = 0
            for i in range(len(waveforms)):
                if i in valid_indices:
                    if result_idx < len(results):
                        final_results.append(results[result_idx])
                    else:
                        final_results.append({'prediction': '', 'confidence': 0.0, 'probabilities': []})
                    result_idx += 1
                else:
                    final_results.append({'prediction': '', 'confidence': 0.0, 'probabilities': []})
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Batch acoustic sentiment analysis error: {e}")
            return [{'prediction': '', 'confidence': 0.0, 'probabilities': []} for _ in waveforms]

    def _extract_acoustic_features(self, waveform: torch.Tensor, sample_rate: int) -> Optional[Dict[str, float]]:
        """Extract acoustic features from waveform"""
        try:
            # Ensure waveform is on device and flatten if needed
            waveform = waveform.to(self.device)
            
            # Use the waveform directly (should be mono at this point)
            y = waveform.squeeze(0)
            sr = sample_rate

            # Frame parameters
            frame_length = int(0.025 * sr)  # 25 ms
            hop_length   = int(0.010 * sr)  # 10 ms
            if y.numel() < frame_length:
                raise ValueError('Audio too short')

            frames = y.unfold(0, frame_length, hop_length)

            # RMS
            rms       = torch.sqrt(torch.mean(frames ** 2, dim=1))
            rms_mean  = rms.mean();  rms_std = rms.std();  rms_rng = rms.max() - rms.min()

            # ZCR
            signs      = torch.sign(frames)
            zc         = ((signs[:, :-1] * signs[:, 1:]) < 0).sum(dim=1).float() / frame_length
            zcr_mean   = zc.mean();  zcr_std = zc.std()

            # STFT
            n_fft  = 512
            window = torch.hann_window(frame_length).to(y.device)
            stft   = torch.stft(y, n_fft=n_fft, hop_length=hop_length,
                                win_length=frame_length, window=window,
                                return_complex=True)
            mag    = stft.abs()  # (freq_bins, time)

            freqs  = torch.linspace(0, sr / 2, mag.shape[0], device=self.device)
            energy = mag.sum(dim=0) + 1e-8

            # Spectral centroid & bandwidth
            centroid = (mag * freqs.unsqueeze(1)).sum(dim=0) / energy
            sc_mean, sc_std = centroid.mean(), centroid.std()

            diff_sq   = (freqs.unsqueeze(1) - centroid.unsqueeze(0)) ** 2
            bandwidth = torch.sqrt((mag * diff_sq).sum(dim=0) / energy)
            sb_mean, sb_std = bandwidth.mean(), bandwidth.std()

            # Rolloff 0.85
            cum_energy = mag.cumsum(dim=0)
            thresh     = 0.85 * (cum_energy[-1] + 1e-8)
            roll_idx   = ((cum_energy >= thresh).float().argmax(dim=0)).long()
            roll_freqs = freqs[roll_idx]
            sr_mean, sr_std = roll_freqs.mean(), roll_freqs.std()

            # MFCC (13) – GPU
            mfcc_tf = torchaudio.transforms.MFCC(
                sample_rate=sr, n_mfcc=13,
                melkwargs={'n_fft': n_fft, 'hop_length': hop_length, 'win_length': frame_length}
            ).to(self.device)
            mfcc      = mfcc_tf(y.unsqueeze(0)).squeeze(0)  # (13, time)
            mfcc_mean = mfcc.mean(dim=1);  mfcc_std = mfcc.std(dim=1)

            # Rough tempo via spectral flux autocorrelation
            flux      = torch.relu(mag[:, 1:] - mag[:, :-1]).sum(dim=0)
            onset_env = flux.unsqueeze(0).unsqueeze(0)  # (1,1,T)
            autocorr  = torch.nn.functional.conv1d(onset_env, onset_env, padding=onset_env.shape[-1] - 1).squeeze()
            autocorr[0] = 0
            max_lag    = autocorr.argmax()
            period     = max_lag.item() * hop_length / sr if max_lag > 0 else 0.0
            tempo      = 60.0 / period if period > 0 else 0.0

            feats = {
                'rms_mean': rms_mean.item(),      'rms_std': rms_std.item(),      'rms_range': rms_rng.item(),
                'zcr_mean': zcr_mean.item(),      'zcr_std': zcr_std.item(),
                'spectral_centroid_mean': sc_mean.item(),  'spectral_centroid_std': sc_std.item(),
                'spectral_bandwidth_mean': sb_mean.item(), 'spectral_bandwidth_std': sb_std.item(),
                'spectral_rolloff_mean': sr_mean.item(),   'spectral_rolloff_std': sr_std.item(),
                'tempo': tempo,
            }
            for i in range(13):
                feats[f'mfcc_{i}_mean'] = mfcc_mean[i].item()
                feats[f'mfcc_{i}_std']  = mfcc_std[i].item()

            return feats
        except Exception as exc:
            print(f"[Feature Extraction Error]: {exc}")
            import traceback
            traceback.print_exc()
            return None

    def _get_zero_features(self) -> Dict[str, float]:
        """Return zero features when audio is too short or processing fails"""
        feats = {
            'rms_mean': 0.0, 'rms_std': 0.0, 'rms_range': 0.0,
            'zcr_mean': 0.0, 'zcr_std': 0.0,
            'spectral_centroid_mean': 0.0, 'spectral_centroid_std': 0.0,
            'spectral_bandwidth_mean': 0.0, 'spectral_bandwidth_std': 0.0,
            'spectral_rolloff_mean': 0.0, 'spectral_rolloff_std': 0.0,
            'tempo': 0.0,
        }
        # Add zero MFCC features
        for i in range(13):
            feats[f'mfcc_{i}_mean'] = 0.0
            feats[f'mfcc_{i}_std'] = 0.0
        
        # Debug zero features count
        self.logger.debug(f"Zero features count: {len(feats)}")
        return feats


class TextSentimentAnalyzer:
    """Analyzes sentiment from text using pre-trained models"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('gpu_index', 0)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.device}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        self.logger = logging.getLogger(__name__)
        
        # Model paths
        self.client_text_model_path = config.get('client_text_model_path')
        self.agent_text_model_path = config.get('agent_text_model_path')

        # Only load models if paths are provided
        if self.client_text_model_path and self.agent_text_model_path:
            try:
                self.load_models()
            except Exception as e:
                self.logger.error(f"Failed to load text models: {e}")
                self._set_models_unavailable()
        else:
            self.logger.warning("Text model paths not found in config - using fallback mode")
            self._set_models_unavailable()

    def load_models(self):
        """Load text sentiment models"""
        try:
            self.logger.info(f"Loading text model from {self.client_text_model_path}")
            self.client_text_model = AutoModelForSequenceClassification.from_pretrained(self.client_text_model_path)
            try:
                with open(f"{self.client_text_model_path}/config.json", "r") as f:
                    config = json.load(f)
                    base_model_name = config.get("_name_or_path", "SI2M-Lab/DarijaBERT")
                    self.client_text_model.config.id2label = config.get("id2label", {})
                    self.client_text_model.config.label2id = config.get("label2id", {})
                    self.client_text_model.config.id2label = {int(k): v for k, v in self.client_text_model.config.id2label.items()}
                    self.client_text_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                    self.logger.info(f"Client text model config loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading text model config client: {e}")
                # Fallback: try to load tokenizer with default model name
                try:
                    self.client_text_tokenizer = AutoTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")
                    self.logger.info("Using fallback tokenizer for client text model")
                except Exception as fallback_e:
                    self.logger.error(f"Failed to load fallback tokenizer: {fallback_e}")
                    raise

            self.client_text_model.to(self.device)
            self.client_text_model.eval()
            self.logger.info(f"Client text model loaded successfully")
            self.logger.info(f"Client text model classes: {self.client_text_model.config.id2label}")

            self.agent_text_model = AutoModelForSequenceClassification.from_pretrained(self.agent_text_model_path)
            try:
                with open(f"{self.agent_text_model_path}/config.json", "r") as f:
                    config = json.load(f)
                    base_model_name = config.get("_name_or_path", "SI2M-Lab/DarijaBERT")
                    self.agent_text_model.config.id2label = config.get("id2label", {})
                    self.agent_text_model.config.label2id = config.get("label2id", {})
                    self.agent_text_model.config.id2label = {int(k): v for k, v in self.agent_text_model.config.id2label.items()}
                    self.agent_text_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                    self.logger.info(f"Agent text model config loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading text model config agent: {e}")
                # Fallback: try to load tokenizer with default model name
                try:
                    self.agent_text_tokenizer = AutoTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")
                    self.logger.info("Using fallback tokenizer for agent text model")
                except Exception as fallback_e:
                    self.logger.error(f"Failed to load fallback tokenizer: {fallback_e}")
                    raise

            self.agent_text_model.to(self.device)
            self.agent_text_model.eval()
            self.logger.info(f"Agent text model loaded successfully")
            self.logger.info(f"Agent text model classes: {self.agent_text_model.config.id2label}")
            
            # Mark models as available
            self.models_available = True
            
        except Exception as e:
            self.logger.error(f"Error loading text models: {e}")
            raise
    
    def _set_models_unavailable(self):
        """Set models as unavailable and initialize fallback attributes"""
        self.client_text_model = None
        self.client_text_tokenizer = None
        self.agent_text_model = None
        self.agent_text_tokenizer = None
        self.models_available = False

    def analyze_sentiment(self, text: str, speaker: str) -> Dict[str, Any]:
        """Analyze sentiment from text"""
        try:
            if not text or not text.strip():
                return {'prediction': '', 'confidence': 0.0}
            
            # Check if models are available
            if not getattr(self, 'models_available', False):
                return {'prediction': '', 'confidence': 0.0}
            
            if speaker == 'client':
                text_tokenizer = self.client_text_tokenizer
                text_model = self.client_text_model
            else:
                text_tokenizer = self.agent_text_tokenizer
                text_model = self.agent_text_model
            
            # Double check models exist
            if text_tokenizer is None or text_model is None:
                return {'prediction': '', 'confidence': 0.0}

            # Tokenize text
            inputs = text_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Predict sentiment (AMP on CUDA)
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = text_model(**inputs)
                else:
                    outputs = text_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities.max().item()
                
                # Get prediction label
                prediction = text_model.config.id2label.get(prediction_idx, 'unknown')
            
            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'probabilities': probabilities.cpu().numpy().tolist()[0]
            }
            
        except Exception as e:
            self.logger.error(f"Text sentiment analysis error: {e}")
            return {'prediction': '', 'confidence': 0.0}

    def analyze_batch_sentiment(self, texts: List[str], speaker: str) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of texts"""
        try:
            if not texts:
                return []
            
            # Check if models are available
            if not getattr(self, 'models_available', False):
                return [{'prediction': '', 'confidence': 0.0, 'probabilities': []} for _ in texts]
            
            # Filter out empty texts and keep track of original indices
            valid_texts = []
            valid_indices = []
            for i, text in enumerate(texts):
                if text and text.strip() and len(text.strip()) >= 5:
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            # If no valid texts, return empty results for all
            if not valid_texts:
                return [{'prediction': '', 'confidence': 0.0, 'probabilities': []} for _ in texts]
            
            # Select model and tokenizer based on speaker
            if speaker == 'client':
                text_tokenizer = self.client_text_tokenizer
                text_model = self.client_text_model
            else:
                text_tokenizer = self.agent_text_tokenizer
                text_model = self.agent_text_model
            
            # Double check models exist
            if text_tokenizer is None or text_model is None:
                return [{'prediction': '', 'confidence': 0.0, 'probabilities': []} for _ in texts]

            # Batch tokenize all valid texts
            inputs = text_tokenizer(
                valid_texts, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Batch predict sentiment (AMP on CUDA)
            results = []
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = text_model(**inputs)
                else:
                    outputs = text_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction_indices = torch.argmax(probabilities, dim=1)
                confidences = probabilities.max(dim=1).values
                
                # Process each result
                for i in range(len(valid_texts)):
                    prediction_idx = prediction_indices[i].item()
                    confidence = confidences[i].item()
                    prediction = text_model.config.id2label.get(prediction_idx, 'unknown')
                    
                    results.append({
                        'prediction': prediction,
                        'confidence': float(confidence),
                        'probabilities': probabilities[i].cpu().numpy().tolist()
                    })
            
            # Map results back to original text positions
            final_results = []
            result_idx = 0
            for i in range(len(texts)):
                if i in valid_indices:
                    final_results.append(results[result_idx])
                    result_idx += 1
                else:
                    final_results.append({'prediction': '', 'confidence': 0.0, 'probabilities': []})
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Batch text sentiment analysis error: {e}")
            return [{'prediction': '', 'confidence': 0.0, 'probabilities': []} for _ in texts]


class LateFusionSentimentAnalyzer:
    """Combines text and acoustic sentiment using late fusion"""
    
    def __init__(self, config: dict):
        self.config = config
        if torch.cuda.is_available():
            gpu_index = config.get('gpu_index', 0)
            self.device = torch.device(f"cuda:{gpu_index}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        self.logger = logging.getLogger(__name__)
        self.load_models()
        

    def load_models(self):
        """Load fusion model (if available)"""
        # For now, we'll use a simple weighted average
        # In the future, this could load a trained fusion model
        self.client_text_weight = 0.42
        self.client_acoustic_weight = 0.58
        self.agent_text_weight = 0.54
        self.agent_acoustic_weight = 0.46
        self.agent_id2label = None
        self.client_id2label = None
        self.logger.info("Late fusion analyzer initialized with weighted average")

    def analyze_sentiment(self, results: Dict[str, Any], speaker: str) -> Dict[str, Any]:
        """Combine text and acoustic sentiment using late fusion"""
        try:
            # Get text and acoustic sentiment from results
            text_sentiment = results.get(f'{speaker}_text_sentiment', '')
            acoustic_sentiment = results.get(f'{speaker}_acoustic_sentiment', '')
            text_confidence = results.get(f'{speaker}_text_confidence', 0.0)
            acoustic_confidence = results.get(f'{speaker}_acoustic_confidence', 0.0)
            text_probabilities = results.get(f'{speaker}_text_probabilities', [])
            acoustic_probabilities = results.get(f'{speaker}_acoustic_probabilities', [])
            
            # Simple weighted fusion (can be replaced with more sophisticated fusion)
            # print(f'text_sentiment: {text_probabilities} , acoustic_sentiment: {acoustic_probabilities}')
            
            if text_sentiment and acoustic_sentiment and len(text_probabilities) > 0 and len(acoustic_probabilities) > 0:
                # Convert lists to numpy arrays for mathematical operations
                text_probs = np.array(text_probabilities)
                acoustic_probs = np.array(acoustic_probabilities)
                
                # Combine confidences using weighted average
                if speaker == 'client': 
                    fusion_confidence = self.client_text_weight * text_probs + self.client_acoustic_weight * acoustic_probs
                else:
                    fusion_confidence = self.agent_text_weight * text_probs + self.agent_acoustic_weight * acoustic_probs

                # print(f"fusion_confidence: {fusion_confidence} , text_confidence: {text_confidence} , acoustic_confidence: {acoustic_confidence}")

                fusion_prediction_idx = int(np.argmax(fusion_confidence))
                fusion_confidence_max = float(np.max(fusion_confidence))

                if speaker == 'client':
                    fusion_prediction = self.client_id2label.get(fusion_prediction_idx, 'unknown')
                else:
                    fusion_prediction = self.agent_id2label.get(fusion_prediction_idx, 'unknown')
                    
                # self.logger.info(f"Late fusion prediction for {speaker}: {fusion_prediction}, confidence: {fusion_confidence_max:.4f}")
                return {
                    'prediction': fusion_prediction,
                    'confidence': fusion_confidence_max,
                    'probabilities': fusion_confidence.tolist()
                }
            else:
                # Fallback to whichever is available
                if text_sentiment:
                    return {
                        'prediction': text_sentiment,
                        'confidence': text_confidence,
                        'probabilities': text_probabilities
                    }
                elif acoustic_sentiment:
                    return {
                        'prediction': acoustic_sentiment,
                        'confidence': acoustic_confidence,
                        'probabilities': acoustic_probabilities
                    }
                else:
                    return {
                        'prediction': '',
                        'confidence': 0.0,
                        'probabilities': []
                    }
                    
        except Exception as e:
            self.logger.error(f"Late fusion sentiment analysis error: {e}")
            return {
                'prediction': '',
                'confidence': 0.0,
                'probabilities': []
            }

   