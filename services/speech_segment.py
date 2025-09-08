import os
from typing import Any
from pyannote.audio import Pipeline
import logging
import torch
import torchaudio
import tempfile
from typing import List, Dict, Any, Tuple
import numpy as np
from utils.utils import remove_special_characters
import pandas as pd
from rapidfuzz import process, fuzz
import re
from typing import Optional

logger = logging.getLogger(__name__)


class DarijaFrenchConverter:
    """Handles Darija to French text conversion and number replacement"""
    
    def __init__(self, config: dict):
        self.config = config
        self.mapping: Dict[str, str] = {}
        self.sorted_keys: List[str] = []
        self.word_to_number: Dict[str, int] = {}
        self.reference_words: List[str] = []
        self._load_dictionary()
        self._load_number_dictionary()
        
    def _load_dictionary(self):
        """Load Darija to French conversion dictionary"""
        try:
            df = pd.read_excel(self.config.get('darija_french_dict'))
            
            # First column contains French words
            french_words = df.iloc[:, 0]
            
            # Other columns contain Darija variants
            for col in df.columns[1:]:
                for french, darija in zip(french_words, df[col]):
                    if pd.notna(darija):
                        self.mapping[darija.strip()] = french.strip()
                        
            # Sort keys by length (longest first) for better matching
            self.sorted_keys = sorted(self.mapping.keys(), key=len, reverse=True)
            
            logger.info(f"Loaded Darija dictionary with {len(self.mapping)} entries")
            
        except Exception as e:
            logger.error(f"Error loading Darija dictionary: {e}")
    
    def _load_number_dictionary(self):
        """Load Darija number conversion dictionary"""
        try:
            df = pd.read_excel(self.config.get('darija_numbers_dict'))
            
            # Build mapping: darija expression → number
            for _, row in df.iterrows():
                number = row["Nombre"]
                for word in row[1:].dropna():
                    word = str(word).strip()
                    if word:
                        self.word_to_number[word] = number
            
            self.reference_words = list(self.word_to_number.keys())
            logger.info(f"Loaded Darija number dictionary with {len(self.word_to_number)} entries")
            
        except Exception as e:
            logger.error(f"Error loading Darija number dictionary: {e}")
    
    def _fuzzy_map_darija_number(self, chunk: str, threshold: int = 90) -> Tuple[Optional[int], Optional[str], float]:
        """
        Fuzzy match a chunk to a Darija number with improved matching
        
        Args:
            chunk: Text chunk to match
            threshold: Minimum similarity score
            
        Returns:
            Tuple of (number, matched_word, score)
        """
        if not self.reference_words:
            return None, None, 0.0
            
        match, score, _ = process.extractOne(chunk, self.reference_words, scorer=fuzz.ratio)
        if score >= threshold:
            return self.word_to_number[match], match, score
        return None, None, score
    
    def _replace_numbers_in_sentence(self, sentence: str, base_threshold: int = 90, max_ngram: int = 5) -> str:
        """
        Replace Darija number expressions with numeric values using adaptive threshold
        
        Args:
            sentence: Input sentence
            base_threshold: Base minimum similarity score for fuzzy matching
            max_ngram: Maximum n-gram size to consider
            
        Returns:
            Sentence with numbers replaced
        """
        if not self.word_to_number:
            return sentence
            
        words = sentence.strip().split()
        replaced = [None] * len(words)
        used_positions = set()

        for n in range(max_ngram, 0, -1):  # From longest to shortest
            for i in range(len(words) - n + 1):
                positions = set(range(i, i + n))
                if positions & used_positions:
                    continue

                chunk = " ".join(words[i:i + n])

                # Adapt threshold to tolerate differences on long chunks
                threshold = base_threshold - (max(n - 2, 0) * 3)
                number, match, score = self._fuzzy_map_darija_number(chunk, threshold)

                if number is not None:
                    replaced[i] = str(int(number))
                    for j in range(i + 1, i + n):
                        replaced[j] = ""
                    used_positions.update(positions)

        # Final reconstruction
        final = [
            rep if rep is not None else word
            for word, rep in zip(words, replaced)
            if rep != ""
        ]
        return " ".join(final)
            
    def convert_text(self, text: str) -> str:
        """
        Convert embedded Darija words to French and replace numbers
        
        Args:
            text: Input text with Darija words and numbers
            
        Returns:
            Text with Darija words converted to French and numbers replaced
        """
        # First replace numbers
        text = self._replace_numbers_in_sentence(text)
        
        # Then convert Darija words to French
        if not self.mapping:
            return text
            
        for darija_variant in self.sorted_keys:
            pattern = re.compile(rf"\b{re.escape(darija_variant)}\b", flags=re.IGNORECASE)
            text = pattern.sub(f" {self.mapping[darija_variant]} ", text)
            
        # Clean up extra spaces
        return ' '.join(text.split())


class SpeechSegment:
    def __init__(self, config: dict):
        self.converter = DarijaFrenchConverter(config)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vad_pipeline = self._load_vad_pipeline()
        self.transcription_model, self.processor = None, None
        
    def _load_vad_pipeline(self) -> Any:
            """Load Voice Activity Detection pipeline"""
            try:
                model_id = self.config.get('vad_model_id', "pyannote/voice-activity-detection")
                hf_token = self.config.get('hf_token', os.getenv("HF_TOKEN"))
                logger.info(f"Loading VAD pipeline: {model_id}")
                pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)
                # # Default to CPU to reduce GPU memory pressure; can be overridden via config
                # vad_device = self.config.get('vad_device', 'cpu')
                pipeline.to(self.device)
                return pipeline
            except Exception as e:
                logger.error(f"Failed to load VAD pipeline: {e}")
                raise

    def transcribe_segments_batched(
        self,
        segments: List[Dict[str, Any]],
        sample_rate: int,
    ) -> List[Dict[str, Any]]:
        """Transcribe segments using efficient batching."""
        if not segments:
            return []

        # prepare indexed items as (idx, array) pairs
        items = []
        for idx, seg in enumerate(segments):
            arr = seg["segment_waveform"]
            # ensure numpy array on CPU
            if torch.is_tensor(arr):
                try:
                    arr = arr.squeeze().cpu().numpy()
                except Exception:
                    arr = arr.squeeze().detach().cpu().numpy()
            items.append((idx, arr))

        
        try:
            items.sort(key=lambda x: x[1].shape[-1])
        except Exception:
            pass

        model, processor = self.transcription_model, self.processor
        batch_size = int(self.config.get('batch_size', self.config.get('chunk_batch_size', 8)))
        texts: Dict[int, str] = {}

        i = 0
        while i < len(items):
            batch_slice = items[i:i + batch_size]
            arrays = [arr for _, arr in batch_slice]
            try:
                inputs = processor(
                    arrays,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding=True
                )
                features = inputs["input_features"].to(self.device)
                with torch.no_grad():
                    if torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            logits = model(features).logits
                    else:
                        logits = model(features).logits
                pred_ids = torch.argmax(logits, dim=-1)
                decoded = processor.batch_decode(pred_ids, skip_special_tokens=True)
                for (orig_idx, _), text in zip(batch_slice, decoded):
                    cleaned = remove_special_characters(text or "")
                    if hasattr(self, 'converter') and getattr(self, 'converter', None):
                        try:
                            cleaned = self.converter.convert_text(cleaned)
                        except Exception:
                            pass
                    texts[orig_idx] = cleaned.strip()
                # free GPU cache for long jobs
                del features, logits, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Batch transcription error at i={i}: {e}")
                # OOM backoff: reduce batch size and retry this slice
                if torch.cuda.is_available() and 'out of memory' in str(e).lower() and batch_size > 1:
                    torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    logger.warning(f"Reducing segments batch size due to OOM. New batch_size={batch_size}")
                    continue
                # Fallback to per-item decode for this slice
                for (orig_idx, arr) in batch_slice:
                    try:
                        inputs = processor(arr, sampling_rate=sample_rate, return_tensors="pt", padding=True)
                        features = inputs["input_features"].to(self.device)
                        with torch.no_grad():
                            if torch.cuda.is_available():
                                with torch.cuda.amp.autocast():
                                    logits = model(features).logits
                            else:
                                logits = model(features).logits
                        pred_ids = torch.argmax(logits, dim=-1)
                        text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
                        texts[orig_idx] = remove_special_characters(text or "").strip()
                    except Exception:
                        texts[orig_idx] = ""
                    finally:
                        try:
                            del features, logits, inputs
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
            i += batch_size

        # Assemble results preserving original order
        out: List[Dict[str, Any]] = []
        for idx, seg in enumerate(segments):
            out.append({ **seg, "text": texts.get(idx, "") })
        return out

    def get_speech_segments(
            self,
            waveform: torch.Tensor,
            sample_rate: int,
            speaker_label: str
        ) -> List[Dict[str, Any]]:
            """
            Extract speech segments using voice activity detection
            
            Args:
                waveform: Audio waveform
                sample_rate: Sample rate
                speaker_label: Speaker label (Agent/Client)
                
            Returns:
                List of speech segments
            """
            path = None
            try:
                # Get VAD pipeline
                vad_pipeline = self.vad_pipeline
                
                # Save waveform to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    path = tmp.name
                    # Ensure waveform is 2D [channels, samples]
                    if waveform.dim() == 1:
                        waveform_to_save = waveform.unsqueeze(0)
                    else:
                        waveform_to_save = waveform
                    torchaudio.save(tmp.name, waveform_to_save, sample_rate)

                # Run VAD
                vad_result = vad_pipeline(path)
                timeline = list(vad_result.get_timeline())
                
                if not timeline:
                    logger.warning(f"No speech segments found for {speaker_label}")
                    return []

                # Merge close segments
                merged_segments = self._merge_segments(
                    timeline,
                    gap_threshold=self.config.get('vad_gap_threshold', 0.8)
                )

                # Extract audio chunks with padding
                segments = self._extract_segments(
                    waveform,
                    sample_rate,
                    merged_segments,
                    speaker_label,
                    padding=self.config.get('vad_padding', 0.5)
                )
                
                # logger.info(f"Extracted {len(segments)} speech segments for {speaker_label}")
                # Remove the temporary file after VAD is done
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {path}: {e}")
                return segments

            except Exception as e:
                logger.error(f"Error in speech segmentation: {e}")
                return []
            finally:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass

    def _merge_segments(self, timeline: List[Any], gap_threshold: float) -> List[Tuple[float, float]]:
            """Merge segments that are close together"""
            if not timeline:
                return []
                
            merged = []
            current_start = timeline[0].start
            current_end = timeline[0].end
            
            for turn in timeline[1:]:
                if turn.start - current_end <= gap_threshold:
                    current_end = turn.end
                else:
                    merged.append((current_start, current_end))
                    current_start, current_end = turn.start, turn.end
                    
            merged.append((current_start, current_end))
            return merged

    def _extract_segments(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        segments: List[Tuple[float, float]],
        speaker_label: str,
        padding: float
    ) -> List[Dict[str, Any]]:
        """Extract audio segments with padding"""
        total_duration = waveform.shape[1] / sample_rate
        extracted = []
        max_len_sec = float(self.config.get('vad_max_segment_sec', 20.0))
        overlap = float(self.config.get('vad_window_overlap_sec', 0.5))
        
        for start, end in segments:
            padded_start = max(0.0, start - padding)
            padded_end = min(total_duration, end + padding)
            # split into windows if longer than max_len_sec
            cur = padded_start
            while cur < padded_end:
                win_end = min(padded_end, cur + max_len_sec)
                start_sample = int(cur * sample_rate)
                end_sample = int(win_end * sample_rate)
                extracted.append({
                    "segment_waveform": waveform[:, start_sample:end_sample],
                    "start": cur,
                    "end": win_end,
                    "speaker": speaker_label
                })
                if win_end >= padded_end:
                    break
                cur = max(cur + max_len_sec - overlap, cur + 0.1)

        return extracted

class SpeechBatchTranscriber:
    def __init__(self, config: dict):
        self.segmenter = SpeechSegment(config)

    def transcribe_mono(self, waveform: torch.Tensor, sample_rate: int, speaker_label: str = "unknown") -> List[Dict[str, Any]]:
        segments = self.segmenter.get_speech_segments(waveform, sample_rate, speaker_label)
        results = self.segmenter.transcribe_segments_batched(segments, sample_rate)
        return results

    