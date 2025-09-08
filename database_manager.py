#!/usr/bin/env python3
"""
Database Manager for Audio Processing Pipeline v1
Handles call and chunk data storage with PostgreSQL/SQLite support
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
import uuid
import json

try:
    import psycopg2 
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("PostgreSQL support not available. Install psycopg2-binary for PostgreSQL support.")


class DatabaseManager:
    """Database manager for CALL, CHUNK and MESSAGE data storage"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_type = config.get('database_type', 'postgresql').lower()
        self.connection = None
        
        # PostgreSQL configuration only
        if not POSTGRES_AVAILABLE:
            raise ImportError("PostgreSQL support not available. Install psycopg2-binary.")
        
        self.db_config = {
            'host': config.get('db_host', 'localhost'),
            'port': config.get('db_port', 55432),
            'database': config.get('db_name', 'audio_processing'),
            'user': config.get('db_user', 'postgres'),
            'password': config.get('db_password', ''),
        }
        
        self.setup_database()
    
    def setup_database(self):
        """Setup PostgreSQL database connection and create tables"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.connection.autocommit = False
            
            self.create_tables()
            self.logger.info("PostgreSQL database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup PostgreSQL database: {e}")
            raise
    
    def create_tables(self):
        """Create call and chunk tables in PostgreSQL"""
        self._create_postgresql_tables()
    
    def _create_postgresql_tables(self):
        """Create PostgreSQL tables"""
        with self.connection.cursor() as cursor:
            # Create call table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS call (
                    id_enregistrement VARCHAR(255) PRIMARY KEY,
                    emotion_client_globale VARCHAR,
                    ton_agent_global VARCHAR,
                    topics VARCHAR,
                    date_appel DATE,
                    duration_seconds FLOAT,
                    file VARCHAR,
                    partenaire VARCHAR,
                    login_conseiller VARCHAR
                )
            """)
            
            # Create chunk table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunk (
                    id_chunk VARCHAR,
                    id_enregistrement VARCHAR(255),
                    PRIMARY KEY (id_chunk, id_enregistrement),
                    FOREIGN KEY (id_enregistrement) REFERENCES call(id_enregistrement) ON DELETE CASCADE,
                    transcription_chunk TEXT,
                    transcription_agent TEXT,
                    transcription_client TEXT,
                    emotion_client VARCHAR,
                    ton_agent VARCHAR
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS message (
                    order_message INTEGER NOT NULL,
                    id_enregistrement VARCHAR(255),
                    PRIMARY KEY (order_message, id_enregistrement),
                    FOREIGN KEY (id_enregistrement) REFERENCES call(id_enregistrement) ON DELETE CASCADE,
                    text TEXT,
                    speaker VARCHAR,
                    CHECK (order_message >= 1)
                )
            """)
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_call_date ON call(date_appel)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_call_id_enregistrement ON call(id_enregistrement)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_id_enregistrement ON chunk(id_enregistrement)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_message_id_enregistrement ON message(id_enregistrement)")
            self.connection.commit()
    

    
    def insert_call(self, call_data: Dict[str, Any]) -> str:
        """Insert a new call record"""
        try:
            # Ensure id_enregistrement is provided
            if 'id_enregistrement' not in call_data:
                raise ValueError("id_enregistrement is required for call records")
            
            # Set default date if not provided
            if 'date_appel' not in call_data:
                call_data['date_appel'] = date.today().isoformat()
            
            return self._insert_call_postgresql(call_data)
                
        except Exception as e:
            self.logger.error(f"Failed to insert call: {e}")
            raise
    
    def _insert_call_postgresql(self, call_data: Dict[str, Any]) -> str:
        """Insert call record in PostgreSQL"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO call (
                    id_enregistrement, emotion_client_globale, ton_agent_global,
                    topics, date_appel, duration_seconds, file, partenaire, login_conseiller
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id_enregistrement) DO UPDATE SET
                    emotion_client_globale = EXCLUDED.emotion_client_globale,
                    ton_agent_global = EXCLUDED.ton_agent_global,
                    topics = EXCLUDED.topics,
                    duration_seconds = EXCLUDED.duration_seconds,
                    file = EXCLUDED.file,
                    partenaire = EXCLUDED.partenaire,
                    login_conseiller = EXCLUDED.login_conseiller
            """, (
                call_data['id_enregistrement'],
                call_data.get('emotion_client_globale', ''),
                call_data.get('ton_agent_global', ''),
                call_data.get('topics', ''),
                call_data.get('date_appel'),
                call_data.get('duration_seconds'),
                call_data.get('file', ''),
                call_data.get('partenaire', ''),
                call_data.get('login_conseiller','')
            ))
            self.connection.commit()
            return call_data['id_enregistrement']
    

    
    def insert_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """Insert a new chunk record"""
        try:
            # Generate unique ID if not provided
            if 'id_chunk' not in chunk_data:
                chunk_data['id_chunk'] = str(uuid.uuid4())
            
            # Ensure id_enregistrement is provided
            if 'id_enregistrement' not in chunk_data:
                raise ValueError("id_enregistrement is required for chunk records")
            
            return self._insert_chunk_postgresql(chunk_data)
                
        except Exception as e:
            self.logger.error(f"Failed to insert chunk: {e}")
            raise
    
    def _insert_chunk_postgresql(self, chunk_data: Dict[str, Any]) -> str:
        """Insert chunk record in PostgreSQL"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO chunk (
                    id_chunk, id_enregistrement, transcription_chunk, transcription_agent,
                    transcription_client, emotion_client, ton_agent
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id_chunk, id_enregistrement) DO UPDATE SET
                    transcription_chunk = EXCLUDED.transcription_chunk,
                    transcription_agent = EXCLUDED.transcription_agent,
                    transcription_client = EXCLUDED.transcription_client,
                    emotion_client = EXCLUDED.emotion_client,
                    ton_agent = EXCLUDED.ton_agent
            """, (
                chunk_data['id_chunk'],
                chunk_data['id_enregistrement'],
                chunk_data.get('transcription_chunk', ''),
                chunk_data.get('transcription_agent', ''),
                chunk_data.get('transcription_client', ''),
                chunk_data.get('emotion_client', ''),
                chunk_data.get('ton_agent', '')
            ))
            self.connection.commit()
            return chunk_data['id_chunk']
    

    
    def get_call_by_id_enregistrement(self, id_enregistrement: str) -> Optional[Dict[str, Any]]:
        """Get call record by id_enregistrement"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM call WHERE id_enregistrement = %s", (id_enregistrement,))
                result = cursor.fetchone()
                return dict(result) if result else None
                
        except Exception as e:
            self.logger.error(f"Failed to get call by id_enregistrement: {e}")
            return None
    
    def get_chunks_by_id_enregistrement(self, id_enregistrement: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific id_enregistrement"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM chunk WHERE id_enregistrement = %s ORDER BY id_chunk", (id_enregistrement,))
                results = cursor.fetchall()
                return [dict(row) for row in results]
                
        except Exception as e:
            self.logger.error(f"Failed to get chunks by id_enregistrement: {e}")
            return []
    def get_chunk_by_id_enregistrement_and_id(self,id_enregistrement: str, id_chunk: str) -> Optional[Dict[str, Any]]:
        """Get chunk record by id_chunk"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM chunk WHERE id_enregistrement = %s AND id_chunk = %s", (id_enregistrement, id_chunk))
                result = cursor.fetchone()
                return dict(result) if result else None
        except Exception as e:
            self.logger.error(f"Failed to get chunk by id_enregistrement and id_chunk: {e}")
            return None
    
    def update_call_sentiment(self, id_enregistrement: str, emotion_client: str, ton_agent: str, topics: str):
        """Update call with sentiment analysis results"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE call SET 
                        emotion_client_globale = %s,
                        ton_agent_global = %s,
                        topics = %s
                    WHERE id_enregistrement = %s
                """, (emotion_client, ton_agent, topics, id_enregistrement))
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to update call sentiment: {e}")
            raise
    
      
    def insert_message(self, message_data: Dict[str, Any]) -> str:
        """Insert a new message record"""
        if 'order_message' not in message_data:
            raise ValueError("order_message is required for message records")
        if 'id_enregistrement' not in message_data:
            raise ValueError("id_enregistrement is required for message records")
        try:
            return self._insert_message_postgresql(message_data)
        except Exception as e:
            self.logger.error(f"Failed to insert message: {e}")
            raise
    
    def _insert_message_postgresql(self, message_data: Dict[str, Any]) -> str:
        """Insert message record in PostgreSQL"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO message (
                    order_message, id_enregistrement, text, speaker
                ) VALUES (%s, %s, %s, %s)
            """, (
                message_data['order_message'],
                message_data['id_enregistrement'],
                message_data['text'],
                message_data['speaker']
            ))
            self.connection.commit()
            return message_data['order_message']
    
    def get_messages_by_id_enregistrement(self, id_enregistrement: str) -> List[Dict[str, Any]]:
        """Get all messages for a specific id_enregistrement"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM message WHERE id_enregistrement = %s ORDER BY order_message", (id_enregistrement,))
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            self.logger.error(f"Failed to get messages by id_enregistrement: {e}")
            return []
    
    def get_message_by_id_enregistrement_and_order_message(self, id_enregistrement: str, order_message: int) -> Optional[Dict[str, Any]]:
        """Get message record by id_enregistrement and order_message"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM message WHERE id_enregistrement = %s AND order_message = %s", (id_enregistrement, order_message))
                result = cursor.fetchone()
                return dict(result) if result else None
        except Exception as e:
            self.logger.error(f"Failed to get message by id_enregistrement and order_message: {e}")
            return None
    
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as total_calls FROM call")
                total_calls = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) as total_chunks FROM chunk")
                total_chunks = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) as processed_calls FROM call WHERE emotion_client_globale IS NOT NULL")
                processed_calls = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) as total_messages FROM message")
                total_messages = cursor.fetchone()[0]
            
            return {
                'total_calls': total_calls,
                'total_chunks': total_chunks,
                'processed_calls': processed_calls,
                'total_messages': total_messages,
                'processing_rate': (processed_calls / total_calls * 100) if total_calls > 0 else 0

            }
            
        except Exception as e:
            self.logger.error(f"Failed to get processing stats: {e}")
            return {'total_calls': 0, 'total_chunks': 0, 'processed_calls': 0, 'total_messages': 0, 'processing_rate': 0}
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 
  