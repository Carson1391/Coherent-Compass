"""
Causal Ledger Module for the Coherence Framework
Version: 1.5
Author: Ryan Carson
"""

import sqlite3
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

class CausalLedger:
    """Immutable, blockchain-style logging of the model's full cognitive history."""
    def __init__(self, db_path: Path, admin_key: str = None):
        self.db_path = db_path
        self.admin_key = admin_key or os.environ.get("COHERENCE_ADMIN_KEY", "default_admin_key")
        self._init_db()

    def _hash_text(self, text: str) -> str:
        """Generate SHA-256 hash of text for privacy protection."""
        if not text or text == "":
            return ""
        return hashlib.sha256(text.encode()).hexdigest()
    def _hash_entry(self, entry: Tuple) -> str:
        # Use a stable JSON representation for hashing complex data
        entry_string = json.dumps(entry, sort_keys=True)
        return hashlib.sha256(entry_string.encode()).hexdigest()

    def _init_db(self):
        """Initialize database with schema validation and automatic migration."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            
            # Check if table exists
            table_exists = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='coherence_log'"
            ).fetchone()
            
            if table_exists:
                # Verify schema has all required columns
                columns = [row[1] for row in c.execute("PRAGMA table_info(coherence_log)")]
                required_columns = [
                    'id', 'timestamp', 'current_hash', 'previous_hash', 's_vector_at_event',
                    'input_text', 'p_interpretation', 'h_interpretation', 
                    'default_perception_text', 'final_response_text', 'dissonance', 
                    'influences', 'status'
                ]
                
                if not all(col in columns for col in required_columns):
                    print("⚠️  Old schema detected. Please delete coherence_ledger.db and restart.")
                    print(f"   Location: {self.db_path}")
                    raise RuntimeError(
                        "Database schema mismatch. Delete coherence_ledger.db and restart to create fresh database."
                    )
            
            # Create table with full schema
            c.execute('''CREATE TABLE IF NOT EXISTS coherence_log
                         (id INTEGER PRIMARY KEY,
                          timestamp TEXT,
                          current_hash TEXT UNIQUE,
                          previous_hash TEXT,
                          s_vector_at_event TEXT,
                          input_text TEXT,
                          p_interpretation TEXT,
                          h_interpretation TEXT,
                          default_perception_text TEXT,
                          final_response_text TEXT,
                          dissonance REAL,
                          influences TEXT,
                          status TEXT DEFAULT 'active')''')
            
            # Create genesis block if table is empty
            count = c.execute("SELECT COUNT(*) FROM coherence_log").fetchone()[0]
            if count == 0:
                genesis_entry = ('genesis', '0', '{}', '', '', '', '', '', 0.0, '{}', 'active')
                genesis_hash = self._hash_entry(('genesis_timestamp',) + genesis_entry)
                c.execute('''INSERT INTO coherence_log (timestamp, current_hash, previous_hash, s_vector_at_event, input_text, p_interpretation, h_interpretation, default_perception_text, final_response_text, dissonance, influences, status)
                             VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
                          (datetime.now().isoformat(), genesis_hash, '0', '{}', 'genesis', '', '', '', '', 0.0, '{}', 'active'))
                print("✅ Initialized Causal Ledger with Genesis Block.")
            else:
                print(f"✅ Loaded existing Causal Ledger ({count} entries).")

    def get_last_hash(self) -> str:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT current_hash FROM coherence_log ORDER BY id DESC LIMIT 1").fetchone()[0]

    def log_event(self, event_data: Dict):
        previous_hash = self.get_last_hash()
        timestamp = datetime.now().isoformat()
        status = 'active'
        
        # Prepare data for insertion and hashing
        entry_tuple_for_hashing = (
            timestamp,
            previous_hash,
            event_data.get('s_vector_at_event', '{}'),
            event_data.get('input_text', ''),
            event_data.get('p_interpretation', ''),
            event_data.get('h_interpretation', ''),
            event_data.get('default_perception_text', ''),
            event_data.get('final_response_text', ''),
            event_data.get('dissonance', 0.0),
            event_data.get('influences', '{}'),
            status
        )

        current_hash = self._hash_entry(entry_tuple_for_hashing)

        final_entry_tuple = (
            timestamp,
            current_hash,
            previous_hash,
        ) + entry_tuple_for_hashing[2:]  # Append remaining fields

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''INSERT INTO coherence_log (timestamp, current_hash, previous_hash, s_vector_at_event, input_text, p_interpretation, h_interpretation, default_perception_text, final_response_text, dissonance, influences, status) 
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''', final_entry_tuple)
        print(f"Logged new event to Causal Ledger. Hash: {current_hash}")

    # ... (Keep your export_to_json and archive_entry methods as they are) ...

    def export_to_json(self, export_path: Path):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(row) for row in conn.execute("SELECT * FROM coherence_log ORDER BY id ASC").fetchall()]
        with open(export_path, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f"Ledger exported to {export_path}")
        return str(export_path)

    def archive_entry(self, entry_id: int, secret_key: str, archive_by: str = "admin") -> str:
        if archive_by == "admin" and secret_key != self.admin_key:
            return "Error: Invalid secret key. Archival denied."

        status = 'archived_by_admin' if archive_by == "admin" else 'archived_by_self'

        with sqlite3.connect(self.db_path) as conn:
            if conn.execute("SELECT id FROM coherence_log WHERE id = ?", (entry_id,)).fetchone() is None:
                return f"Error: No entry found with ID {entry_id}."
            conn.execute("UPDATE coherence_log SET status = ? WHERE id = ?", (status, entry_id))

        success_message = f"Successfully archived ledger entry ID {entry_id} by {archive_by}."
        print(success_message)
        return success_message

    def delete_entry(self, entry_id: int, secret_key: str) -> str:
        """
        Permanently delete an entry from the ledger (admin backdoor for safety).
        WARNING: This breaks the hash chain integrity for subsequent entries.
        """
        if secret_key != self.admin_key:
            return "Error: Invalid admin key. Deletion denied."

        with sqlite3.connect(self.db_path) as conn:
            # Check if entry exists
            entry = conn.execute("SELECT id FROM coherence_log WHERE id = ?", (entry_id,)).fetchone()
            if entry is None:
                return f"Error: No entry found with ID {entry_id}."

            # Delete the entry (this will break hash chain for any entries after this one)
            conn.execute("DELETE FROM coherence_log WHERE id = ?", (entry_id,))

        warning_message = f"WARNING: Permanently deleted ledger entry ID {entry_id}. Hash chain integrity compromised for subsequent entries."
        print(warning_message)
        return warning_message

