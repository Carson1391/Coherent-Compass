# Security and Data Integrity Policy — Coherent Compass

## Overview
The Coherence Framework stores cognitive-event logs in an immutable **Causal Ledger**.
Each entry contains derived embeddings, interpretations, and metrics produced during runtime.
The ledger operates entirely **locally** — no remote servers or telemetry are involved.

## Ledger Integrity
- Every entry is chained using a SHA-256 hash (`previous_hash` → `current_hash`) for auditability.
- The genesis block is created automatically when no previous ledger exists.
- Altering a record breaks the hash chain and invalidates all subsequent hashes.

## Privacy & Data Minimization
- Only derived summaries are stored by default (e.g., first 16 dims of S-vector).
- No API keys, user identifiers, or sensitive text are required to operate the framework.
- All logs are local to your workspace and can be safely deleted or exported to JSON for offline analysis.

## Administrative Access
- Administrative archival operations require an environment variable:
  ```bash
  export COHERENCE_ADMIN_KEY="your_secure_admin_key"
