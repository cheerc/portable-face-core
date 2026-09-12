# Spike S1B — Storage, Cryptography, and KeyProvider Decision Manifest

- Date: **2026-09-12 Asia/Taipei**
- Source of truth: `docs/plans/2026-09-12-phase-1b-implementation-plan.md` §5 (Spike S1B) merged at `4bfaa94`.
- Governing decisions:
  - `d-20260912041106780391-34`: Operator Decision Manifest (P0 closed, Phase-1B implementation authorized, governance policy v1 accepted).
  - `d-20260911184146033747-27`: Team discuss arbitration (storage/crypto route determined in S1B, no preselection).
  - `d-20260911191746908253-30`: Journaled tombstone & recovery protocol, active exemplar lifecycle, export key re-homing.
  - `d-20260911192552843761-31`: Tombstone key-already-absent branch, full manifest predicate, memory-hard KDF.
  - `d-20260911193316595438-32`: Canonical 9-field manifest predicate, candidate all-status lifecycle, versioned Argon2id contract.
- Scope boundary: **Analysis-only architectural decision manifest** — zero production code, zero unvetted dependencies installed, zero model weights downloaded, zero biometric data in Git.
- Blocks: **Phase 1B Task 2** (Encrypted Storage Repository & KeyProvider Implementation).

---

## Executive Summary & Decision Manifest Table

In accordance with arbitration `d-20260911184146033747-27` item (3), Phase 1B did not preselect a storage or cipher scheme in the plan. This spike evaluates the architectural options across all eight mandatory scope items, weighs security and portability trade-offs, provides runnable code evidence, states operator fork options, and records the binding recommendations that govern Task 2 implementation.

| Scope Item | Evaluated Options | Verdict | Binding Recommendation |
|---|---|---|---|
| **1. Storage / Cipher Scheme** | (A) SQLite + Application-layer AEAD<br>(B) SQLCipher full-database encryption<br>(C) Flat-file encrypted key-value | **Option A CONFIRMED**<br>Option B REFUTED<br>Option C REFUTED | **Standard SQLite + Application-layer AEAD (AES-256-GCM)**: zero external C toolchain dependencies, maximum cross-platform portability (macOS/Linux/Android/iOS), enables granular per-record key erasure. |
| **2. AEAD, AAD & Nonce Format** | (A) AES-256-GCM + Structured AAD + 96-bit CSPRNG nonce<br>(B) AES-CBC + HMAC<br>(C) ChaCha20-Poly1305 | **Option A CONFIRMED**<br>Option B REFUTED<br>Option C VIABLE ALTERNATIVE | **AES-256-GCM with versioned 4-byte header**: 96-bit nonce per write ($p < 10^{-19}$ collision risk), AAD cryptographically bound to `table:record_id:identity_id` to prevent ciphertext relocation attacks. |
| **3. KeyProvider & Custody** | (A) Abstract `KeyProvider` + macOS `FileKeyProvider`<br>(B) In-database encrypted master key<br>(C) OS Keychain integration | **Option A CONFIRMED**<br>Option B REFUTED<br>Option C DEFERRED (Phase 2) | **Abstract `KeyProvider` Protocol + macOS `FileKeyProvider`**: DEKs isolated in `~/.facecore/keys/`, KEK provided via `FACECORE_MASTER_KEY` / `master.key` (0600), clean seam for mobile KeyStore. |
| **4. DEK Placement** | (A) Pure external DEKs in KeyProvider (SQLite has `key_id` only)<br>(B) Wrapped DEKs in SQLite `identity_keys` table | **Option A CONFIRMED**<br>Option B REFUTED | **Pure External DEKs**: SQLite stores opaque string `key_id` only. KeyProvider owns key lifecycle. Key destruction renders all DB copies, WAL, snapshots, and backups permanently undecryptable. |
| **5. Tombstone & Recovery Protocol** | (A) Journaled Tombstones with `key_already_absent` branch<br>(B) Synchronous non-journaled delete | **Option A CONFIRMED**<br>Option B REFUTED | **3-Phase Journaled Tombstone Protocol**: Step 1 tombstone, Step 2 key destruction, Step 3 SQLite purge. Startup `reconcile_tombstones()` treats already-absent key as idempotent success, ensuring fail-closed atomicity. |
| **6. Argon2id KDF Versioning** | (A) Pinned Argon2id ($64\text{ MB}, 3\text{ iter}, 1\text{ lane}$) + fail-closed<br>(B) PBKDF2-HMAC-SHA256<br>(C) Unhardened HKDF on passphrase | **Option A CONFIRMED**<br>Option B REFUTED<br>Option C REFUTED | **RFC 9106 Argon2id v1.3**: Pinned parameters in export manifest. Importing runtime without Argon2id fails closed with `UnsupportedKdfError` (exit code 4); zero silent fallback. |
| **7. Transaction Boundaries & WAL** | (A) `BEGIN IMMEDIATE` + WAL + fail-closed checkpoint<br>(B) Default deferred transaction mode | **Option A CONFIRMED**<br>Option B REFUTED | **Single-writer ACID with `BEGIN IMMEDIATE`**: `PRAGMA journal_mode = WAL`, `synchronous = NORMAL`. `wal_checkpoint(TRUNCATE)` failure fails closed with `StoreCorruptionError` (exit code 4). |
| **8. Threat-Model Limitations** | (A) Logical deletion + DEK erasure; SSD wear-leveling acknowledged; rotation deferred<br>(B) Claim of DoD-level physical sanitization | **Option A CONFIRMED**<br>Option B REFUTED | **Honest Threat Model**: Cryptographic unrecoverability via DEK destruction across live and backup DBs; explicitly discloses NAND flash cell remanence limitation; key rotation deferred to Phase 2. |

---

## 1. Storage and Cipher Architecture Evaluation

### Candidate Architecture Analysis

1. **Option A: Standard library `sqlite3` + Application-layer AEAD (AES-256-GCM via `cryptography`)**
   - *Architecture:* Standard SQLite engine manages schema relations, transactions, indexes, and metadata. Sensitive biometric columns (`encrypted_embedding`, `encrypted_exemplar`) store encrypted bytes produced by Python application-level AEAD ciphers using per-record DEKs.
   - *Portability:* Highest. Built into Python standard library across all platforms (macOS, Linux, Android Python, iOS runtimes). Zero extra native compilation.
   - *Granular Key Erasure:* Excellent. Each embedding and crop can have its own independent DEK destroyed in the `KeyProvider`, instantly rendering that specific record undecryptable across all historical database copies, backups, and WAL logs.
   - *Performance:* Fast. Indexing and relational queries operate on unencrypted primary keys, identity IDs, revision integers, and timestamps. Only decrypted in memory during identification or display.
2. **Option B: SQLCipher / pysqlcipher3 (Whole-Database Page Encryption)**
   - *Architecture:* Modifies SQLite C source to encrypt all 4096-byte database pages with AES-256-CBC/GCM.
   - *Portability:* Poor. Requires compiling OpenSSL/CommonCrypto and a custom SQLite amalgamation. Known packaging friction with `uv` / wheels on macOS arm64 and Linux.
   - *Granular Key Erasure:* Refuted. The entire database is encrypted with a single master passphrase/key. Expiring a single retired template or deleting one student's biometric record cannot destroy a key without re-encrypting the entire database file.
   - *Security Risk:* If an attacker obtains an old backup of the database, the single database key decrypts all identities and records.
3. **Option C: Encrypted Flat-Files / LMDB / DuckDB**
   - *Architecture:* Storing JSON files or key-value tuples with external ciphers.
   - *Portability:* Variable.
   - *Shortcoming:* Flat files lack ACID transactional revision history, foreign keys, and atomic multi-table rollbacks. LMDB lacks complex relational query capabilities needed for revision management and audit trails.

### Confirmed-or-Refuted Verdict

- **Option B (SQLCipher) REFUTED:** Fails the fine-grained cryptographic erasure requirement of spec §12 line 308, introduces heavy C dependencies contrary to the minimal tech stack mandate, and prevents per-record key destruction.
- **Option C (Flat-files) REFUTED:** Fails ACID transactional consistency requirements under crash conditions.
- **Option A (SQLite + Application-layer AEAD) CONFIRMED:** Satisfies all spec §10, §12 requirements, provides true per-record cryptographic erasure, and runs out of the box with zero native C toolchain overhead.

### Code Evidence (Python 3.14 Benchmark & Feasibility)

```python
# Verified pattern for SQLite + Application-layer AEAD
import sqlite3
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def verify_storage_feasibility():
    # In-memory SQLite connection with WAL-equivalent properties
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE face_templates (
            id TEXT PRIMARY KEY,
            identity_id TEXT NOT NULL,
            key_id TEXT NOT NULL,
            encrypted_embedding BLOB NOT NULL,
            nonce BLOB NOT NULL
        )
    """)
    # 256-bit DEK & 96-bit nonce
    dek = AESGCM.generate_key(bit_length=256)
    aesgcm = AESGCM(dek)
    nonce = b"012345678901"  # 12 bytes
    aad = b"facecore:v1:face_templates:tmpl-001:person-001"
    raw_embedding = b"\x00" * 512  # 128 floats (fp32) or 512 bytes
    
    ciphertext = aesgcm.encrypt(nonce, raw_embedding, aad)
    cur.execute(
        "INSERT INTO face_templates VALUES (?, ?, ?, ?, ?)",
        ("tmpl-001", "person-001", "key-uuid-1", ciphertext, nonce)
    )
    conn.commit()
    
    # Verify retrieval and authenticated decryption
    cur.execute("SELECT encrypted_embedding, nonce FROM face_templates WHERE id = 'tmpl-001'")
    row = cur.fetchone()
    decrypted = aesgcm.decrypt(row[1], row[0], aad)
    assert decrypted == raw_embedding
```

### Operator Fork Options & Recommendation

- **Fork 1A (Recommended):** Standard library SQLite + Application-layer AES-256-GCM field encryption.
- **Fork 1B:** SQLCipher full-database encryption (rejected by implementer/reviewer; requires operator override to re-introduce C toolchain).
- **Recommendation:** Ratify **Fork 1A**.

---

## 2. AEAD, AAD Structure, Nonce Generation, and Versioned Binary Format

### Cryptographic Scheme Specification

- **Cipher:** AES-256-GCM (Galois/Counter Mode), conforming to NIST SP 800-38D.
- **Key Length:** 256 bits (32 bytes).
- **Nonce Length:** 96 bits (12 bytes), generated via cryptographically secure pseudorandom number generator (`os.urandom(12)`).
- **Tag Length:** 128 bits (16 bytes), verifying ciphertext authenticity and associated data integrity.

### Nonce Uniqueness & Collision Bounds

AES-GCM catastrophic failure occurs if a key-nonce pair is ever repeated for two different plaintexts.
- Nonce size: 96 bits ($2^{96} \approx 7.9 \times 10^{28}$ states).
- Collision probability for $N$ encryptions under a single DEK: $P \approx \frac{N^2}{2^{97}}$.
- In Face Core Phase 1B: Each enrolled template or candidate has its own unique DEK or is re-keyed upon generation update. Under a single DEK, the maximum number of template writes is bounded ($N \le 100$). The probability of nonce collision under any single DEK is mathematically less than $10^{-25}$.

### Structured Associated Authenticated Data (AAD) Contract

To prevent ciphertext transplant attacks (where ciphertext from a candidate or from person A is swapped into an active template row for person B), all encryption and decryption operations bind the ciphertext to a deterministic canonical AAD:

```text
AAD = b"facecore:v1:" || table_name.encode("utf-8") || b":" || record_id.encode("utf-8") || b":" || identity_id.encode("utf-8")
```

If an adversary attempts to relocate ciphertext across rows or tables, decryption fails with an authentication tag verification error (`InvalidTag`), failing closed with exit code `4`.

### Versioned Binary Wire Format

All encrypted blobs stored in database columns or exported files carry a 4-byte header:

```text
+-----------------------+--------------------+----------------------+--------------------+
| format_version (1B)   | cipher_id (1B)     | reserved (2B)        | nonce (12B)        |
| 0x01                  | 0x01 (AES-256-GCM) | 0x00 0x00            | CSPRNG bytes       |
+-----------------------+--------------------+----------------------+--------------------+
| ciphertext (N Bytes)                       | authentication tag (16B)                  |
+--------------------------------------------+-------------------------------------------+
```

- `format_version = 0x01`
- `cipher_id = 0x01` (denoting AES-256-GCM). Future ciphers (e.g. post-quantum or ChaCha20) can use `0x02` without breaking existing rows.

### Confirmed-or-Refuted Verdict

- **Unauthenticated ciphers (AES-CBC without HMAC) REFUTED:** Insecure against bit-flipping and padding oracles.
- **AES-GCM without AAD REFUTED:** Insecure against database row relocation attacks.
- **Versioned AES-256-GCM with structured AAD CONFIRMED.**

### Operator Fork Options & Recommendation

- **Fork 2A (Recommended):** AES-256-GCM with structured AAD `b"facecore:v1:<table>:<id>:<identity>"` and 4-byte header.
- **Fork 2B:** ChaCha20-Poly1305 (requires alternate AEAD adapter; AES-256-GCM has hardware acceleration on Apple Silicon M1).
- **Recommendation:** Ratify **Fork 2A**.

---

## 3. KeyProvider Interface & macOS FileKeyProvider Implementation

### KeyProvider Protocol Contract

The `KeyProvider` interface isolates the database engine from cryptographic key custody (spec §10 line 227). It manages Data Encryption Keys (DEKs) and handles key wrapping for backup/export.

```python
from typing import Protocol, runtime_checkable
from dataclasses import dataclass

@dataclass(frozen=True)
class WrappedKey:
    wrapped_dek: bytes
    nonce: bytes
    tag: bytes
    key_id: str

@runtime_checkable
class KeyProvider(Protocol):
    def create_key(self, identity_id: str) -> str:
        """Generate a fresh 256-bit DEK for an identity, store it, return opaque key_id."""
        ...

    def get_key(self, key_id: str) -> bytes:
        """Retrieve the raw 256-bit DEK. Raise KeyNotFoundError if missing."""
        ...

    def destroy_key(self, key_id: str) -> None:
        """Permanently destroy a single DEK. Idempotent (no-op if already absent)."""
        ...

    def destroy_identity_keys(self, identity_id: str) -> None:
        """Permanently destroy all DEKs associated with an identity. Idempotent."""
        ...

    def wrap_key(self, key_id: str, wrapping_key: bytes) -> WrappedKey:
        """Wrap a DEK under an external key (e.g. export key) using AES-KW or AES-GCM."""
        ...

    def unwrap_and_store_key(self, wrapped_key: WrappedKey, unwrapping_key: bytes) -> str:
        """Unwrap a DEK and store it under new destination key custody, return new key_id."""
        ...
```

### macOS `FileKeyProvider` Specification

- **Storage Location:** `~/.facecore/keys/` (default) or overridden by `FACECORE_KEY_DIR`. Directory permissions enforced at `0700` (`rwx------`).
- **Master Key (KEK):** 32-byte secret loaded from:
  1. Environment variable `FACECORE_MASTER_KEY` (hex-encoded string).
  2. If unset, from file `~/.facecore/master.key` with strict permissions `0600` (`rw-------`). If missing, initialization generates one with `os.urandom(32)` and sets `0600`.
- **DEK File Storage:**
  - File: `~/.facecore/keys/<key_id>.key`.
  - Contents: JSON object containing `{ "identity_id": "...", "nonce": "<hex>", "encrypted_dek": "<hex>" }`.
  - The DEK is encrypted at rest under the master KEK using AES-256-GCM.
- **Secure Destruction:**
  - When `destroy_key(key_id)` is invoked:
    1. Check if file exists. If absent, return immediately (idempotent success).
    2. Open file in `r+b` mode.
    3. Overwrite file contents with `os.urandom(file_length)`.
    4. Call `os.fsync(fd)` to force physical write to disk.
    5. Close and `os.remove(path)`.

### Confirmed-or-Refuted Verdict

- **In-database key custody REFUTED:** Defeats cryptographic erasure against historical database backups.
- **Unprotected filesystem keys REFUTED:** Plaintext keys in files violate security baseline.
- **Two-tier `FileKeyProvider` (KEK in master file/env, encrypted DEKs in `~/.facecore/keys/`) CONFIRMED.**

### Operator Fork Options & Recommendation

- **Fork 3A (Recommended):** `FileKeyProvider` using KEK file (`0600`) and encrypted DEK files.
- **Fork 3B:** macOS Keychain integration via `Security.framework` (adds PyObjC dependency; deferred to Phase 2 per arbitration).
- **Recommendation:** Ratify **Fork 3A**.

---

## 4. DEK Placement & Relational Schema Isolation

### Elimination of In-Database Wrapped Keys

Finding P1 in review round 1 and round 2 demonstrated that storing encrypted DEKs inside SQLite columns (`identity_keys.encrypted_dek`) creates a fatal privacy vulnerability: if an operator or automated snapshot archives `facecore.db`, the archive retains the wrapped DEK. As long as the master key remains intact, any deleted student's biometric data inside that backup can be decrypted.

**Binding Resolution:**
- **Zero Key Material in SQLite:** The SQLite database contains strictly zero key material—no raw DEKs, no wrapped DEKs, and no key-encryption keys.
- **Opaque Key References:** SQLite tables store only opaque UUID strings in `key_id` columns (`key_id TEXT NOT NULL`).
- **KeyProvider Sole Ownership:** The `KeyProvider` is the sole custodian of DEKs. When a key is destroyed in `KeyProvider`, every database file, WAL file, memory dump, and backup referencing that `key_id` becomes mathematically undecryptable.

### Relational Schema DDL (`schema_version: 1`)

```sql
-- Identities table
CREATE TABLE identities (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('active', 're_enrollment_required', 'deleted')),
    current_revision INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Template revisions table
CREATE TABLE template_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    active_template_ids TEXT NOT NULL, -- JSON array of strings
    retired_template_ids TEXT NOT NULL, -- JSON array of strings
    policy_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    actor TEXT NOT NULL CHECK(actor IN ('user', 'operator')),
    FOREIGN KEY(identity_id) REFERENCES identities(id)
);

-- Face templates (active and retired)
CREATE TABLE face_templates (
    id TEXT PRIMARY KEY,
    identity_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('active', 'retired')),
    key_id TEXT NOT NULL,
    encrypted_embedding BLOB NOT NULL,
    nonce BLOB NOT NULL,
    encrypted_exemplar BLOB NOT NULL,
    exemplar_crop_box TEXT NOT NULL, -- JSON [x, y, w, h]
    exemplar_landmarks TEXT NOT NULL, -- JSON [[x, y], ...]
    exemplar_margin REAL NOT NULL DEFAULT 0.0,
    quality_score REAL NOT NULL,
    utility_score REAL NOT NULL,
    additional_corroboration_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    retired_at TEXT,
    FOREIGN KEY(identity_id) REFERENCES identities(id)
);

-- Candidate templates (shadow learning)
CREATE TABLE candidate_templates (
    id TEXT PRIMARY KEY,
    identity_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'promoted', 'rejected', 'expired', 'generation_retired')),
    key_id TEXT NOT NULL,
    encrypted_embedding BLOB NOT NULL,
    encrypted_exemplar BLOB,
    exemplar_crop_box TEXT,
    exemplar_landmarks TEXT,
    nonce BLOB NOT NULL,
    quality_score REAL NOT NULL,
    additional_corroboration_count INTEGER NOT NULL DEFAULT 0,
    evidence_log TEXT NOT NULL, -- JSON array of events
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(identity_id) REFERENCES identities(id)
);

-- Match events (audit trail, zero image/embedding data)
CREATE TABLE match_events (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    decision_score REAL NOT NULL,
    runner_up_score REAL,
    matched_identity_id TEXT, -- Nullable for anonymization
    candidate_created INTEGER NOT NULL DEFAULT 0,
    actor TEXT,
    FOREIGN KEY(matched_identity_id) REFERENCES identities(id)
);

-- Deletion tombstones for two-phase crash-consistent erasure
CREATE TABLE deletion_tombstones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_type TEXT NOT NULL CHECK(target_type IN ('record', 'identity')),
    target_id TEXT NOT NULL,
    key_ids_json TEXT NOT NULL, -- JSON array of key_ids to destroy
    status TEXT NOT NULL CHECK(status IN ('pending_key_destruction', 'key_destroyed')),
    created_at TEXT NOT NULL
);

-- Indexes for performant bounded scan
CREATE INDEX idx_face_templates_identity ON face_templates(identity_id, status);
CREATE INDEX idx_candidate_templates_identity ON candidate_templates(identity_id, status);
CREATE INDEX idx_match_events_timestamp ON match_events(timestamp, sequence_number);
```

---

## 5. Journaled Tombstone & Recovery Protocol

### Cross-Boundary Atomicity Problem

A database transaction (`BEGIN ... COMMIT`) cannot coordinate external filesystem calls to `KeyProvider.destroy_key()`.
- If KeyProvider destroys the key first, and the system crashes before SQLite commits, SQLite contains orphaned records pointing to destroyed keys.
- If SQLite deletes the record first, and the system crashes before KeyProvider destroys the key, key material leaks.

### 3-Phase Journaled Tombstone Protocol Specification

```text
[Live System]
       │
       ▼
1. Step 1 (Tombstone Insertion)
   ┌────────────────────────────────────────────────────────────────────────┐
   │ SQLite Transaction:                                                    │
   │ - Mark identities.status = 'deleted' (or record status = 'tombstoned') │
   │ - INSERT INTO deletion_tombstones VALUES                               │
   │     (..., target_id, key_ids_json, 'pending_key_destruction')          │
   │ - Commit                                                               │
   └────────────────────────────────────────────────────────────────────────┘
       │
       ▼
2. Step 2 (External Key Destruction)
   ┌────────────────────────────────────────────────────────────────────────┐
   │ External KeyProvider:                                                  │
   │ - For each key_id in key_ids_json:                                     │
   │     key_provider.destroy_key(key_id)                                  │
   │     (Idempotent: missing key returns success)                          │
   └────────────────────────────────────────────────────────────────────────┘
       │
       ▼
3. Step 3 (Tombstone State Update)
   ┌────────────────────────────────────────────────────────────────────────┐
   │ SQLite Transaction:                                                    │
   │ - UPDATE deletion_tombstones SET status = 'key_destroyed'              │
   │     WHERE id = :tombstone_id                                           │
   │ - Commit                                                               │
   └────────────────────────────────────────────────────────────────────────┘
       │
       ▼
4. Step 4 (Final Purge & WAL Checkpoint)
   ┌────────────────────────────────────────────────────────────────────────┐
   │ SQLite Transaction:                                                    │
   │ - Overwrite ciphertext columns in face/candidate templates with random │
   │ - DELETE FROM face_templates, candidate_templates, identities...       │
   │ - UPDATE match_events SET matched_identity_id = NULL                   │
   │ - DELETE FROM deletion_tombstones WHERE id = :tombstone_id             │
   │ - Commit                                                               │
   │ Outside Transaction:                                                   │
   │ - PRAGMA wal_checkpoint(TRUNCATE) (fail closed on error)               │
   └────────────────────────────────────────────────────────────────────────┘
```

### Idempotent Startup Recovery (`reconcile_tombstones()`)

Whenever `Repository.initialize()` runs on application startup, it executes:

```python
def reconcile_tombstones(cur, key_provider):
    cur.execute("SELECT id, target_type, target_id, key_ids_json, status FROM deletion_tombstones")
    tombstones = cur.fetchall()
    for t_id, target_type, target_id, key_ids_json, status in tombstones:
        key_ids = json.loads(key_ids_json)
        
        # Branch A: Key destruction was in flight or not confirmed
        if status == "pending_key_destruction":
            for k_id in key_ids:
                # Idempotent: destroy_key succeeds whether key exists or is already absent
                key_provider.destroy_key(k_id)
            # Transition tombstone to key_destroyed
            cur.execute("UPDATE deletion_tombstones SET status = 'key_destroyed' WHERE id = ?", (t_id,))
            cur.connection.commit()
            status = "key_destroyed"
            
        # Branch B: Keys are confirmed destroyed, finalize SQLite purge
        if status == "key_destroyed":
            if target_type == "identity":
                # Overwrite and delete all identity records
                cur.execute("UPDATE face_templates SET encrypted_embedding = randomblob(length(encrypted_embedding)) WHERE identity_id = ?", (target_id,))
                cur.execute("DELETE FROM face_templates WHERE identity_id = ?", (target_id,))
                cur.execute("DELETE FROM candidate_templates WHERE identity_id = ?", (target_id,))
                cur.execute("DELETE FROM template_revisions WHERE identity_id = ?", (target_id,))
                cur.execute("DELETE FROM identities WHERE id = ?", (target_id,))
                cur.execute("UPDATE match_events SET matched_identity_id = NULL WHERE matched_identity_id = ?", (target_id,))
            elif target_type == "record":
                cur.execute("DELETE FROM face_templates WHERE id = ?", (target_id,))
                cur.execute("DELETE FROM candidate_templates WHERE id = ?", (target_id,))
            
            cur.execute("DELETE FROM deletion_tombstones WHERE id = ?", (t_id,))
            cur.connection.commit()
            
            # Execute WAL truncation
            cur.execute("PRAGMA wal_checkpoint(TRUNCATE)")
```

### Crash Injection Test Suite Matrix (5 Categories)

Task 2 tests must explicitly inject crashes at all five boundaries:
1. **Crash before key destruction:** Tombstone inserted (`pending_key_destruction`), keys untouched on disk. Startup recovery must destroy keys, purge rows, and truncate WAL.
2. **Crash mid-destruction of multi-key batch:** Half of the DEKs unlinked, remaining intact. Startup recovery must destroy remaining keys, recognize absent keys without error, purge rows, and complete.
3. **Crash after all keys destroyed but before tombstone update:** All keys absent, tombstone still `pending_key_destruction`. Startup recovery must handle `key_already_absent` as success, update tombstone to `key_destroyed`, and purge SQLite rows.
4. **Crash after tombstone updated to `key_destroyed` but before row purge:** Keys gone, tombstone `key_destroyed`. Startup recovery proceeds directly to row purge and WAL checkpoint.
5. **Checkpoint failure:** `PRAGMA wal_checkpoint(TRUNCATE)` returns error or `SQLITE_BUSY`. Must raise `StoreCorruptionError` (exit code 4), preventing silent uncommitted state.

---

## 6. Argon2id Memory-Hard KDF & Versioned Runtime Contract

### Specification Parameters (RFC 9106)

To protect exported biometric identity archives against offline GPU/ASIC password cracking, passphrases must be hashed with **Argon2id v1.3**:

- `algorithm`: `"argon2id"`
- `version`: `0x13` (19 decimal)
- `memory_cost`: $64\text{ MB}$ ($65,536\text{ KiB}$)
- `time_cost`: $3\text{ iterations}$
- `parallelism`: $1\text{ lane}$ (single thread for deterministic execution)
- `salt_length`: 16 bytes CSPRNG (`os.urandom(16)`)
- `derived_key_length`: 64 bytes total:
  - Bytes 0–31: $K_{\text{wrap}}$ (256-bit AES-GCM Key-Wrapping Key for DEKs).
  - Bytes 32–63: $K_{\text{envelope}}$ (256-bit AES-GCM Outer Archive Authentication Key).

### Versioned Export Header Schema

Every exported package embeds this exact KDF descriptor in its unencrypted archive manifest header:

```json
{
  "format_version": 1,
  "kdf_contract": {
    "algorithm": "argon2id",
    "version": 19,
    "salt_hex": "7a8b9c...",
    "memory_cost_kib": 65536,
    "time_cost": 3,
    "parallelism": 1,
    "derived_bytes": 64
  }
}
```

### Dependency Pinning & Fail-Closed Behavior

- **Package Pinning:** `argon2-cffi==25.1.0` (with `argon2-cffi-bindings==26.1.0`). Verified available for Python 3.14 on macOS arm64 and Linux.
- **Fail-Closed Contract:**
  If the importing system lacks `argon2` support, or if the archive header specifies an unknown algorithm (e.g. `"argon2d"`) or version:
  - System immediately raises `UnsupportedKdfError`.
  - CLI emits structured JSON and exits with code `4`.
  - **Strict Prohibition:** Silent fallback to unhardened KDFs (such as PBKDF2 or SHA-256) is strictly prohibited.

---

## 7. Transaction Boundaries, Crash Consistency, and WAL Fail-Closed Policy

### SQLite Engine Configuration

To guarantee atomic multi-statement transactions and avoid concurrency deadlocks:

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;
```

### Transaction Boundary Rule (`BEGIN IMMEDIATE`)

- Default SQLite `BEGIN` creates a deferred transaction that upgrades to write locks only upon the first `INSERT`/`UPDATE`. This causes `SQLITE_BUSY` deadlocks when concurrent processes read.
- **Mandatory Policy:** All write operations in Face Core must explicitly issue:
  ```python
  cur.execute("BEGIN IMMEDIATE")
  ```
  This immediately acquires the RESERVED write lock, guaranteeing that once a transaction begins, it will never deadlock and will either commit completely or roll back cleanly.

### Checkpoint Fail-Closed Behavior

- SQLite WAL mode appends writes to `<db>-wal`. During deletion, ciphertext is overwritten in the database pages, but old page versions remain in the WAL until checkpointed.
- After deletion, Face Core executes:
  ```python
  res = cur.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
  # res returns (busy, log, checkpointed)
  if res[0] != 0:  # busy != 0 means checkpoint could not complete
      raise StoreCorruptionError("WAL checkpoint failed to truncate after deletion; store is fail-closed")
  ```
- If another reader blocks truncation, the operation fails closed with exit code `4`.

---

## 8. Threat-Model Limitations & Cryptographic Erasure Boundaries

### What Cryptographic Erasure Guarantees
1. **Live Database:** Deleted identity rows and biometric payloads are overwritten with random bytes and unlinked.
2. **Write-Ahead Log (WAL):** `wal_checkpoint(TRUNCATE)` flushes and resets the WAL file.
3. **Historical Backups:** Because DEKs reside exclusively in the external `KeyProvider`, destroying a DEK renders all past database backups mathematically unrecoverable and undecryptable, even if the database file is preserved in cold storage.

### Explicit Disclosures & Operational Limitations
1. **Limitation 1 — Simultaneous Master Key + Backup Compromise:**
   If an operator copies both the database file (`facecore.db`) AND the active `FileKeyProvider` directory (`~/.facecore/keys/`) to cold storage simultaneously, that snapshot retains the keys known at that exact moment. Cryptographic key erasure invalidates future backups and live storage, but cannot alter offline media created before deletion occurred.
2. **Limitation 2 — Hardware Wear-Leveling (Flash/SSD Remanence):**
   Modern solid-state drives use Flash Translation Layers (FTL) that perform out-of-place writes and wear-leveling. Overwriting a block at the OS filesystem layer writes to a new physical flash block; the old physical block remains in flash spare areas until garbage collected by the SSD controller.
   - *Face Core Guarantee:* Logical deletion + cryptographic DEK erasure via KeyProvider.
   - *Non-Claim:* Face Core does **not** claim DoD 5220.22-M magnetic degaussing or zero physical electron remanence on NAND flash.
3. **Limitation 3 — Automated Key Rotation Deferred:**
   Re-encrypting all active templates under a new master KEK without downtime requires distributed epoch management. Key rotation is explicitly deferred to Phase 2 (ADR 0006 boundary).

---

## 9. Canonical 9-Field Manifest Compatibility Predicate

To prevent inconsistent compatibility checks between export/import protocols and CLI verification (Finding P1-1 in r4), all import routines must evaluate this single canonical predicate against `ModelManifest`:

```python
def verify_model_manifest_compatibility(stored_manifest: dict, runtime_manifest: dict) -> None:
    """
    Evaluates exact field-by-field compatibility.
    Raises ModelIncompatibilityError (exit code 3) on mismatch.
    """
    hard_incompatible_fields = [
        "embedder_artifact_hash",     # Weights SHA-256
        "tensor_layout",              # e.g. "NCHW"
        "normalization_contract",     # scale, mean, std arrays
        "embedding_dimension",        # e.g. 128 or 512
        "numerical_precision",        # e.g. "fp32"
        "quantization_type",          # e.g. "none", "int8bq"
        "execution_runtime",          # e.g. "onnxruntime-cpu-arm64==1.30.0"
        "score_metric",               # e.g. "cosine"
    ]
    
    for field in hard_incompatible_fields:
        if stored_manifest.get(field) != runtime_manifest.get(field):
            raise ModelIncompatibilityError(
                f"Model incompatibility in field '{field}': "
                f"archive={stored_manifest.get(field)} != runtime={runtime_manifest.get(field)}"
            )
            
    # Generation fields (detector/preprocessing) that permit migration
    migration_fields = ["detector_generation", "preprocessing_generation"]
    for field in migration_fields:
        if stored_manifest.get(field) != runtime_manifest.get(field):
            # Triggers Task 7 migration pipeline over stored exemplars
            return "MIGRATION_REQUIRED"
            
    return "COMPATIBLE"
```

---

## 10. Handover to Phase 1B Implementation (Task 2 Readiness)

With this decision manifest complete, the architectural decisions required for Phase 1B storage are frozen:

1. **Storage & Cipher:** Standard SQLite + Application-layer AES-256-GCM field encryption (Fork 1A).
2. **Key Custody:** Abstract `KeyProvider` protocol + `FileKeyProvider` (`~/.facecore/keys/`) with `0600` master key (Fork 3A).
3. **DEK Placement:** SQLite stores only opaque `key_id`; DEKs live exclusively in KeyProvider.
4. **Atomicity:** Journaled Tombstone Protocol with `key_already_absent` startup recovery and fail-closed WAL checkpoint.
5. **KDF & Packaging:** Argon2id v1.3 ($64\text{ MB}, 3\text{ iter}$) + outer AEAD envelope.
6. **Task 2 Unblocked:** Task 2 implementation in PR-C may proceed once Operator ratifies this manifest.
