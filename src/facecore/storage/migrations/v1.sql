-- Phase-1B schema v1 (plan §7, S1B §4). Zero key material in SQLite:
-- tables store opaque key_id strings only; DEKs live in KeyProvider.
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS identities (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('active', 're_enrollment_required', 'deleted')),
    current_revision INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS template_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    active_template_ids TEXT NOT NULL,
    retired_template_ids TEXT NOT NULL,
    policy_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    actor TEXT NOT NULL CHECK(actor IN ('user', 'operator')),
    FOREIGN KEY(identity_id) REFERENCES identities(id)
);

CREATE TABLE IF NOT EXISTS face_templates (
    id TEXT PRIMARY KEY,
    identity_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('active', 'retired')),
    key_id TEXT NOT NULL,
    embedding_blob BLOB NOT NULL,
    embedding_nonce BLOB NOT NULL,
    exemplar_blob BLOB NOT NULL,
    exemplar_nonce BLOB NOT NULL,
    exemplar_crop_box TEXT NOT NULL,
    exemplar_landmarks TEXT NOT NULL,
    exemplar_margin REAL NOT NULL DEFAULT 0.0,
    quality_score REAL NOT NULL,
    utility_score REAL NOT NULL DEFAULT 0.0,
    additional_corroboration_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    retired_at TEXT,
    FOREIGN KEY(identity_id) REFERENCES identities(id)
);

CREATE TABLE IF NOT EXISTS candidate_templates (
    id TEXT PRIMARY KEY,
    identity_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'promoted', 'rejected', 'expired', 'generation_retired')),
    key_id TEXT NOT NULL,
    embedding_blob BLOB NOT NULL,
    embedding_nonce BLOB NOT NULL,
    exemplar_blob BLOB,
    exemplar_nonce BLOB,
    exemplar_crop_box TEXT,
    exemplar_landmarks TEXT,
    quality_score REAL NOT NULL,
    additional_corroboration_count INTEGER NOT NULL DEFAULT 0,
    evidence_log TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(identity_id) REFERENCES identities(id)
);

CREATE TABLE IF NOT EXISTS match_events (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    decision_score REAL NOT NULL,
    runner_up_score REAL,
    matched_identity_id TEXT,
    candidate_created INTEGER NOT NULL DEFAULT 0,
    actor TEXT,
    FOREIGN KEY(matched_identity_id) REFERENCES identities(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS deletion_tombstones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_type TEXT NOT NULL CHECK(target_type IN ('record', 'identity')),
    target_id TEXT NOT NULL,
    key_ids_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending_key_destruction', 'key_destroyed')),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_face_templates_identity ON face_templates(identity_id, status);
CREATE INDEX IF NOT EXISTS idx_candidate_templates_identity ON candidate_templates(identity_id, status);
CREATE INDEX IF NOT EXISTS idx_match_events_timestamp ON match_events(timestamp, sequence_number);
