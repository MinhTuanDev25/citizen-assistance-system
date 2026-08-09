-- 000001_init_schema.down.sql

DROP TABLE IF EXISTS audit_logs;
DROP TABLE IF EXISTS knowledge_chunks;
DROP TABLE IF EXISTS session_slot_states;
DROP TABLE IF EXISTS conversation_messages;
DROP TABLE IF EXISTS conversation_sessions;
DROP TABLE IF EXISTS procedure_version_documents;

ALTER TABLE IF EXISTS procedures
    DROP CONSTRAINT IF EXISTS fk_procedures_active_version;

DROP TABLE IF EXISTS procedure_versions;
DROP TABLE IF EXISTS procedure_drafts;
DROP TABLE IF EXISTS documents;
DROP TABLE IF EXISTS procedures;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS domains;
DROP TABLE IF EXISTS communes;

DROP EXTENSION IF EXISTS vector;
DROP EXTENSION IF EXISTS pgcrypto;
