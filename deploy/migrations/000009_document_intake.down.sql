DROP INDEX IF EXISTS ux_documents_xa_checksum;

ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_checksum_sha256_check;
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_file_size_check;
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_mime_type_check;

ALTER TABLE documents DROP COLUMN IF EXISTS file_size_bytes;
ALTER TABLE documents DROP COLUMN IF EXISTS mime_type;
