-- P3.1 admin document intake. Adds the columns an uploaded PDF needs
-- and a commune-scoped checksum unique key. Does not change 000001.

ALTER TABLE documents
    ADD COLUMN mime_type text,
    ADD COLUMN file_size_bytes bigint;

UPDATE documents
SET mime_type = 'application/pdf'
WHERE mime_type IS NULL;

UPDATE documents
SET file_size_bytes = 0
WHERE file_size_bytes IS NULL;

ALTER TABLE documents
    ALTER COLUMN mime_type SET NOT NULL,
    ALTER COLUMN file_size_bytes SET NOT NULL;

ALTER TABLE documents
    ADD CONSTRAINT documents_mime_type_check CHECK (mime_type = 'application/pdf'),
    ADD CONSTRAINT documents_file_size_check CHECK (file_size_bytes >= 0),
    ADD CONSTRAINT documents_checksum_sha256_check CHECK (checksum ~ '^[0-9a-f]{64}$');

CREATE UNIQUE INDEX ux_documents_xa_checksum ON documents (xa_id, checksum);
