package document

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"log/slog"
	"os"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/storage"
	"github.com/google/uuid"
)

var pdfMagic = []byte("%PDF-")

const objectCleanupTimeout = 10 * time.Second

// cleanupContext is detached from the request so a canceled caller still
// deletes the object that was already stored.
func cleanupContext() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), objectCleanupTimeout)
}

// Repo is the persistence port. The PostgreSQL implementation lives in repository.
type Repo interface {
	Insert(ctx context.Context, doc Document, requestID uuid.UUID) error
	Get(ctx context.Context, xaID string, id uuid.UUID) (Document, error)
	List(ctx context.Context, f ListFilter) (ListResult, error)
	DomainActive(ctx context.Context, id string) error
}

type Service struct {
	Repo     Repo
	Objects  storage.ObjectStore
	Bucket   string
	XAID     string
	MaxBytes int64
	TempDir  string
	Logger   *slog.Logger
}

// Prepared is a temp file plus the checksum computed while it was written.
type Prepared struct {
	file     *os.File
	path     string
	sum      string
	size     int64
	filename string
}

func (p *Prepared) Cleanup() {
	if p == nil {
		return
	}
	if p.file != nil {
		_ = p.file.Close()
		p.file = nil
	}
	if p.path != "" {
		_ = os.Remove(p.path)
		p.path = ""
	}
}

// Prepare streams the PDF to a temp file, hashing and sizing as it goes.
// The temp file is removed by Cleanup. The PDF is not held in memory.
func (s *Service) Prepare(r io.Reader, filename, contentType string, declared int64) (*Prepared, error) {
	if s.MaxBytes <= 0 {
		return nil, ErrStorage
	}
	if err := cleanFilename(filename); err != nil {
		return nil, err
	}
	if !allowedMIME(contentType) {
		return nil, errValidation("content type must be application/pdf")
	}
	if declared > s.MaxBytes {
		return nil, ErrTooLarge
	}
	dir := s.TempDir
	if dir == "" {
		dir = os.TempDir()
	}
	f, err := os.CreateTemp(dir, "cas-upload-*")
	if err != nil {
		return nil, ErrStorage
	}
	prep := &Prepared{file: f, path: f.Name(), filename: filename}
	h := sha256.New()
	mw := io.MultiWriter(f, h)
	head := make([]byte, len(pdfMagic))
	n, err := io.ReadFull(r, head)
	if err != nil || n < len(pdfMagic) || !bytes.Equal(head[:n], pdfMagic) {
		prep.Cleanup()
		if declared == 0 && (err == io.EOF || err == io.ErrUnexpectedEOF) {
			return nil, errValidation("file is empty or not a pdf")
		}
		return nil, ErrBadPDF
	}
	if _, err := mw.Write(head); err != nil {
		prep.Cleanup()
		return nil, ErrStorage
	}
	restLimit := s.MaxBytes - int64(len(pdfMagic))
	rest, err := io.Copy(mw, io.LimitReader(r, restLimit+1))
	if err != nil {
		prep.Cleanup()
		return nil, ErrStorage
	}
	if rest > restLimit {
		prep.Cleanup()
		return nil, ErrTooLarge
	}
	size := int64(len(pdfMagic)) + rest
	prep.size = size
	prep.sum = hex.EncodeToString(h.Sum(nil))
	return prep, nil
}

func (s *Service) Save(ctx context.Context, prep *Prepared, meta UploadMeta) (doc Document, err error) {
	defer prep.Cleanup()
	title, err := cleanMeta(meta.Title, true)
	if err != nil {
		return Document{}, err
	}
	domainID, err := cleanDomainID(meta.DomainID)
	if err != nil {
		return Document{}, ErrDomain
	}
	number, err := cleanMeta(meta.DocumentNumber, false)
	if err != nil {
		return Document{}, err
	}
	issuer, err := cleanMeta(meta.Issuer, false)
	if err != nil {
		return Document{}, err
	}
	effective, err := cleanDate(meta.EffectiveDate)
	if err != nil {
		return Document{}, err
	}
	expire, err := cleanDate(meta.ExpireDate)
	if err != nil {
		return Document{}, err
	}
	issued, err := cleanDate(meta.IssuedDate)
	if err != nil {
		return Document{}, err
	}
	if effective != "" && expire != "" && expire < effective {
		return Document{}, errValidation("expire_date is before effective_date")
	}
	if err := s.Repo.DomainActive(ctx, domainID); err != nil {
		if errors.Is(err, ErrDomain) || errors.Is(err, ErrNotFound) {
			return Document{}, ErrDomain
		}
		return Document{}, err
	}
	id := uuid.New()
	key := storage.ObjectKey(s.XAID, id.String(), prep.sum)
	if _, err := prep.file.Seek(0, io.SeekStart); err != nil {
		return Document{}, ErrStorage
	}
	if err := s.Objects.Put(ctx, key, prep.file, prep.size, MimePDF); err != nil {
		return Document{}, ErrStorage
	}
	doc = Document{
		ID:               id,
		XAID:             s.XAID,
		DomainID:         domainID,
		Title:            title,
		Filename:         prep.filename,
		Checksum:         prep.sum,
		MimeType:         MimePDF,
		FileSizeBytes:    prep.size,
		ProcessingStatus: StatusUploaded,
		ValidityStatus:   ValidityPending,
		UploadedBy:       meta.ActorUserID,
		StorageURI:       storage.InternalURI(s.Bucket, key),
		ObjectKey:        key,
	}
	if number != "" {
		doc.DocumentNumber = &number
	}
	if issuer != "" {
		doc.Issuer = &issuer
	}
	if effective != "" {
		doc.EffectiveDate = &effective
	}
	if expire != "" {
		doc.ExpireDate = &expire
	}
	if issued != "" {
		doc.IssuedDate = &issued
	}
	if err := s.Repo.Insert(ctx, doc, meta.RequestID); err != nil {
		cleanCtx, cancel := cleanupContext()
		delErr := s.Objects.Delete(cleanCtx, key)
		cancel()
		if delErr != nil {
			s.log().Warn("document object cleanup failed", "document_id", id.String())
		}
		if errors.Is(err, ErrDuplicate) {
			return Document{}, ErrDuplicate
		}
		return Document{}, err
	}
	return doc, nil
}

func (s *Service) List(ctx context.Context, f ListFilter) (ListResult, error) {
	f.XAID = s.XAID
	if f.Limit <= 0 {
		f.Limit = 20
	}
	if f.Limit > 100 {
		f.Limit = 100
	}
	if f.Offset < 0 {
		return ListResult{}, errValidation("offset is invalid")
	}
	return s.Repo.List(ctx, f)
}

func (s *Service) Get(ctx context.Context, id uuid.UUID) (Document, error) {
	return s.Repo.Get(ctx, s.XAID, id)
}

func (s *Service) Open(ctx context.Context, id uuid.UUID) (Document, io.ReadCloser, error) {
	doc, err := s.Get(ctx, id)
	if err != nil {
		return Document{}, nil, err
	}
	key := storage.ObjectKey(s.XAID, doc.ID.String(), doc.Checksum)
	rc, err := s.Objects.Get(ctx, key)
	if err != nil {
		if errors.Is(err, storage.ErrNotFound) {
			return Document{}, nil, ErrObjectMissing
		}
		return Document{}, nil, ErrStorage
	}
	return doc, rc, nil
}

func (s *Service) log() *slog.Logger {
	if s.Logger != nil {
		return s.Logger
	}
	return slog.Default()
}
