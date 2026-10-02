package documentapi

import (
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/document"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/index"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/response"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

type Handler struct {
	Svc   *document.Service
	Index *index.Service
}

func NewHandler(svc *document.Service) *Handler {
	return &Handler{Svc: svc}
}

// Upload godoc
//
//	@Summary		Upload an admin PDF
//	@Description	Admin JWT required. Stores one application/pdf. Client xa_id is ignored.
//	@Tags			admin-documents
//	@Accept			mpfd
//	@Produce		json
//	@Param			file				formData	file	true	"PDF file"
//	@Param			title				formData	string	true	"Title"
//	@Param			domain_id			formData	string	true	"Active domain id"
//	@Param			document_number		formData	string	false	"Document number"
//	@Param			issuer				formData	string	false	"Issuer"
//	@Param			effective_date		formData	string	false	"Effective date YYYY-MM-DD"
//	@Param			expire_date			formData	string	false	"Expire date YYYY-MM-DD"
//	@Param			issued_date			formData	string	false	"Issued date YYYY-MM-DD"
//	@Success		201					{object}	response.Envelope
//	@Failure		401					{object}	response.Envelope
//	@Failure		403					{object}	response.Envelope
//	@Failure		409					{object}	response.Envelope
//	@Failure		413					{object}	response.Envelope
//	@Failure		422					{object}	response.Envelope
//	@Failure		500					{object}	response.Envelope
//	@Security		BearerAuth
//	@Router			/api/v1/admin/documents [post]
func (h *Handler) Upload(c *gin.Context) {
	if c.Request.ContentLength > h.Svc.MaxBytes+middleware.DocumentUploadMultipartOverhead {
		response.Fail(c, http.StatusRequestEntityTooLarge, "PAYLOAD_TOO_LARGE", "request body exceeds size limit")
		return
	}
	actor, ok := middleware.UserID(c)
	if !ok {
		response.Fail(c, http.StatusUnauthorized, "UNAUTHORIZED", "admin role required")
		return
	}
	reqID, err := uuid.Parse(middleware.GetRequestID(c))
	if err != nil {
		reqID = uuid.New()
	}
	reader, err := c.Request.MultipartReader()
	if err != nil {
		response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", "expected a multipart upload")
		return
	}
	meta := document.UploadMeta{ActorUserID: actor, RequestID: reqID}
	var prep *document.Prepared
	files := 0
	for {
		part, err := reader.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			if prep != nil {
				prep.Cleanup()
			}
			response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", "invalid multipart body")
			return
		}
		if part.FileName() != "" || part.FormName() == "file" {
			if unsafeDisposition(part.Header.Get("Content-Disposition")) {
				part.Close()
				if prep != nil {
					prep.Cleanup()
				}
				response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", "filename is not allowed")
				return
			}
			files++
			if files > 1 {
				part.Close()
				if prep != nil {
					prep.Cleanup()
				}
				response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", "exactly one file is required")
				return
			}
			prepared, err := h.Svc.Prepare(part, part.FileName(), part.Header.Get("Content-Type"), partLength(part))
			part.Close()
			if err != nil {
				writeUploadErr(c, err)
				return
			}
			prep = prepared
			continue
		}
		val, err := readField(part)
		part.Close()
		if err != nil {
			if prep != nil {
				prep.Cleanup()
			}
			response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", "metadata is too long")
			return
		}
		switch part.FormName() {
		case "title":
			meta.Title = val
		case "domain_id":
			meta.DomainID = val
		case "document_number":
			meta.DocumentNumber = val
		case "issuer":
			meta.Issuer = val
		case "effective_date":
			meta.EffectiveDate = val
		case "expire_date":
			meta.ExpireDate = val
		case "issued_date":
			meta.IssuedDate = val
		case "xa_id":
			// Server commune is authoritative. Ignore the client value.
		default:
			if prep != nil {
				prep.Cleanup()
			}
			response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", "unknown form field")
			return
		}
	}
	if files != 1 || prep == nil {
		response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", "exactly one file is required")
		return
	}
	doc, err := h.Svc.Save(c.Request.Context(), prep, meta)
	if err != nil {
		writeUploadErr(c, err)
		return
	}
	c.Set("log_action", "DOCUMENT_UPLOADED")
	response.Created(c, doc)
}

// List godoc
//
//	@Summary	List admin documents for this commune
//	@Tags		admin-documents
//	@Produce	json
//	@Param		domain_id			query	string	false	"Domain id"
//	@Param		processing_status	query	string	false	"UPLOADED, PROCESSING, PROCESSED, READY, or FAILED"
//	@Param		validity_status		query	string	false	"PENDING, VALID, EXPIRED, or SUPERSEDED"
//	@Param		limit				query	int		false	"Page size"
//	@Param		offset				query	int		false	"Page offset"
//	@Success	200					{object}	response.Envelope
//	@Failure	401					{object}	response.Envelope
//	@Failure	403					{object}	response.Envelope
//	@Failure	422					{object}	response.Envelope
//	@Failure	500					{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/documents [get]
func (h *Handler) List(c *gin.Context) {
	limit, _ := strconv.Atoi(c.Query("limit"))
	offset, err := strconv.Atoi(c.DefaultQuery("offset", "0"))
	if err != nil {
		response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", "offset is invalid")
		return
	}
	f := document.ListFilter{
		DomainID:         c.Query("domain_id"),
		ProcessingStatus: c.Query("processing_status"),
		ValidityStatus:   c.Query("validity_status"),
		Limit:            limit,
		Offset:           offset,
	}
	if !allowedStatus(f.ProcessingStatus, "UPLOADED", "PROCESSING", "PROCESSED", "READY", "FAILED") ||
		!allowedStatus(f.ValidityStatus, "PENDING", "VALID", "EXPIRED", "SUPERSEDED") {
		response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", "status filter is invalid")
		return
	}
	out, err := h.Svc.List(c.Request.Context(), f)
	if err != nil {
		if errors.Is(err, document.ErrValidation) {
			response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", document.SafeMessage(err))
			return
		}
		response.Fail(c, http.StatusInternalServerError, "INTERNAL", "failed to list documents")
		return
	}
	out.MaxBytes = h.Svc.MaxBytes
	response.OK(c, out)
}

// Get godoc
//
//	@Summary	Get one admin document
//	@Tags		admin-documents
//	@Produce	json
//	@Param		id	path		string	true	"Document id"
//	@Success	200	{object}	response.Envelope
//	@Failure	401	{object}	response.Envelope
//	@Failure	403	{object}	response.Envelope
//	@Failure	404	{object}	response.Envelope
//	@Failure	500	{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/documents/{id} [get]
func (h *Handler) Get(c *gin.Context) {
	id, err := uuid.Parse(c.Param("id"))
	if err != nil {
		response.Fail(c, http.StatusNotFound, "NOT_FOUND", "document not found")
		return
	}
	doc, err := h.Svc.Get(c.Request.Context(), id)
	if err != nil {
		if errors.Is(err, document.ErrNotFound) {
			response.Fail(c, http.StatusNotFound, "NOT_FOUND", "document not found")
			return
		}
		response.Fail(c, http.StatusInternalServerError, "INTERNAL", "failed to load document")
		return
	}
	response.OK(c, doc)
}

// Content godoc
//
//	@Summary	Download the stored PDF
//	@Tags		admin-documents
//	@Produce	application/pdf
//	@Param		id	path		string	true	"Document id"
//	@Success	200	{file}		file
//	@Failure	401	{object}	response.Envelope
//	@Failure	403	{object}	response.Envelope
//	@Failure	404	{object}	response.Envelope
//	@Failure	500	{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/documents/{id}/content [get]
func (h *Handler) Content(c *gin.Context) {
	id, err := uuid.Parse(c.Param("id"))
	if err != nil {
		response.Fail(c, http.StatusNotFound, "NOT_FOUND", "document not found")
		return
	}
	doc, rc, err := h.Svc.Open(c.Request.Context(), id)
	if err != nil {
		if errors.Is(err, document.ErrNotFound) || errors.Is(err, document.ErrObjectMissing) {
			response.Fail(c, http.StatusNotFound, "NOT_FOUND", "document content unavailable")
			return
		}
		response.Fail(c, http.StatusInternalServerError, "INTERNAL", "failed to read document")
		return
	}
	defer rc.Close()
	name := strings.ReplaceAll(doc.Filename, `"`, "")
	c.Header("Content-Type", document.MimePDF)
	c.Header("X-Content-Type-Options", "nosniff")
	c.Header("Content-Disposition", `attachment; filename="`+name+`"`)
	c.Status(http.StatusOK)
	if _, err := io.Copy(c.Writer, rc); err != nil {
		return
	}
}

func writeUploadErr(c *gin.Context, err error) {
	switch {
	case errors.Is(err, document.ErrTooLarge):
		response.Fail(c, http.StatusRequestEntityTooLarge, "PAYLOAD_TOO_LARGE", "request body exceeds size limit")
	case errors.Is(err, document.ErrDuplicate):
		response.Fail(c, http.StatusConflict, "DOCUMENT_DUPLICATE", "a document with this content already exists")
	case errors.Is(err, document.ErrDomain):
		response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", "domain is not available")
	case errors.Is(err, document.ErrBadPDF), errors.Is(err, document.ErrValidation):
		response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", document.SafeMessage(err))
	default:
		response.Fail(c, http.StatusInternalServerError, "INTERNAL", "upload failed")
	}
}

func readField(part *multipart.Part) (string, error) {
	buf, err := io.ReadAll(io.LimitReader(part, 2049))
	if err != nil {
		return "", err
	}
	if len(buf) > 2048 {
		return "", errors.New("field too long")
	}
	return string(buf), nil
}

func partLength(part *multipart.Part) int64 {
	return -1
}

func unsafeDisposition(v string) bool {
	return strings.ContainsRune(v, 0) || strings.Contains(v, "..") || strings.Contains(v, "/") || strings.Contains(v, `\`)
}

func allowedStatus(v string, allowed ...string) bool {
	if v == "" {
		return true
	}
	for _, a := range allowed {
		if v == a {
			return true
		}
	}
	return false
}
