package procedure

import (
	"errors"
	"net/http"
	"strings"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/response"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

type Handler struct {
	Repo        *repository.ProcedureRepo
	DefaultXaID string
}

func NewHandler(repo *repository.ProcedureRepo, defaultXaID string) *Handler {
	return &Handler{Repo: repo, DefaultXaID: defaultXaID}
}

func (h *Handler) resolveXaID(c *gin.Context) string {
	xa := strings.TrimSpace(c.Query("xa_id"))
	if xa == "" {
		xa = h.DefaultXaID
	}
	return xa
}

// ProcedureInDomain is a procedure row inside a domain group (list response).
type ProcedureInDomain struct {
	ID                  string `json:"id"`
	ProcedureCode       string `json:"procedure_code"`
	Name                string `json:"name"`
	XaID                string `json:"xa_id"`
	ActiveVersionID     string `json:"active_version_id"`
	ActiveVersion       string `json:"active_version"`
	ActiveVersionStatus string `json:"active_version_status"`
}

// DomainGroup groups procedures under a domain for catalog listing.
type DomainGroup struct {
	DomainID   string              `json:"domain_id"`
	DomainName string              `json:"domain_name"`
	Procedures []ProcedureInDomain `json:"procedures"`
	Count      int                 `json:"count"`
}

// ListData is the data payload for GET /procedures.
type ListData struct {
	XaID    string        `json:"xa_id"`
	Domains []DomainGroup `json:"domains"`
	Count   int           `json:"count"`
}

// List godoc
//
//	@Summary		List ACTIVE procedures grouped by domain
//	@Tags			procedures
//	@Produce		json
//	@Param			xa_id		query		string	false	"Commune id (default from config)"
//	@Param			domain_id	query		string	false	"Filter by domain id"
//	@Success		200			{object}	response.Envelope
//	@Failure		400			{object}	response.Envelope
//	@Failure		500			{object}	response.Envelope
//	@Router			/api/v1/procedures [get]
func (h *Handler) List(c *gin.Context) {
	xaID := h.resolveXaID(c)
	if xaID == "" {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "xa_id is required")
		return
	}
	domainID := strings.TrimSpace(c.Query("domain_id"))

	items, err := h.Repo.ListActive(c.Request.Context(), xaID, domainID)
	if err != nil {
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to list procedures", err)
		return
	}

	groups := make([]DomainGroup, 0)
	indexByDomain := map[string]int{}
	for _, item := range items {
		idx, ok := indexByDomain[item.DomainID]
		if !ok {
			idx = len(groups)
			indexByDomain[item.DomainID] = idx
			groups = append(groups, DomainGroup{
				DomainID:   item.DomainID,
				DomainName: item.DomainName,
				Procedures: make([]ProcedureInDomain, 0),
			})
		}
		groups[idx].Procedures = append(groups[idx].Procedures, ProcedureInDomain{
			ID:                  item.ID.String(),
			ProcedureCode:       item.ProcedureCode,
			Name:                item.Name,
			XaID:                item.XaID,
			ActiveVersionID:     item.ActiveVersionID.String(),
			ActiveVersion:       item.ActiveVersion,
			ActiveVersionStatus: item.ActiveVersionStatus,
		})
		groups[idx].Count = len(groups[idx].Procedures)
	}

	response.OK(c, ListData{
		XaID:    xaID,
		Domains: groups,
		Count:   len(items),
	})
}

// GetByCode godoc
//
//	@Summary		Resolve procedure by code
//	@Tags			procedures
//	@Produce		json
//	@Param			code	path		string	true	"procedure_code"
//	@Param			xa_id	query		string	false	"Commune id (default from config)"
//	@Success		200		{object}	response.Envelope
//	@Failure		400		{object}	response.Envelope
//	@Failure		404		{object}	response.Envelope
//	@Failure		500		{object}	response.Envelope
//	@Router			/api/v1/procedures/by-code/{code} [get]
func (h *Handler) GetByCode(c *gin.Context) {
	code := strings.TrimSpace(c.Param("code"))
	if code == "" {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "procedure code is required")
		return
	}
	xaID := h.resolveXaID(c)
	if xaID == "" {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "xa_id is required")
		return
	}

	item, err := h.Repo.GetByCode(c.Request.Context(), xaID, code)
	if err != nil {
		if errors.Is(err, repository.ErrNotFound) {
			response.Fail(c, http.StatusNotFound, "NOT_FOUND", "procedure not found")
			return
		}
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to get procedure by code", err)
		return
	}
	response.OK(c, item)
}

// GetActiveVersion godoc
//
//	@Summary		Get ACTIVE version + definition
//	@Tags			procedures
//	@Produce		json
//	@Param			id	path		string	true	"Procedure UUID"
//	@Success		200	{object}	response.Envelope
//	@Failure		400	{object}	response.Envelope
//	@Failure		404	{object}	response.Envelope
//	@Failure		500	{object}	response.Envelope
//	@Router			/api/v1/procedures/{id}/active-version [get]
func (h *Handler) GetActiveVersion(c *gin.Context) {
	id, err := uuid.Parse(strings.TrimSpace(c.Param("id")))
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid procedure id")
		return
	}

	item, err := h.Repo.GetActiveVersion(c.Request.Context(), id)
	if err != nil {
		if errors.Is(err, repository.ErrNotFound) {
			response.Fail(c, http.StatusNotFound, "NOT_FOUND", "active procedure version not found")
			return
		}
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to get active version", err)
		return
	}
	response.OK(c, item)
}
