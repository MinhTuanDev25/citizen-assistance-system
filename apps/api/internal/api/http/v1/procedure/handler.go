package procedure

import (
	"errors"
	"net/http"
	"strings"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/response"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

type Handler struct {
	Repo             *repository.ProcedureRepo
	DefaultXaID      string
	CitizenDomainIDs []string
}

func NewHandler(repo *repository.ProcedureRepo, defaultXaID string, citizenDomainIDs []string) *Handler {
	ids := append([]string{}, citizenDomainIDs...)
	if len(ids) == 0 {
		ids = []string{"ho_tich_chung_thuc"}
	}
	return &Handler{Repo: repo, DefaultXaID: defaultXaID, CitizenDomainIDs: ids}
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
	XaID             string        `json:"xa_id"`
	CitizenDomainIDs []string      `json:"citizen_domain_ids,omitempty"`
	Domains          []DomainGroup `json:"domains"`
	Count            int           `json:"count"`
}

// List godoc
//
//	@Summary		List ACTIVE procedures grouped by domain
//	@Description	Default / citizen=true: public CitizenDomainIDs only. citizen=false: full ACTIVE catalog — ADMIN JWT required (not a public privilege flag). Invalid citizen values → 400.
//	@Tags			procedures
//	@Produce		json
//	@Param			xa_id			query		string	false	"Commune id (default from config)"
//	@Param			domain_id		query		string	false	"Filter by domain id"
//	@Param			citizen			query		bool	false	"true/omit: CitizenDomainIDs; false: admin full catalog"
//	@Param			Authorization	header		string	false	"Bearer ADMIN JWT required when citizen=false"
//	@Success		200				{object}	response.Envelope
//	@Failure		400				{object}	response.Envelope
//	@Failure		401				{object}	response.Envelope
//	@Failure		403				{object}	response.Envelope
//	@Failure		500				{object}	response.Envelope
//	@Router			/api/v1/procedures [get]
func (h *Handler) List(c *gin.Context) {
	xaID := h.resolveXaID(c)
	if xaID == "" {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "xa_id is required")
		return
	}
	domainID := strings.TrimSpace(c.Query("domain_id"))
	citizenOnly, ok := parseCitizenQuery(c)
	if !ok {
		return
	}

	items, err := h.Repo.ListActive(c.Request.Context(), xaID, domainID)
	if err != nil {
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to list procedures", err)
		return
	}

	allowed := map[string]struct{}{}
	for _, id := range h.CitizenDomainIDs {
		allowed[id] = struct{}{}
	}

	groups := make([]DomainGroup, 0)
	indexByDomain := map[string]int{}
	total := 0
	for _, item := range items {
		if citizenOnly {
			if _, ok := allowed[item.DomainID]; !ok {
				continue
			}
		}
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
		total++
	}

	out := ListData{
		XaID:    xaID,
		Domains: groups,
		Count:   total,
	}
	if citizenOnly {
		out.CitizenDomainIDs = append([]string{}, h.CitizenDomainIDs...)
	}
	response.OK(c, out)
}

// parseCitizenQuery returns (citizenOnly, ok). On failure it already wrote the HTTP error.
// citizen=false is NOT a public privilege: ADMIN JWT is required.
func parseCitizenQuery(c *gin.Context) (citizenOnly bool, ok bool) {
	raw := strings.TrimSpace(c.Query("citizen"))
	if raw == "" {
		return true, true
	}
	switch {
	case raw == "1" || strings.EqualFold(raw, "true"):
		return true, true
	case raw == "0" || strings.EqualFold(raw, "false"):
		role := middleware.UserRole(c)
		if role == "" {
			response.Fail(c, http.StatusUnauthorized, "UNAUTHORIZED", "admin Bearer token required for full catalog")
			return false, false
		}
		if role != auth.RoleAdmin {
			response.Fail(c, http.StatusForbidden, "FORBIDDEN", "admin role required for full catalog")
			return false, false
		}
		return false, true
	default:
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "citizen must be true or false")
		return false, false
	}
}

// domainAllowed reports whether the caller may read a procedure in domainID.
// ADMIN JWT → any domain; guest/citizen → CitizenDomainIDs only.
func (h *Handler) domainAllowed(c *gin.Context, domainID string) bool {
	if middleware.UserRole(c) == auth.RoleAdmin {
		return true
	}
	for _, id := range h.CitizenDomainIDs {
		if id == domainID {
			return true
		}
	}
	return false
}

func (h *Handler) denyOutsideCitizenScope(c *gin.Context) {
	role := middleware.UserRole(c)
	if role != "" && role != auth.RoleAdmin {
		response.Fail(c, http.StatusForbidden, "FORBIDDEN", "procedure outside citizen catalog scope")
		return
	}
	// Guest: do not leak that the procedure exists outside public catalog.
	response.Fail(c, http.StatusNotFound, "NOT_FOUND", "procedure not found")
}

// GetByCode godoc
//
//	@Summary		Resolve procedure by code
//	@Description	Guest or citizen callers may only read procedures whose domain_id is in CitizenDomainIDs. ADMIN Bearer JWT is required to read procedures outside that public/citizen scope. Guest requests for out-of-scope codes receive 404; citizen JWT receives 403.
//	@Tags			procedures
//	@Produce		json
//	@Param			code			path		string	true	"procedure_code"
//	@Param			xa_id			query		string	false	"Commune id (default from config)"
//	@Param			Authorization	header		string	false	"Bearer ADMIN JWT required for procedures outside CitizenDomainIDs"
//	@Success		200				{object}	response.Envelope
//	@Failure		400				{object}	response.Envelope	"validation error"
//	@Failure		403				{object}	response.Envelope	"citizen JWT requested a procedure outside CitizenDomainIDs"
//	@Failure		404				{object}	response.Envelope	"not found or guest out-of-scope"
//	@Failure		500				{object}	response.Envelope
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
	if !h.domainAllowed(c, item.DomainID) {
		h.denyOutsideCitizenScope(c)
		return
	}
	response.OK(c, item)
}

// GetActiveVersion godoc
//
//	@Summary		Get ACTIVE version + definition
//	@Description	Guest or citizen callers may only read ACTIVE definitions for procedures whose domain_id is in CitizenDomainIDs. ADMIN Bearer JWT is required for procedures outside that public/citizen scope. Guest out-of-scope → 404; citizen JWT out-of-scope → 403.
//	@Tags			procedures
//	@Produce		json
//	@Param			id				path		string	true	"Procedure UUID"
//	@Param			Authorization	header		string	false	"Bearer ADMIN JWT required for procedures outside CitizenDomainIDs"
//	@Success		200				{object}	response.Envelope
//	@Failure		400				{object}	response.Envelope	"validation error"
//	@Failure		403				{object}	response.Envelope	"citizen JWT requested a procedure outside CitizenDomainIDs"
//	@Failure		404				{object}	response.Envelope	"not found or guest out-of-scope"
//	@Failure		500				{object}	response.Envelope
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
	if !h.domainAllowed(c, item.DomainID) {
		h.denyOutsideCitizenScope(c)
		return
	}
	response.OK(c, item)
}
