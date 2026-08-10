package commune

import (
	"net/http"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/response"
	"github.com/gin-gonic/gin"
)

type Handler struct {
	Repo *repository.CommuneRepo
}

func NewHandler(repo *repository.CommuneRepo) *Handler {
	return &Handler{Repo: repo}
}

// List GET /api/v1/communes?active=true
func (h *Handler) List(c *gin.Context) {
	activeOnly := c.Query("active") == "true"
	items, err := h.Repo.List(c.Request.Context(), activeOnly)
	if err != nil {
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to list communes", err)
		return
	}
	response.OK(c, gin.H{"items": items, "count": len(items)})
}

// Get GET /api/v1/communes/:id
func (h *Handler) Get(c *gin.Context) {
	item, err := h.Repo.GetByID(c.Request.Context(), c.Param("id"))
	if err != nil {
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to get commune", err)
		return
	}
	if item == nil {
		response.Fail(c, http.StatusNotFound, "NOT_FOUND", "commune not found")
		return
	}
	response.OK(c, item)
}
