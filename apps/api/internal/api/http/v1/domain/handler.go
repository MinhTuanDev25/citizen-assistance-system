package domain

import (
	"errors"
	"net/http"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/response"
	"github.com/gin-gonic/gin"
)

type Handler struct {
	Repo *repository.DomainRepo
}

func NewHandler(repo *repository.DomainRepo) *Handler {
	return &Handler{Repo: repo}
}

func (h *Handler) List(c *gin.Context) {
	activeOnly := c.Query("active") == "true"
	items, err := h.Repo.List(c.Request.Context(), activeOnly)
	if err != nil {
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to list domains", err)
		return
	}
	response.OK(c, gin.H{"items": items, "count": len(items)})
}

func (h *Handler) Get(c *gin.Context) {
	item, err := h.Repo.GetByID(c.Request.Context(), c.Param("id"))
	if err != nil {
		if errors.Is(err, repository.ErrNotFound) {
			response.Fail(c, http.StatusNotFound, "NOT_FOUND", "domain not found")
			return
		}
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to get domain", err)
		return
	}
	response.OK(c, item)
}

func (h *Handler) Create(c *gin.Context) {
	var body repository.DomainCreate
	if err := c.ShouldBindJSON(&body); err != nil {
		response.FailErr(c, http.StatusBadRequest, "VALIDATION", "invalid request body", err)
		return
	}
	item, err := h.Repo.Create(c.Request.Context(), body)
	if err != nil {
		if errors.Is(err, repository.ErrConflict) {
			response.Fail(c, http.StatusConflict, "CONFLICT", "domain id already exists")
			return
		}
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to create domain", err)
		return
	}
	response.Created(c, item)
}

func (h *Handler) Update(c *gin.Context) {
	var body repository.DomainUpdate
	if err := c.ShouldBindJSON(&body); err != nil {
		response.FailErr(c, http.StatusBadRequest, "VALIDATION", "invalid request body", err)
		return
	}
	item, err := h.Repo.Update(c.Request.Context(), c.Param("id"), body)
	if err != nil {
		if errors.Is(err, repository.ErrNotFound) {
			response.Fail(c, http.StatusNotFound, "NOT_FOUND", "domain not found")
			return
		}
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to update domain", err)
		return
	}
	response.OK(c, item)
}

func (h *Handler) Delete(c *gin.Context) {
	id := c.Param("id")
	if c.Query("hard") == "true" {
		if err := h.Repo.HardDelete(c.Request.Context(), id); err != nil {
			if errors.Is(err, repository.ErrNotFound) {
				response.Fail(c, http.StatusNotFound, "NOT_FOUND", "domain not found")
				return
			}
			if errors.Is(err, repository.ErrFKRestricted) {
				response.Fail(c, http.StatusConflict, "FK_RESTRICTED", "domain is referenced by procedures/documents")
				return
			}
			response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to delete domain", err)
			return
		}
		response.OK(c, gin.H{"id": id, "deleted": true, "mode": "hard"})
		return
	}

	item, err := h.Repo.SoftDelete(c.Request.Context(), id)
	if err != nil {
		if errors.Is(err, repository.ErrNotFound) {
			response.Fail(c, http.StatusNotFound, "NOT_FOUND", "domain not found")
			return
		}
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to delete domain", err)
		return
	}
	response.OK(c, item)
}
