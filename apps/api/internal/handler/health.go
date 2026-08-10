package handler

import (
	"context"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Health handles liveness and readiness probes.
type Health struct {
	Pool    *pgxpool.Pool
	Timeout time.Duration
}

type liveResponse struct {
	Status string `json:"status" example:"OK"`
}

type readyResponse struct {
	Status string            `json:"status" example:"READY"`
	Checks map[string]string `json:"checks"`
}

// Live godoc
//
//	@Summary		Liveness probe
//	@Tags			health
//	@Produce		json
//	@Success		200	{object}	liveResponse
//	@Router			/health [get]
func (h *Health) Live(c *gin.Context) {
	c.JSON(http.StatusOK, liveResponse{Status: "OK"})
}

// Ready godoc
//
//	@Summary		Readiness probe
//	@Tags			health
//	@Produce		json
//	@Success		200	{object}	readyResponse
//	@Failure		503	{object}	map[string]string
//	@Router			/ready [get]
func (h *Health) Ready(c *gin.Context) {
	timeout := h.Timeout
	if timeout <= 0 {
		timeout = 2 * time.Second
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), timeout)
	defer cancel()

	if err := h.Pool.Ping(ctx); err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status": "NOT_READY",
			"error":  "database_unavailable",
		})
		return
	}

	c.JSON(http.StatusOK, readyResponse{
		Status: "READY",
		Checks: map[string]string{"database": "OK"},
	})
}
