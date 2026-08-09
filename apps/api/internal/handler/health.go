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

// Live is process liveness — does not check dependencies.
// GET /health
func (h *Health) Live(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "OK",
	})
}

// Ready checks database connectivity.
// GET /ready
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

	c.JSON(http.StatusOK, gin.H{
		"status": "READY",
		"checks": gin.H{
			"database": "OK",
		},
	})
}
