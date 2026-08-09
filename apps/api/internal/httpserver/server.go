package httpserver

import (
	"log/slog"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/handler"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
)

// New builds the Gin engine with middleware and routes.
func New(logger *slog.Logger, pool *pgxpool.Pool) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)

	r := gin.New()
	r.Use(gin.Recovery())
	r.Use(middleware.RequestID())
	r.Use(middleware.AccessLog(logger))

	health := &handler.Health{Pool: pool}
	r.GET("/health", health.Live)
	r.GET("/ready", health.Ready)

	return r
}
