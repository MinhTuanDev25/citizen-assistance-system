package httpserver

import (
	"log/slog"
	"time"

	v1 "github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1"
	authapi "github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/commune"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/domain"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/procedure"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/session"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/config"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/handler"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

func New(logger *slog.Logger, pool *pgxpool.Pool, cfg config.Config) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)

	r := gin.New()
	r.Use(gin.Recovery())
	r.Use(middleware.RequestID())
	r.Use(middleware.ApiLog(logger))
	r.Use(middleware.CORS())

	health := &handler.Health{Pool: pool, Timeout: 2 * time.Second}
	r.GET("/health", health.Live)
	r.GET("/ready", health.Ready)
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	tokens := &auth.TokenService{
		Secret: []byte(cfg.JWTSecret),
		TTL:    time.Duration(cfg.JWTExpireHours) * time.Hour,
	}

	communeRepo := &repository.CommuneRepo{Pool: pool}
	userRepo := &repository.UserRepo{Pool: pool}
	communeHandler := commune.NewHandler(communeRepo)
	domainHandler := domain.NewHandler(&repository.DomainRepo{Pool: pool})
	procedureHandler := procedure.NewHandler(&repository.ProcedureRepo{Pool: pool}, cfg.XAID)
	sessionHandler := session.NewHandler(&repository.SessionRepo{Pool: pool}, communeRepo)
	authHandler := authapi.NewHandler(userRepo, tokens)

	api := r.Group("/api")
	v1.MapRoutes(
		api,
		communeHandler,
		domainHandler,
		procedureHandler,
		sessionHandler,
		authHandler,
		middleware.OptionalJWT(tokens),
		middleware.RequireJWT(tokens),
		middleware.RequireAdmin(tokens),
	)

	return r
}
