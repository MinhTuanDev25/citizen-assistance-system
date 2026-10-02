package v1

import (
	authapi "github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/commune"
	documentapi "github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/document"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/domain"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/procedure"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/session"
	"github.com/gin-gonic/gin"
)

// MapRoutes registers /api/v1 routes.
func MapRoutes(
	api *gin.RouterGroup,
	communeHandler *commune.Handler,
	domainHandler *domain.Handler,
	procedureHandler *procedure.Handler,
	sessionHandler *session.Handler,
	authHandler *authapi.Handler,
	documentHandler *documentapi.Handler,
	optionalJWT gin.HandlerFunc,
	requireJWT gin.HandlerFunc,
	requireAdmin gin.HandlerFunc,
) {
	v1 := api.Group("/v1")
	{
		authGroup := v1.Group("/auth")
		{
			authGroup.POST("/register", authHandler.Register)
			authGroup.POST("/login", authHandler.Login)
			authGroup.GET("/me", requireJWT, authHandler.Me)
			authGroup.POST("/logout", requireJWT, authHandler.Logout)
		}

		// Public catalog (citizen/guest)
		communes := v1.Group("/communes")
		{
			communes.GET("", communeHandler.List)
			communes.GET("/:id", communeHandler.Get)
		}

		domains := v1.Group("/domains")
		{
			domains.GET("", domainHandler.List)
			domains.GET("/:id", domainHandler.Get)
			// Admin write
			domains.POST("", requireAdmin, domainHandler.Create)
			domains.PUT("/:id", requireAdmin, domainHandler.Update)
			domains.DELETE("/:id", requireAdmin, domainHandler.Delete)
		}

		procedures := v1.Group("/procedures")
		procedures.Use(optionalJWT)
		{
			procedures.GET("", procedureHandler.List)
			procedures.GET("/by-code/:code", procedureHandler.GetByCode)
			procedures.GET("/:id/active-version", procedureHandler.GetActiveVersion)
		}

		// Guest or logged-in citizen/admin
		sessions := v1.Group("/sessions")
		sessions.Use(optionalJWT)
		{
			sessions.POST("", sessionHandler.Create)
			sessions.GET("/:sessionId/messages", sessionHandler.ListMessages)
			sessions.POST("/:sessionId/messages", sessionHandler.CreateMessage)
			sessions.POST("/:sessionId/turns", sessionHandler.Turn)
		}

		if documentHandler != nil {
			docs := v1.Group("/admin/documents")
			docs.Use(requireAdmin)
			{
				docs.POST("", documentHandler.Upload)
				docs.GET("", documentHandler.List)
				if documentHandler.Index != nil {
					docs.GET("/link-targets", documentHandler.LinkTargets)
					docs.GET("/index-metrics", documentHandler.IndexMetrics)
					docs.GET("/:id/links", documentHandler.ListLinks)
					docs.POST("/:id/links", documentHandler.Link)
					docs.POST("/:id/links/:versionId/index/retry", documentHandler.RetryIndex)
					docs.POST("/:id/links/:versionId/index/reindex", documentHandler.Reindex)
					docs.POST("/:id/links/:versionId/index", documentHandler.RequestIndex)
					docs.DELETE("/:id/links/:versionId", documentHandler.Unlink)
				}
				docs.GET("/:id/content", documentHandler.Content)
				docs.GET("/:id", documentHandler.Get)
			}
		}
	}
}
