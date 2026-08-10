package v1

import (
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/commune"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/domain"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/procedure"
	"github.com/gin-gonic/gin"
)

// MapRoutes registers /api/v1 routes.
func MapRoutes(
	api *gin.RouterGroup,
	communeHandler *commune.Handler,
	domainHandler *domain.Handler,
	procedureHandler *procedure.Handler,
) {
	v1 := api.Group("/v1")
	{
		communes := v1.Group("/communes")
		{
			communes.GET("", communeHandler.List)
			communes.GET("/:id", communeHandler.Get)
		}

		domains := v1.Group("/domains")
		{
			domains.GET("", domainHandler.List)
			domains.GET("/:id", domainHandler.Get)
			domains.POST("", domainHandler.Create)
			domains.PUT("/:id", domainHandler.Update)
			domains.DELETE("/:id", domainHandler.Delete)
		}

		procedures := v1.Group("/procedures")
		{
			procedures.GET("", procedureHandler.List)
			procedures.GET("/by-code/:code", procedureHandler.GetByCode)
			procedures.GET("/:id/active-version", procedureHandler.GetActiveVersion)
		}
	}
}
