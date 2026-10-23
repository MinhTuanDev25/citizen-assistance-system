package v1

import (
	"net/http"
	"testing"

	authapi "github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/commune"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/domain"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/procedure"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/session"
	speechapi "github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/api/http/v1/speech"
	"github.com/gin-gonic/gin"
)

func TestMapRoutesRegistersSpeechWithoutReplacingChat(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	api := r.Group("/api")
	passthrough := func(c *gin.Context) { c.Next() }
	MapRoutes(
		api,
		&commune.Handler{},
		&domain.Handler{},
		&procedure.Handler{},
		&session.Handler{},
		&authapi.Handler{},
		nil,
		speechapi.NewHandler(speechapi.Handler{Enabled: false}),
		passthrough,
		passthrough,
		passthrough,
	)
	var speech, legacy, turns, messages int
	for _, rt := range r.Routes() {
		switch {
		case rt.Method == http.MethodPost && rt.Path == "/api/v1/sessions/:sessionId/speech/translate":
			speech++
		case rt.Method == http.MethodPost && rt.Path == "/api/v1/speech/translate":
			legacy++
		case rt.Method == http.MethodPost && rt.Path == "/api/v1/sessions/:sessionId/turns":
			turns++
		case rt.Method == http.MethodPost && rt.Path == "/api/v1/sessions/:sessionId/messages":
			messages++
		}
	}
	if speech != 1 || turns != 1 || messages != 1 {
		t.Fatalf("speech=%d turns=%d messages=%d", speech, turns, messages)
	}
	if legacy != 1 {
		t.Fatalf("legacy gone route %d", legacy)
	}
}
