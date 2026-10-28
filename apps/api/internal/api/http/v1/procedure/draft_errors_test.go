package procedure

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/activation"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/drafting"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
)

func TestDraftActivateStaleActiveVersionIsConflict(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/", nil)
	writeDraftErr(c, fmt.Errorf("wrapped: %w", activation.ErrStale))
	if w.Code != http.StatusConflict {
		t.Fatalf("status = %d, want 409", w.Code)
	}
	if body := w.Body.String(); !strings.Contains(body, "STALE_ACTIVE_VERSION") {
		t.Fatalf("body = %s", body)
	}
}

func TestDraftJobErrorMapping(t *testing.T) {
	gin.SetMode(gin.TestMode)
	cases := []struct {
		err  error
		code int
		body string
	}{
		{fmt.Errorf("%w: %s", drafting.ErrPlanBlocked, drafting.PlanMixedProcedures), http.StatusUnprocessableEntity, `"message":"MIXED_PROCEDURES"`},
		{drafting.ErrJobNotRetryable, http.StatusConflict, "JOB_NOT_RETRYABLE"},
		{repository.ErrIdempotency, http.StatusConflict, "IDEMPOTENCY_CONFLICT"},
		{repository.ErrNotFound, http.StatusNotFound, "NOT_FOUND"},
		{fmt.Errorf("drafting: invalid mode"), http.StatusBadRequest, "VALIDATION"},
	}
	for _, tc := range cases {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodPost, "/", nil)
		writeDraftJobErr(c, tc.err)
		if w.Code != tc.code || !strings.Contains(w.Body.String(), tc.body) {
			t.Errorf("%v -> %d %s", tc.err, w.Code, w.Body.String())
		}
	}
}
