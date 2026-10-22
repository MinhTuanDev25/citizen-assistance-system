package activation

import "testing"

func testEnv(pairs map[string]string) func(string) string {
	return func(key string) string { return pairs[key] }
}

func TestValidateP4TestDSNAcceptsDedicatedLocal(t *testing.T) {
	env := testEnv(map[string]string{"CAS_P4_TEST": "1"})
	err := ValidateP4TestDSN("postgres://cas:cas@127.0.0.1:5432/cas_p4_test?sslmode=disable", env)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateP4TestDSN("postgres://cas:cas@localhost:5432/cas_p4_test?sslmode=disable", env); err != nil {
		t.Fatal(err)
	}
	custom := testEnv(map[string]string{"CAS_P4_TEST": "1", "CAS_P4_TEST_PORT": "5434"})
	if err := ValidateP4TestDSN("postgres://cas:cas@127.0.0.1:5434/cas_p4_test?sslmode=disable", custom); err != nil {
		t.Fatal(err)
	}
}

func TestValidateP4TestDSNRejects(t *testing.T) {
	okEnv := testEnv(map[string]string{"CAS_P4_TEST": "1"})
	cases := []struct {
		name string
		url  string
		env  func(string) string
	}{
		{name: "missing flag", url: "postgres://cas:cas@127.0.0.1:5432/cas_p4_test?sslmode=disable", env: testEnv(nil)},
		{name: "missing url", url: "", env: okEnv},
		{name: "dev database", url: "postgres://cas:cas@127.0.0.1:5432/citizen_assistance?sslmode=disable", env: okEnv},
		{name: "substring name", url: "postgres://cas:cas@127.0.0.1:5432/cas_p4_test_extra?sslmode=disable", env: okEnv},
		{name: "remote host", url: "postgres://cas:cas@db.internal:5432/cas_p4_test?sslmode=disable", env: okEnv},
		{name: "production host", url: "postgres://cas:cas@prod.example.com:5432/cas_p4_test?sslmode=disable", env: okEnv},
		{name: "wrong port", url: "postgres://cas:cas@127.0.0.1:5434/cas_p4_test?sslmode=disable", env: okEnv},
		{name: "query host override", url: "postgres://cas:cas@127.0.0.1:5432/cas_p4_test?sslmode=disable&host=db.internal", env: okEnv},
		{name: "query dbname override", url: "postgres://cas:cas@127.0.0.1:5432/cas_p4_test?sslmode=disable&dbname=citizen_assistance", env: okEnv},
		{name: "malformed", url: "not-a-dsn", env: okEnv},
		{name: "pghost override", url: "postgres://cas:cas@127.0.0.1:5432/cas_p4_test?sslmode=disable", env: testEnv(map[string]string{"CAS_P4_TEST": "1", "PGHOST": "db.internal"})},
		{name: "pgdatabase override", url: "postgres://cas:cas@127.0.0.1:5432/cas_p4_test?sslmode=disable", env: testEnv(map[string]string{"CAS_P4_TEST": "1", "PGDATABASE": "citizen_assistance"})},
		{name: "missing credentials", url: "postgres://cas@127.0.0.1:5432/cas_p4_test?sslmode=disable", env: okEnv},
		{name: "multi host", url: "postgres://cas:cas@127.0.0.1,localhost:5432/cas_p4_test?sslmode=disable", env: okEnv},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := ValidateP4TestDSN(tc.url, tc.env); err == nil {
				t.Fatal("expected rejection")
			}
		})
	}
}
