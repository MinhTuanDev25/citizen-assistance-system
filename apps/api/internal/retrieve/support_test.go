package retrieve

import "testing"

const (
	g3Doc    = "60b677df-acc0-4198-a3b6-a919c01482a1"
	g3Gen    = "e2ad49da-4db7-4e18-90e3-c2f0bb358940"
	g3Clone  = "21125806-0a32-4e84-a866-d497859be2fe"
	g2Gen    = "06679d02-3a43-4831-8dde-82799fc9b4a4"
	gcsSHA   = "410276e9993151a4f2ff17e5e8a3ea7b5cdc6abd42faf6ded0ac07cf57e8b17c"

	stmtBirth    = "Giấy chứng sinh (nếu có) hoặc giấy tờ thay thế theo hướng dẫn xã"
	stmtIDParent = "CCCD/hộ chiếu của cha và/hoặc mẹ"
	stmtMarriage = "Giấy đăng ký kết hôn (nếu cha mẹ đã kết hôn)"
	stmtForm     = "Tờ khai đăng ký khai sinh (mẫu tại bộ phận tiếp nhận)"
	stmtGCS103   = "Giấy chứng sinh; nếu không có Giấy chứng sinh thì văn bản người làm chứng xác nhận việc sinh; nếu không có người làm chứng thì giấy cam đoan về việc sinh"
)

func g3Hit(page int, sha string) Hit {
	return Hit{
		DocumentID:    g3Doc,
		GenerationID:  g3Gen,
		PageStart:     page,
		PageEnd:       page,
		Excerpt:       "identity match is not entailment",
		ContentSHA256: sha,
	}
}

func TestClassifySupportRequiresAllowlistNotOverlap(t *testing.T) {
	birth := g3Hit(12, "410276e9993151a4f2ff17e5e8a3ea7b5cdc6abd42faf6ded0ac07cf57e8b17c")
	if ClassifySupport(birth, []string{stmtBirth}, "dk_khai_sinh") != "related" {
		t.Fatal("demoted birth-certificate wording must not be supporting")
	}
	form := g3Hit(12, "f355aaa615088e28a895b5ac3f26d10447db6d6ffdf95cb1db8e1a6f1e0e126a")
	if ClassifySupport(form, []string{stmtForm}, "dk_khai_sinh") != "related" {
		t.Fatal("demoted declaration-form wording must not be supporting")
	}
	wrongGen := birth
	wrongGen.GenerationID = g2Gen
	if ClassifySupport(wrongGen, []string{stmtBirth}, "dk_khai_sinh") != "related" {
		t.Fatal("G2 generation must stay related")
	}
	wrongSHA := birth
	wrongSHA.ContentSHA256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	if ClassifySupport(wrongSHA, []string{stmtBirth}, "dk_khai_sinh") != "related" {
		t.Fatal("wrong chunk hash must stay related")
	}
	if ClassifySupport(birth, []string{stmtMarriage}, "dk_khai_sinh") != "related" {
		t.Fatal("unmapped checklist item is not supporting")
	}
	if ClassifySupport(Hit{Excerpt: stmtBirth}, []string{stmtBirth}, "dk_khai_sinh") != "related" {
		t.Fatal("excerpt overlap without identity is not supporting")
	}
}

func TestClassifySupportIdentityWithoutAllowlistStaysRelated(t *testing.T) {
	cccd := g3Hit(13, "dbb5a7a599f32d2d5ec7be05f210eccf8dcb1bb1216d8c297816c026049887ef")
	if ClassifySupport(cccd, []string{stmtIDParent}, "dk_khai_sinh") != "related" {
		t.Fatal("parent ID claim is not fully supported by applicant-ID page")
	}
	if ClassifySupport(cccd, []string{stmtMarriage}, "dk_khai_sinh") != "related" {
		t.Fatal("marriage stays related even on a G3 identity of a different passage")
	}
}

func TestClassifySupportNonParentRegistrantDoesNotSupportParentID(t *testing.T) {
	// Page 13 is identity of the person performing registration (may be authorized).
	hit := g3Hit(13, "dbb5a7a599f32d2d5ec7be05f210eccf8dcb1bb1216d8c297816c026049887ef")
	hit.Excerpt = "Văn bản ủy quyền (được chứng thực) theo quy định của pháp luật trong trường hợp ủy quyền thực hiện việc đăng ký khai sinh. Giấy tờ tùy thân để chứng minh về nhân thân."
	if ClassifySupport(hit, []string{stmtIDParent}, "dk_khai_sinh") != "related" {
		t.Fatal("authorized non-parent registrant must not support parent CCCD checklist item")
	}
}

func TestVerifiedGCSAllowlistSupportsG3Identity(t *testing.T) {
	if len(compiledSupport) != 2 {
		t.Fatalf("allowlist must have G3 + clone GCS rows, got %d", len(compiledSupport))
	}
	gcs := g3Hit(12, gcsSHA)
	if ClassifySupport(gcs, []string{stmtGCS103}, "dk_khai_sinh") != "supporting" {
		t.Fatal("1.0.3 GCS statement must be supporting on G3 identity")
	}
	clone := gcs
	clone.GenerationID = g3Clone
	if ClassifySupport(clone, []string{stmtGCS103}, "dk_khai_sinh") != "supporting" {
		t.Fatal("1.0.3 clone generation must also be supporting")
	}
	oldChecklist := []string{stmtBirth, stmtIDParent, stmtMarriage, stmtForm}
	if ClassifySupport(gcs, oldChecklist, "dk_khai_sinh") != "related" {
		t.Fatal("seed 1.0.2 wording must stay related")
	}
}
