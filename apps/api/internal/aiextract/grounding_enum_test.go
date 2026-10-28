package aiextract

import "testing"

// truong_hop_tre as drafted for khai_sinh_v1 (no hand-written cue table).
func truongHopTre(labels bool) SlotSpec {
	spec := SlotSpec{Type: "enum", Question: "Trường hợp khai sinh của trẻ thuộc diện nào sau đây?",
		EnumValues: []string{"tre_sinh_thuong", "tre_bi_bo_roi", "mang_thai_ho", "chua_xac_dinh_cha_me"}}
	if labels {
		spec.EnumLabels = []string{"Trẻ sinh bình thường", "Trẻ bị bỏ rơi", "Mang thai hộ", "Chưa xác định được cha, mẹ"}
	}
	return spec
}

func TestGroundEnumAcceptsNaturalAnswers(t *testing.T) {
	cases := []struct {
		labels               bool
		msg, value, evidence string
	}{
		// The live failure (09:08:28): "sinh thường" names tre_sinh_thuong only.
		{false, "sinh thường", "tre_sinh_thuong", "sinh thường"},
		{false, "trẻ sinh thường", "tre_sinh_thuong", "trẻ sinh thường"},
		{false, "à quên trẻ bị bỏ rơi", "tre_bi_bo_roi", "trẻ bị bỏ rơi"},
		{false, "bé bị bỏ rơi", "tre_bi_bo_roi", "bị bỏ rơi"},
		{false, "nhờ mang thai hộ", "mang_thai_ho", "mang thai hộ"},
		{true, "con tôi sinh bình thường", "tre_sinh_thuong", "sinh bình thường"},
	}
	for _, c := range cases {
		if ok, reason := GroundSlot(c.msg, truongHopTre(c.labels), "truong_hop_tre", c.value, c.evidence); !ok {
			t.Errorf("%q → %s rejected: %s", c.msg, c.value, reason)
		}
	}
}

func TestGroundEnumStillRejectsUnsupported(t *testing.T) {
	cases := []struct {
		msg, value, evidence string
	}{
		// "trẻ" alone names every choice.
		{"trẻ", "tre_sinh_thuong", "trẻ"},
		// The span talks about a birth certificate, not the case of the child.
		{"tôi chưa có giấy chứng sinh", "tre_sinh_thuong", "chưa có giấy chứng sinh"},
		// The span names another choice too.
		{"sinh thường hay bị bỏ rơi", "tre_sinh_thuong", "sinh thường hay bị bỏ rơi"},
		// Negated.
		{"không phải sinh thường", "tre_sinh_thuong", "sinh thường"},
		// Invented evidence.
		{"con tôi", "tre_sinh_thuong", "sinh thường"},
	}
	for _, c := range cases {
		if ok, _ := GroundSlot(c.msg, truongHopTre(false), "truong_hop_tre", c.value, c.evidence); ok {
			t.Errorf("%q → %s must be rejected", c.msg, c.value)
		}
	}
}

// Yes/no questions drafted by AI have no hand-written cue table.
func TestGroundQuestionBoolean(t *testing.T) {
	banSao := SlotSpec{Type: "boolean", Question: "Bạn có yêu cầu cấp bản sao Giấy khai sinh hay không?"}
	uyQuyen := SlotSpec{Type: "boolean", Question: "Người thực hiện thủ tục có phải là người được ủy quyền hay không?"}
	ok := []struct {
		spec          SlotSpec
		msg, evidence string
		value         bool
	}{
		{banSao, "tôi muốn xin thêm 2 bản sao giấy khai sinh", "bản sao giấy khai sinh", true},
		{banSao, "có, cấp bản sao cho tôi", "cấp bản sao", true},
		{banSao, "không cần bản sao", "không cần bản sao", false},
		{uyQuyen, "tôi được ủy quyền đi làm", "được ủy quyền", true},
		{uyQuyen, "không có ủy quyền gì", "không có ủy quyền", false},
	}
	for _, c := range ok {
		if got, reason := GroundSlotFor("Đăng ký khai sinh", c.msg, c.spec, "k", c.value, c.evidence); !got {
			t.Errorf("%q → %v rejected: %s", c.msg, c.value, reason)
		}
	}
	bad := []struct {
		spec          SlotSpec
		msg, evidence string
		value         bool
	}{
		{banSao, "không cần bản sao", "không cần bản sao", true},                       // polarity
		{banSao, "có", "có", true},                                                     // says nothing about the topic
		{banSao, "giấy khai sinh của con", "giấy khai sinh", true},                     // one topic word is not enough
		{banSao, "không biết có cần bản sao không", "không biết có cần bản sao", true}, // hedged
		{uyQuyen, "mẹ đi đăng ký", "mẹ đi đăng ký", true},
		{banSao, "con tôi", "bản sao", true}, // invented evidence
	}
	for _, c := range bad {
		if got, _ := GroundSlotFor("Đăng ký khai sinh", c.msg, c.spec, "k", c.value, c.evidence); got {
			t.Errorf("%q → %v must be rejected", c.msg, c.value)
		}
	}
}
