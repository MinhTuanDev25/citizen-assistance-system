from app.eval.extract_metrics import score_case, score_intent, score_slots, summarize_cases, values_equal_strict

TYPES = {
    "noi_sinh": "string",
    "co_giay_chung_sinh": "boolean",
    "da_ket_hon": "boolean",
    "nguoi_di_dang_ky": "enum",
    "so_ban_sao": "number",
}


def test_gold_null_predicted_procedure_fails():
    assert score_intent(None, "dk_khai_sinh") is False
    case = score_case(None, "dk_khai_sinh", {}, {}, TYPES)
    assert case["intent_ok"] is False
    assert case["case_pass_strict"] is False


def test_gold_null_predicted_null_passes():
    assert score_intent(None, None) is True
    case = score_case(None, None, {}, {}, TYPES)
    assert case["intent_ok"] is True
    assert case["strict"]["exact"] is True
    assert case["strict"]["precision"] is None
    assert case["strict"]["recall"] is None


def test_wrong_slot_value_is_fp_and_fn():
    got = score_slots({"da_ket_hon": True}, {"da_ket_hon": False}, TYPES)
    assert got["tp"] == 0
    assert got["fp"] == 1
    assert got["fn"] == 1
    assert got["details"][0]["status"] == "FP+FN"


def test_extra_slot_is_fp():
    got = score_slots({}, {"da_ket_hon": True}, TYPES)
    assert got["tp"] == 0 and got["fp"] == 1 and got["fn"] == 0


def test_missing_slot_is_fn():
    got = score_slots({"noi_sinh": "Chư Sê"}, {}, TYPES)
    assert got["tp"] == 0 and got["fp"] == 0 and got["fn"] == 1


def test_boolean_false_is_not_missing():
    got = score_slots({"co_giay_chung_sinh": False}, {"co_giay_chung_sinh": False}, TYPES)
    assert got["tp"] == 1 and got["fp"] == 0 and got["fn"] == 0
    assert values_equal_strict(False, False, "boolean") is True
    assert values_equal_strict(False, None, "boolean") is False


def test_none_false_empty_absent_are_distinct():
    assert score_slots({"a": False}, {}, {"a": "boolean"})["fn"] == 1
    assert score_slots({}, {"a": False}, {"a": "boolean"})["fp"] == 1
    assert score_slots({"a": ""}, {"a": ""}, {"a": "string"})["tp"] == 1
    missing = score_slots({"a": "x"}, {}, {"a": "string"})
    assert missing["fn"] == 1 and "a" not in {}


def test_relaxed_does_not_overwrite_strict():
    expected = {"noi_sinh": "Chư Sê"}
    predicted = {"noi_sinh": "Bệnh viện huyện Chư Sê"}
    case = score_case("dk_khai_sinh", "dk_khai_sinh", expected, predicted, TYPES)
    assert case["strict"]["exact"] is False
    assert case["strict"]["tp"] == 0
    assert case["strict"]["fp"] == 1 and case["strict"]["fn"] == 1
    assert case["relaxed"]["exact"] is True
    assert case["relaxed"]["tp"] == 1
    assert case["strict"]["mode"] == "strict"
    assert case["relaxed"]["mode"] == "relaxed"


def test_duplicate_predicted_entries_do_not_inflate_tp():
    predicted = [
        {"key": "da_ket_hon", "value": True},
        {"key": "da_ket_hon", "value": True},
    ]
    got = score_slots({"da_ket_hon": True}, predicted, TYPES)
    assert got["tp"] == 1
    assert got["fp"] == 1
    assert got["duplicate_predicted"] == 1


def test_metrics_deterministic_across_runs():
    expected = {"co_giay_chung_sinh": True, "nguoi_di_dang_ky": "me"}
    predicted = {"co_giay_chung_sinh": True, "nguoi_di_dang_ky": "me"}
    a = score_case("dk_khai_sinh", "dk_khai_sinh", expected, predicted, TYPES)
    b = score_case("dk_khai_sinh", "dk_khai_sinh", expected, predicted, TYPES)
    assert a == b
    summary = summarize_cases([a, a])
    assert summary["intent_correct"] == 2
    assert summary["strict_slot_micro"]["tp"] == 4


def test_number_and_enum_stay_exact_even_in_relaxed_mode():
    assert score_slots({"so_ban_sao": 2}, {"so_ban_sao": 2.0}, TYPES, relaxed=True)["tp"] == 1
    assert score_slots({"nguoi_di_dang_ky": "me"}, {"nguoi_di_dang_ky": "cha"}, TYPES, relaxed=True)["fp"] == 1
