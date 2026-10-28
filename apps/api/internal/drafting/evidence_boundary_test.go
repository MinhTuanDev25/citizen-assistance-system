package drafting

import (
	"testing"

	"github.com/google/uuid"
)

// Real case (khai_sinh_v1, chunks 43/44): chunks overlap and the model quoted
// a sentence that runs from one into the next.
func TestQuoteAcrossOverlappingChunks(t *testing.T) {
	doc, gen := uuid.New(), uuid.New()
	mk := func(chunk, page int, text string) Excerpt {
		return Excerpt{ExcerptID: uuid.New(), DocumentID: doc, GenerationID: gen, ChunkIndex: chunk, PageStart: page, PageEnd: page, Content: text}
	}
	a := mk(43, 12, "Cách thức thực hiện:-Người yêu cầu nộp hồ sơ trực tiếp tại Trung tâm Phục vụ hành chính công,gửi qua hệ thống bưu chính hoặc nộp hồ sơ trực tuyến tại Cổng dịch vụ công quốc gia(hps:")
	b := mk(44, 12, "hoặc nộp hồ sơ trực tuyến tại Cổng dịch vụ công quốc gia(hps://dichvucong.gov.vn)hoặc Ứng dụng định danh quốc gia.Thành phần hồ sơ")
	far := mk(46, 13, "hoặc Ứng dụng định danh quốc gia.")
	idx := newExcerptIndex([]Excerpt{a, b, far})
	quote := "nộp hồ sơ trực tiếp tại Trung tâm Phục vụ hành chính công,gửi qua hệ thống bưu chính hoặc nộp hồ sơ trực tuyến tại Cổng dịch vụ công quốc gia(hps://dichvucong.gov.vn)hoặc Ứng dụng định danh quốc gia."
	if quotationInText(quote, a.Content) {
		t.Fatal("setup: the quote must not fit in one chunk")
	}
	if !idx.quoteFound(quote, 12, a) {
		t.Fatal("a quote running into the next chunk must verify")
	}
	if !idx.quoteFound(quote, 12, b) {
		t.Fatal("cited by the second chunk, the quote must verify from the previous one")
	}
	if idx.quoteFound(quote, 30, a) {
		t.Fatal("the page must still lie in the joined chunks")
	}
	if idx.quoteFound("nộp hồ sơ trực tiếp tại Trung tâm Phục vụ hành chính công,gửi qua hệ thống bưu chính hoặc nộp hồ sơ trực tuyến tại Cổng dịch vụ công quốc gia(hps:hoặc Ứng dụng định danh quốc gia.", 12, a) {
		t.Fatal("text that is not in the joined source must not verify")
	}
	other := mk(44, 12, b.Content)
	other.DocumentID = uuid.New()
	if newExcerptIndex([]Excerpt{a, other}).quoteFound(quote, 12, a) {
		t.Fatal("a chunk of another document is not a neighbour")
	}
	gap := newExcerptIndex([]Excerpt{a, mk(45, 12, b.Content)})
	if gap.quoteFound(quote, 12, a) {
		t.Fatal("only the directly following chunk counts")
	}
}

func TestVerifyQuotesAcceptsBoundaryQuote(t *testing.T) {
	doc, gen := uuid.New(), uuid.New()
	a := Excerpt{ExcerptID: uuid.New(), DocumentID: doc, GenerationID: gen, ChunkIndex: 46, PageStart: 13, PageEnd: 13,
		Content: "Giấy tờ phải xuất trình:-Hộ chiếu/Thẻ căn cước công dân/Thẻ căn cước/Căn cước điện tử/Giấy chứng nhận"}
	b := Excerpt{ExcerptID: uuid.New(), DocumentID: doc, GenerationID: gen, ChunkIndex: 47, PageStart: 13, PageEnd: 13,
		Content: "cước công dân/Thẻ căn cước/Căn cước điện tử/Giấy chứng nhận căn cước hoặc các giấy tờ khác có dán ảnh,còn giá trị sử dụng."}
	ev := []EvidenceItem{{ProposalID: "checklist:5", ExcerptID: b.ExcerptID.String(), Page: 13,
		Quotation: "Hộ chiếu/Thẻ căn cước công dân/Thẻ căn cước/Căn cước điện tử/Giấy chứng nhận căn cước hoặc các giấy tờ khác có dán ảnh,còn giá trị sử dụng."}}
	out := verifyQuotes(ev, []Excerpt{a, b}, SourceSnapshot{DocumentID: doc, GenerationID: gen})
	if !out.items[0].QuoteValid || out.items[0].Status != "source_matched_semantic_unverified" {
		t.Fatalf("boundary quote = %+v", out.items[0])
	}
}
