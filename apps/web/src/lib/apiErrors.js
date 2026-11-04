/**
 * Vietnamese text for API errors. The server answers in English (API contract);
 * the UI never shows that text. Order: exact message → message pattern → code →
 * HTTP status. err.serverMessage keeps the original for logic and debugging.
 */

const BY_MESSAGE = {
  // auth
  'email or password is incorrect': 'Email hoặc mật khẩu không đúng.',
  'email and password are required': 'Nhập email và mật khẩu.',
  'full_name, email, password are required': 'Nhập đủ họ tên, email và mật khẩu.',
  'full_name is too short': 'Họ tên quá ngắn.',
  'email is invalid': 'Email không hợp lệ.',
  'password must be at least 6 characters': 'Mật khẩu phải có ít nhất 6 ký tự.',
  'email already registered': 'Email này đã được đăng ký. Hãy đăng nhập.',
  'Bearer token required': 'Phiên đăng nhập đã hết. Hãy đăng nhập lại.',
  'admin role required': 'Cần tài khoản cán bộ.',
  'admin role required for full catalog': 'Cần tài khoản cán bộ.',
  'admin Bearer token required for full catalog': 'Cần đăng nhập bằng tài khoản cán bộ.',
  'user not found': 'Không tìm thấy tài khoản.',
  // sessions / chat
  'session not found': 'Không tìm thấy phiên trò chuyện.',
  'session is not OPEN': 'Phiên trò chuyện đã đóng. Hãy mở hội thoại mới.',
  'session belongs to another user': 'Phiên trò chuyện thuộc tài khoản khác.',
  'guest token mismatch': 'Phiên khách không khớp. Hãy mở hội thoại mới.',
  'guest session is not OPEN; omit guest_token to create a new one': 'Phiên khách đã đóng. Hãy mở hội thoại mới.',
  'guest_token is bound to another xa_id': 'Phiên khách thuộc xã khác. Hãy mở hội thoại mới.',
  'X-Guest-Token header is required': 'Thiếu phiên khách. Hãy tải lại trang.',
  'message is required': 'Nhập nội dung câu hỏi.',
  'message is too long': 'Câu hỏi quá dài.',
  'speech translation is disabled': 'Chức năng giọng nói đang tắt.',
  'commune not found': 'Không tìm thấy xã.',
  'commune not found or inactive': 'Xã không tồn tại hoặc đã ngưng hoạt động.',
  // catalog / procedures / versions
  'procedure not found': 'Không tìm thấy thủ tục.',
  'procedure version not found': 'Không tìm thấy phiên bản thủ tục.',
  'active procedure version not found': 'Thủ tục chưa có phiên bản đang dùng.',
  'procedure outside citizen catalog scope': 'Thủ tục không có trong danh mục cho người dân.',
  'procedure_code already exists in this commune': 'Mã thủ tục này đã có trong xã. Hãy dùng mã khác.',
  'version already exists': 'Số phiên bản này đã tồn tại. Hãy nhập số khác.',
  'active version changed': 'Phiên bản đang dùng vừa thay đổi. Tải lại trang rồi thử lại.',
  'expected_active_version_id does not match the current ACTIVE version': 'Phiên bản đang dùng vừa thay đổi. Tải lại trang rồi thử lại.',
  'activation conflict': 'Không kích hoạt được vì có thao tác kích hoạt khác. Tải lại trang rồi thử lại.',
  'domain not found': 'Không tìm thấy lĩnh vực.',
  'domain id already exists': 'Mã lĩnh vực đã tồn tại.',
  'domain is referenced by procedures/documents': 'Lĩnh vực đang được thủ tục hoặc tài liệu sử dụng nên không xóa được.',
  'domain is not available': 'Lĩnh vực không khả dụng.',
  // documents / indexing
  'document not found': 'Không tìm thấy tài liệu.',
  'document content unavailable': 'Không đọc được nội dung tài liệu.',
  'document link not found': 'Không tìm thấy liên kết tài liệu.',
  'document is already attached': 'Tài liệu đã được gắn vào phiên bản này.',
  'document indexing is already in progress': 'Tài liệu đang được lập chỉ mục. Chờ xử lý xong rồi thử lại.',
  'a document with this content already exists': 'Tài liệu có cùng nội dung đã tồn tại.',
  'document and procedure version do not match': 'Tài liệu và phiên bản thủ tục không khớp.',
  'document is not attached to a procedure version': 'Tài liệu chưa được gắn vào phiên bản thủ tục.',
  'document status does not allow this action': 'Trạng thái tài liệu không cho phép thao tác này.',
  'exactly one file is required': 'Chọn đúng một file.',
  'filename is not allowed': 'Tên file không hợp lệ.',
  'request body exceeds size limit': 'Dữ liệu gửi lên vượt quá dung lượng cho phép.',
  'upload failed': 'Không tải lên được. Thử lại sau.',
  'indexing failed': 'Không lập được chỉ mục tài liệu.',
  // drafts
  'draft or source not found': 'Không tìm thấy bản nháp hoặc tài liệu nguồn.',
  'draft revision is stale': 'Bản nháp vừa được thay đổi ở nơi khác. Tải lại trang rồi thử lại.',
  'draft must be Admin-approved before activation': 'Bản nháp phải được cán bộ phê duyệt trước khi kích hoạt.',
  'definition drafting is not configured': 'Chức năng soạn tự động chưa được bật.',
  'draft generation failed': 'Dịch vụ soạn tự động gặp lỗi. Thử lại sau.',
  'ai draft timed out': 'Dịch vụ soạn tự động xử lý quá thời gian. Thử lại sau.',
  'ai service rate limited': 'Dịch vụ soạn tự động đang quá tải. Thử lại sau ít phút.',
  'selected source is not a READY verified generation': 'Nguồn đã chọn chưa lập chỉ mục xong.',
  'only a FAILED job can be retried': 'Chỉ thử lại được tiến trình bị lỗi.',
  'request id was reused with a different body': 'Yêu cầu này đã được gửi với nội dung khác. Tải lại trang rồi thử lại.',
  'X-Request-ID was reused with a different message payload': 'Yêu cầu này đã được gửi với nội dung khác. Tải lại trang rồi thử lại.',
}

// Server details after a "<sentinel>: " prefix (fmt.Errorf("%w: ...")).
const BY_PATTERN = [
  [/reason must be 10-500 characters/, 'Lý do phải dài từ 10 đến 500 ký tự.'],
  [/reason must be 5-500 characters/, 'Lý do phải dài từ 5 đến 500 ký tự.'],
  [/only DRAFT rows can be (changed|patched)/, 'Bản nháp đã được phê duyệt nên không sửa được. Dùng "Tạo bản nháp để sửa".'],
  [/this proposal has a valid source quotation; accept it instead/, 'Đề xuất này đã có trích dẫn hợp lệ — hãy bấm Chấp nhận.'],
  [/unknown proposal/, 'Đề xuất này không còn trong bản nháp. Tải lại trang.'],
  [/proposal \S+ has no source evidence to review/, 'Đề xuất này không có nguồn để duyệt.'],
  [/cannot accept (invalid|stale) evidence/, 'Không chấp nhận được vì trích dẫn không hợp lệ hoặc đã cũ.'],
  [/invalid review decision/, 'Quyết định duyệt không hợp lệ.'],
  [/material proposal \S+ is not Admin-accepted/, 'Còn đề xuất chưa được cán bộ chấp nhận.'],
  [/source conflict on \S+ is not resolved/, 'Còn mâu thuẫn nguồn chưa chọn giá trị.'],
  [/source coverage is incomplete/, 'Nguồn chưa bao phủ đủ nội dung.'],
  [/stale evidence remains/, 'Còn trích dẫn cũ cần duyệt lại.'],
  [/unresolved blocking validation findings/, 'Còn lỗi kiểm tra tự động chưa xử lý.'],
  [/draft has no coverage report/, 'Bản nháp chưa có báo cáo phạm vi nguồn.'],
  [/is not a pending coverage finding/, 'Mục này không còn cần xử lý. Tải lại trang.'],
  [/procedure identity mismatch/, 'Nội dung không khớp mã thủ tục.'],
  [/at most \d+ citations; replace one instead/, 'Đã đủ số trích dẫn tối đa — hãy thay một trích dẫn.'],
  [/invalid citation to replace|unknown dropped citation/, 'Trích dẫn không hợp lệ. Tải lại trang.'],
  [/this source is already citation \d+ of the Definition/, 'Văn bản này đã có trong danh sách trích dẫn.'],
  [/this document is no longer cited in the Definition/, 'Văn bản này không còn trong danh sách trích dẫn — hãy giữ giá trị hiện tại.'],
  [/unknown conflict|invalid choice|choice is required/, 'Lựa chọn không hợp lệ. Tải lại trang.'],
  [/only an approved or published draft can be copied/, 'Chỉ sao chép được bản nháp đã phê duyệt; bản này sửa trực tiếp được.'],
  [/pages \d+-\d+ have \d+ excerpts .*choose a shorter page range/, 'Khoảng trang quá dài — chọn khoảng ngắn hơn.'],
  [/source page range is too large/, 'Khoảng trang quá dài — chọn khoảng ngắn hơn.'],
  [/version must be n\.n\.n/, 'Số phiên bản phải có dạng n.n.n, ví dụ 2.0.0.'],
  [/definition or copy_from_version_id is required/, 'Cần bản định nghĩa hoặc phiên bản nguồn để sao chép.'],
  [/a SOURCE version has no definition to copy/, 'Phiên nguồn không có hướng dẫn để sao chép.'],
  [/procedure_code must be/, 'Mã thủ tục chỉ gồm chữ thường không dấu, số và dấu gạch dưới (2–64 ký tự).'],
  [/domain_id is (required|not an active domain)/, 'Lĩnh vực không hợp lệ hoặc đã ngưng.'],
  [/name is required/, 'Nhập tên thủ tục.'],
  [/definition schema is invalid|unexpected schema_version|definition is not procedurally usable/, 'Nội dung hướng dẫn không hợp lệ.'],
  [/confirm is required/, 'Thiếu xác nhận.'],
  [/expected_updated_at is required/, 'Thiếu thông tin phiên bản bản nháp. Tải lại trang.'],
  [/X-Request-ID/, 'Yêu cầu thiếu mã định danh. Tải lại trang rồi thử lại.'],
  [/^invalid (draft|job|procedure|version) id$|^invalid procedure_id$|^invalid sessionId$/, 'Đường dẫn không hợp lệ.'],
  [/^invalid (request body|multipart body)$|^expected a multipart upload$|^unknown form field$/, 'Dữ liệu gửi lên không hợp lệ.'],
]

const BY_CODE = {
  UNAUTHORIZED: 'Phiên đăng nhập đã hết. Hãy đăng nhập lại.',
  FORBIDDEN: 'Bạn không có quyền thực hiện thao tác này.',
  NOT_FOUND: 'Không tìm thấy dữ liệu.',
  VALIDATION: 'Dữ liệu không hợp lệ.',
  BAD_REQUEST: 'Dữ liệu không hợp lệ.',
  CONFLICT: 'Dữ liệu vừa thay đổi. Tải lại trang rồi thử lại.',
  REVISION_CONFLICT: 'Bản nháp vừa được thay đổi ở nơi khác. Tải lại trang rồi thử lại.',
  STALE_ACTIVE_VERSION: 'Phiên bản đang dùng vừa thay đổi. Tải lại trang rồi thử lại.',
  IDEMPOTENCY_CONFLICT: 'Yêu cầu này đã được gửi với nội dung khác. Tải lại trang rồi thử lại.',
  ACTIVATION_CONFLICT: 'Không kích hoạt được vì có thao tác kích hoạt khác. Tải lại trang rồi thử lại.',
  REVIEW_INVALID: 'Thao tác duyệt không hợp lệ.',
  APPROVAL_BLOCKED: 'Chưa đủ điều kiện phê duyệt.',
  NOT_APPROVED: 'Bản nháp phải được cán bộ phê duyệt trước khi kích hoạt.',
  VERSION_NOT_APPROVED: 'Phiên bản này chưa được duyệt.',
  INVALID_DEFINITION: 'Nội dung hướng dẫn không hợp lệ.',
  UNVERIFIED_SOURCE: 'Nguồn đã chọn chưa lập chỉ mục xong.',
  SOURCE_TOO_LARGE: 'Khoảng trang quá dài — chọn khoảng ngắn hơn.',
  PLAN_BLOCKED: 'Chưa thể soạn với phạm vi này.',
  JOB_NOT_RETRYABLE: 'Chỉ thử lại được tiến trình bị lỗi.',
  DOCUMENT_DUPLICATE: 'Tài liệu có cùng nội dung đã tồn tại.',
  FK_RESTRICTED: 'Dữ liệu đang được sử dụng nên không xóa được.',
  PROCEDURE_CODE_EXISTS: 'Mã thủ tục này đã có trong xã. Hãy dùng mã khác.',
  PAYLOAD_TOO_LARGE: 'Dữ liệu gửi lên vượt quá dung lượng cho phép.',
  RATE_LIMITED: 'Hệ thống đang quá tải. Thử lại sau ít phút.',
  DRAFT_TIMEOUT: 'Dịch vụ soạn tự động xử lý quá thời gian. Thử lại sau.',
  AI_PROVIDER: 'Dịch vụ soạn tự động gặp lỗi. Thử lại sau.',
  DRAFTING_DISABLED: 'Chức năng soạn tự động chưa được bật.',
  SPEECH_DISABLED: 'Chức năng giọng nói đang tắt.',
  DEPRECATED: 'Chức năng này không còn được hỗ trợ. Tải lại trang.',
  INTERNAL: 'Hệ thống gặp lỗi. Thử lại sau.',
}

function byStatus(status) {
  if (status === 401) return 'Phiên đăng nhập đã hết. Hãy đăng nhập lại.'
  if (status === 403) return 'Bạn không có quyền thực hiện thao tác này.'
  if (status === 404) return 'Không tìm thấy dữ liệu.'
  if (status === 409) return 'Dữ liệu vừa thay đổi. Tải lại trang rồi thử lại.'
  if (status === 413) return 'Dữ liệu gửi lên vượt quá dung lượng cho phép.'
  if (status === 429) return 'Hệ thống đang quá tải. Thử lại sau ít phút.'
  if (status === 400 || status === 422) return 'Dữ liệu không hợp lệ.'
  if (status >= 500) return 'Hệ thống gặp lỗi. Thử lại sau.'
  return 'Không kết nối được máy chủ. Thử lại sau.'
}

/** Vietnamese message for a failed API call. */
export function apiErrorMessage({ status, code, message } = {}) {
  const msg = String(message || '').trim()
  if (BY_MESSAGE[msg]) return BY_MESSAGE[msg]
  for (const [re, text] of BY_PATTERN) {
    if (re.test(msg)) return text
  }
  if (code && BY_CODE[code]) return BY_CODE[code]
  return byStatus(status)
}

/** Activation / eligibility reason codes (procedure versions and draft activation). */
const REASONS = {
  VERSION_NOT_APPROVED: 'Phiên bản này chưa được duyệt.',
  SOURCE_VERSION: 'Phiên nguồn chỉ để gắn tài liệu và tạo bản nháp tự động, không kích hoạt.',
  VERSION_ARCHIVED: 'Phiên bản đã lưu trữ.',
  VERSION_ALREADY_ACTIVE: 'Phiên bản này đang được dùng.',
  INVALID_DEFINITION: 'Nội dung hướng dẫn của phiên bản không hợp lệ.',
  NO_SOURCE_DOCUMENT: 'Chưa gắn tài liệu nguồn.',
  PARTIAL_SOURCE_INDEX: 'Một số tài liệu gắn vào phiên bản chưa sẵn sàng.',
  DOCUMENT_INDEX_UPLOADED: 'Tài liệu mới gắn, chưa lập chỉ mục.',
  DOCUMENT_INDEX_PROCESSING: 'Tài liệu đang được lập chỉ mục.',
  DOCUMENT_INDEX_FAILED: 'Chỉ mục tài liệu thất bại.',
  DOCUMENT_INDEX_NOT_READY: 'Tài liệu chưa sẵn sàng.',
  ACTIVE_GENERATION_MISSING: 'Tài liệu chưa có bản chỉ mục hoàn chỉnh.',
  MOCK_INDEX_NOT_PUBLISHED: 'Chưa có nội dung chỉ mục để trả lời. Không kích hoạt được.',
  DOCUMENT_NOT_CURRENT: 'Tài liệu hết hiệu lực hoặc đã bị thay thế.',
  SOURCE_SCOPE_MISMATCH: 'Tài liệu không thuộc xã hoặc thủ tục này.',
  DEFINITION_IDENTITY_MISMATCH: 'Nội dung hướng dẫn không khớp thủ tục hoặc xã đang chọn. Tải lại trang rồi thử lại.',
  STALE_ACTIVE_VERSION: 'Phiên bản đang dùng đã đổi. Tải lại trang rồi thử lại.',
  IDEMPOTENCY_CONFLICT: 'Yêu cầu này đã được gửi với nội dung khác. Tải lại trang rồi thử lại.',
  ACTIVATION_CONFLICT: 'Không kích hoạt được vì còn phiên bản đang bật khác. Tải lại trang rồi thử lại.',
  // draft activation (blocked_reason)
  INDEXING_PENDING: 'Tài liệu đang được lập chỉ mục.',
  INDEXING_FAILED: 'Không lập được chỉ mục tài liệu nguồn. Vào Tài liệu để thử lại.',
  INDEX_REQUEST_FAILED: 'Không gửi được yêu cầu lập chỉ mục.',
  INDEX_SERVICE_UNAVAILABLE: 'Dịch vụ lập chỉ mục chưa sẵn sàng.',
  INDEX_METADATA_PERSIST_FAILED: 'Đã gửi lập chỉ mục nhưng chưa lưu được trạng thái. Bấm Kích hoạt / Tiếp tục để hoàn tất.',
  DOCUMENT_LINK_FAILED: 'Không gắn được tài liệu nguồn vào phiên bản mới.',
  ELIGIBILITY_CHECK_FAILED: 'Không kiểm tra được điều kiện kích hoạt. Thử lại.',
  ACTIVATE_FAILED: 'Kích hoạt thất bại. Thử lại.',
  DRAFT_PUBLISH_PENDING: 'Đã kích hoạt nhưng chưa đồng bộ trạng thái bản nháp.',
}

export function reasonLabel(code) {
  return REASONS[code] || ''
}

/** "Lý do 1; Lý do 2" for reason codes, without raw codes. */
export function reasonsText(codes) {
  const seen = new Set()
  for (const code of codes || []) {
    const text = reasonLabel(code)
    if (text) seen.add(text)
  }
  return [...seen].join(' ')
}
