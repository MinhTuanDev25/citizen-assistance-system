/** Local assistant replies until Decision Engine saves ASSISTANT messages. */

const WELCOME =
  'Xin chào. Tôi là trợ lý thủ tục hành chính.\n\nBạn có thể hỏi về đăng ký khai sinh, chứng thực giấy tờ, hoặc các thủ tục hành chính khác. Câu hỏi của bạn đã được lưu trên server; trả lời tự động đầy đủ sẽ có khi nối Decision Engine.'

function detectTopic(text) {
  const t = text.toLowerCase()
  if (t.includes('khai sinh') || t.includes('đẻ') || t.includes('sinh con')) {
    return 'khai_sinh'
  }
  if (
    t.includes('chứng thực') ||
    t.includes('chung thuc') ||
    t.includes('sao y')
  ) {
    return 'chung_thuc'
  }
  return 'general'
}

export async function sendTurn({ message }) {
  await new Promise((r) => setTimeout(r, 400 + Math.random() * 300))

  const topic = detectTopic(message)

  if (topic === 'khai_sinh') {
    return {
      role: 'assistant',
      text:
        'Đã ghi nhận câu hỏi đăng ký khai sinh trên server.\n\n' +
        'Tạm thời (chưa Decision Engine), vui lòng cho biết thêm:\n' +
        '1. Nơi sinh của trẻ?\n' +
        '2. Đã có giấy chứng sinh chưa?\n' +
        '3. Cha/mẹ đã đăng ký kết hôn chưa?',
      action: 'ASK_MISSING_SLOTS',
    }
  }

  if (topic === 'chung_thuc') {
    return {
      role: 'assistant',
      text:
        'Đã ghi nhận câu hỏi chứng thực trên server.\n\n' +
        'Tạm thời: mang bản chính đến Bộ phận Một cửa; cho biết loại giấy tờ và số bản sao nếu cần.',
      action: 'DIRECT_ANSWER',
    }
  }

  return {
    role: 'assistant',
    text:
      'Đã lưu tin nhắn của bạn. Trả lời theo procedure JSON sẽ có khi nối Decision Engine.\n\n' +
      'Thử hỏi: “Đăng ký khai sinh” hoặc “Chứng thực bản sao”.',
    action: 'DIRECT_ANSWER',
  }
}

export { WELCOME }
