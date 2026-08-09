/** Mock assistant until POST /api/v1/chat/turns exists. */

const WELCOME =
  'Xin chào. Tôi là trợ lý thủ tục hành chính xã Chư Sê (Gia Lai).\n\nBạn có thể hỏi về đăng ký khai sinh, chứng thực giấy tờ, hoặc các thủ tục hành chính khác. Tôi sẽ hỏi thêm thông tin nếu còn thiếu.'

function detectTopic(text) {
  const t = text.toLowerCase()
  if (t.includes('khai sinh') || t.includes('đẻ') || t.includes('sinh con')) return 'khai_sinh'
  if (t.includes('chứng thực') || t.includes('chung thuc') || t.includes('sao y')) return 'chung_thuc'
  return 'general'
}

export async function sendTurn({ message, historyLength }) {
  await new Promise((r) => setTimeout(r, 550 + Math.random() * 400))

  if (historyLength <= 1) {
    return { role: 'assistant', text: WELCOME }
  }

  const topic = detectTopic(message)

  if (topic === 'khai_sinh') {
    return {
      role: 'assistant',
      text:
        'Để hướng dẫn đăng ký khai sinh, vui lòng cho biết thêm:\n\n' +
        '1. Nơi sinh của trẻ là đâu?\n' +
        '2. Đã có giấy chứng sinh chưa?\n' +
        '3. Cha/mẹ đã đăng ký kết hôn chưa?\n\n' +
        '(Bản demo — sau này câu hỏi lấy từ Decision Engine / procedure JSON.)',
      action: 'ASK_MISSING_SLOTS',
    }
  }

  if (topic === 'chung_thuc') {
    return {
      role: 'assistant',
      text:
        'Với chứng thực bản sao, cần làm rõ:\n\n' +
        '1. Loại giấy tờ cần chứng thực?\n' +
        '2. Số lượng bản sao?\n\n' +
        'Bạn mang bản chính đến Bộ phận Một cửa xã Chư Sê trong giờ hành chính.',
      action: 'ASK_MISSING_SLOTS',
    }
  }

  return {
    role: 'assistant',
    text:
      'Tôi đã ghi nhận câu hỏi của bạn. Hiện giao diện đang chạy chế độ demo (chưa nối API).\n\n' +
      'Thử hỏi: “Đăng ký khai sinh cần gì?” hoặc “Chứng thực giấy tờ như thế nào?”',
    action: 'DIRECT_ANSWER',
  }
}

export { WELCOME }
