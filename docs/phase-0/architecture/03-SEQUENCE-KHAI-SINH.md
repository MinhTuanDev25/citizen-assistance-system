# Sequence — Đăng ký khai sinh (reference)

```text
Citizen                API                 ConversationMgr          DecisionEngine           ProcedureStore
  |                     |                        |                        |                        |
  | "làm khai sinh..."  |                        |                        |                        |
  |-------------------->|                        |                        |                        |
  |                     | detect intent/domain   |                        |                        |
  |                     |----------------------->|                        |                        |
  |                     |                        | load active JSON       |                        |
  |                     |                        |----------------------->|----------------------->|
  |                     |                        |                        | missing = 3 slots      |
  |                     |                        |<-----------------------| ASK_MISSING_SLOTS      |
  |<--------------------| hỏi noi_sinh           |                        |                        |
  | "sinh BV tỉnh"      |                        |                        |                        |
  |-------------------->| extract + update state |                        |                        |
  |                     |----------------------->|----------------------->| missing = 2            |
  |<--------------------| hỏi da_ket_hon         |                        |                        |
  | ...                 |                        |                        |                        |
  | đủ slots            |                        |                        |                        |
  |                     |                        |                        | PROVIDE_FINAL_GUIDANCE |
  |<--------------------| checklist + citations  |                        |                        |
```

Nhánh direct (chứng thực bản sao): detect procedure → required_slots rỗng → `DIRECT_ANSWER` ngay.
