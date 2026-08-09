# Seeds — Phase 0

Các file JSON thủ tục mẫu để lock schema + runtime trước khi có PDF thật.

| File | procedure_id | Domain | Route kỳ vọng |
|------|--------------|--------|---------------|
| `dk_khai_sinh.json` | dk_khai_sinh | Hộ tịch | ask → final |
| `chung_thuc_ban_sao.json` | chung_thuc_ban_sao | Hộ tịch | DIRECT_ANSWER |
| `xin_giay_phep_xay_dung.json` | xin_giay_phep_xay_dung | Đất đai/GPXD | ask → final |
| `tra_cuu_quy_hoach.json` | tra_cuu_quy_hoach | Quy hoạch | ask → final |
| `dk_bhyt_ho_gia_dinh.json` | dk_bhyt_ho_gia_dinh | BHYT | ask → final |

## Quy tắc

- `status` hiện tại: `DRAFT`
- `citations[].source_type`: `manual_seed`
- `metadata.needs_official_pdf`: `true`
- Không active production cho đến khi cán bộ xã review + thay nguồn official

## Validate local (khi có ajv/jsonschema)

```bash
# ví dụ
npx --yes ajv-cli validate -s ../schemas/procedure_definition.schema.json -d ./dk_khai_sinh.json
```
