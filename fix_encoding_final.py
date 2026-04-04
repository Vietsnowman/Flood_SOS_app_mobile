"""
fix_encoding_final.py
=====================
Fix đúng hoàn toàn: mojibake (UTF-8 bị đọc như cp1252/Latin-1)

Vấn đề chính:
- cp1252 có 5 byte undefined: 0x81, 0x8D, 0x8F, 0x90, 0x9D
  → Python encode('cp1252') raise lỗi hoặc replace bằng '?'
  → Các ký tự tiếng Việt như Đ (U+0110 = C4 90) bị thành 'Ä?'

Giải pháp: smart_encode() dùng Latin-1 bijection (byte = code point cho 0x00-0xFF)
+ cp1252 cho ký tự > U+00FF (emoji mojibake như Ÿ, Œ, Š, €...)
"""
import os

FILES = [
    (r"C:\Flood_SOS_app_mobile\Gop_app\app_SOS_shelters7.py",
     r"C:\Flood_SOS_app_mobile\Gop_app\app_SOS_shelters7.py.bak"),
    (r"C:\Flood_SOS_app_mobile\Gop_app\app6.py",
     r"C:\Flood_SOS_app_mobile\Gop_app\app6.py.bak"),
]


def smart_encode(s: str) -> bytes:
    """
    Encode chuỗi mojibake về byte gốc UTF-8:
    - Char U+0000–U+00FF  → dùng Latin-1 bijection (cp = byte), luôn 1:1
    - Char U+0100+         → dùng cp1252 (dành cho emoji mojibake như Ÿ=0x9F)
    - Không tìm được       → bỏ qua (không replace bằng '?')
    """
    result = bytearray()
    for ch in s:
        cp = ord(ch)
        if cp < 0x100:          # Latin-1 bijection – luôn đúng
            result.append(cp)
        else:                   # Emoji mojibake chars (Ÿ, Œ, Š, €, ...)
            try:
                result.extend(ch.encode('cp1252'))
            except (UnicodeEncodeError, LookupError):
                # Ký tự thực sự không map được - giữ nguyên UTF-8
                result.extend(ch.encode('utf-8'))
    return bytes(result)


def fix_file(src_path: str, bak_path: str):
    # Ưu tiên dùng .bak (file gốc trước các lần fix sai)
    read_path = bak_path if os.path.exists(bak_path) else src_path
    print(f"  Đọc từ: {read_path}")

    with open(read_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Kiểm tra có mojibake không (sample đủ dài)
    sample = content[:5000]
    has_mojibake = any(p in sample for p in ["ðŸ", "Ã", "â€", "á»", "Äiá»", "Ã´ng"])
    if not has_mojibake:
        print(f"  ⚠️  Không phát hiện mojibake trong 5000 ký tự đầu.")
        # Thử thêm ở phần giữa file
        mid = len(content) // 2
        mid_sample = content[mid:mid+2000]
        has_mojibake = any(p in mid_sample for p in ["ðŸ", "Ã", "â€", "á»", "Ã´"])
        if not has_mojibake:
            print(f"  ❌ Bỏ qua.")
            return

    print(f"  Phát hiện mojibake — đang sửa...")

    try:
        raw = smart_encode(content)
        fixed = raw.decode('utf-8', errors='replace')
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        return

    # Kiểm tra kết quả
    bad_remaining = sum(1 for p in ["ðŸ", "Ã´", "á»", "â€"] if p in fixed[:3000])
    good = "🌊" in fixed or "Không" in fixed or "điều" in fixed.lower()

    with open(src_path, 'w', encoding='utf-8', newline='\r\n') as f:
        f.write(fixed)

    if good or bad_remaining == 0:
        print(f"  ✅ Đã sửa thành công: {src_path}")
    else:
        print(f"  ⚠️  Đã sửa nhưng còn {bad_remaining} pattern lỗi")


for (path, bak) in FILES:
    name = os.path.basename(path)
    print(f"\n[{name}]")
    fix_file(path, bak)

print("\n✅ Xong! Khởi động lại Streamlit để xem kết quả.")
