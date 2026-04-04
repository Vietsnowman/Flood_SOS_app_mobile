"""
fix_encoding.py — Sửa file bị lỗi double-encoding (UTF-8 đọc như Latin-1)
Chạy: python fix_encoding.py
"""
import os

FILES_TO_FIX = [
    r"C:\Flood_SOS_app_mobile\Gop_app\app_SOS_shelters7.py",
    r"C:\Flood_SOS_app_mobile\Gop_app\app6.py",
]

def fix_file(path: str):
    with open(path, "rb") as f:
        raw = f.read()

    # Thử decode theo Latin-1 rồi encode lại UTF-8 (fix double-encode)
    try:
        fixed = raw.decode("utf-8")
        print(f"  [{path}] đã là UTF-8 hợp lệ — kiểm tra nội dung...")
        # Thử detect nếu chứa ký tự vỡ kiểu Ã, ðŸ
        if "ðŸ" in fixed or "Ã" in fixed or "â€" in fixed:
            print(f"  => Phát hiện double-encode, đang sửa...")
            fixed = raw.decode("latin-1").encode("utf-8").decode("utf-8")
    except UnicodeDecodeError:
        # File là Latin-1, convert sang UTF-8
        fixed = raw.decode("latin-1")
        print(f"  [{path}] Latin-1 → UTF-8")

    # Ghi lại file với BOM-less UTF-8
    backup = path + ".bak"
    os.rename(path, backup)
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(fixed)
    print(f"  ✅ Đã sửa: {path}  (backup: {backup})")

for fp in FILES_TO_FIX:
    if os.path.exists(fp):
        fix_file(fp)
    else:
        print(f"  ⚠️  Không tìm thấy: {fp}")

print("\nHoàn tất! Khởi động lại Streamlit để áp dụng.")
