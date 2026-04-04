"""
fix_encoding_v2.py — Fix đúng: mojibake (cp1252 đọc sai UTF-8)
Restore .bak rồi encode cp1252 → decode utf-8
"""
import os
import shutil

FILES = [
    r"C:\Flood_SOS_app_mobile\Gop_app\app_SOS_shelters7.py",
    r"C:\Flood_SOS_app_mobile\Gop_app\app6.py",
]

for path in FILES:
    bak = path + ".bak"
    if not os.path.exists(bak):
        print(f"⚠️  Không có .bak cho: {path}, bỏ qua")
        continue

    # Đọc bak (file gốc chứa mojibake, lưu dưới dạng UTF-8)
    with open(bak, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Kiểm tra mojibake
    sample = content[:2000]
    if "ðŸ" in sample or "Ã" in sample or "â€" in sample:
        print(f"[{os.path.basename(path)}] Phát hiện mojibake — đang sửa...")
        try:
            # Encode ngược lại cp1252 để lấy byte gốc UTF-8, rồi decode UTF-8
            fixed = content.encode("cp1252", errors="replace").decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  Lỗi khi sửa: {e}")
            continue
    else:
        print(f"[{os.path.basename(path)}] Không phát hiện mojibake — bỏ qua")
        continue

    # Ghi file đã fix
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(fixed)

    # Kiểm tra sau fix
    if "ðŸ" not in fixed and "â€" not in fixed:
        print(f"  ✅ Đã sửa thành công: {path}")
    else:
        print(f"  ⚠️  Vẫn còn một số ký tự lỗi (có thể là nested encoding)")

print("\n✅ Hoàn tất! Khởi động lại Streamlit để áp dụng.")
