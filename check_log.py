with open(r"D:\Flood_SOS_app_mobile\FloodSOS-Complete\frontend-flutter\build_log.txt", "rb") as f:
    text = f.read().decode("utf-16", errors="ignore")
    if "flutter assemble" not in text:
        # maybe it's utf-8
        f.seek(0)
        text = f.read().decode("utf-8", errors="ignore")

lines = text.splitlines()
out_lines = [line for line in lines if "error" in line.lower() or "failed" in line.lower()]
with open(r"D:\Flood_SOS_app_mobile\extracted_log.txt", "w", encoding="utf-8") as out:
    out.write("\n".join(out_lines[-50:]))
    out.write("\n--- END ERRORS ---\n")
    out.write("\n".join(lines[-50:]))
