import os
import sys
import subprocess
import glob

# Ensure ftfy is installed
try:
    import ftfy
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ftfy"])
    import ftfy

def fix_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # ftfy will magically fix mixed mojibake like "Không" -> "Không", "🌊" -> "🌊"
        fixed = ftfy.fix_text(content)
        
        if fixed != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed)
            print(f"Fixed mojibake in {os.path.basename(filepath)}")
        else:
            print(f"No mojibake found in {os.path.basename(filepath)}")
    except Exception as e:
        print(f"Failed processing {filepath}: {e}")

if __name__ == '__main__':
    os.chdir(r"D:\Flood_SOS_app_mobile\Gop_app")
    for file in glob.glob("*.py"):
        fix_file(file)
