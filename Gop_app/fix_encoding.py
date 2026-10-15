import glob
import os

def fix_mojibake(filepath):
    # Read as raw bytes
    with open(filepath, 'rb') as f:
        content = f.read()

    # Try decoding as UTF-8
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        return # Not utf-8, skip

    # Check for common mojibake patterns
    if 'Ã' not in text and 'á»' not in text and 'ðŸ' not in text:
        return # No obvious mojibake

    try:
        # Re-encode to iso-8859-1 (1-to-1 mapping), then decode back from UTF-8
        fixed_text = text.encode('iso-8859-1').decode('utf-8')
        
        # Save it back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_text)
        print(f"Fixed mojibake in {os.path.basename(filepath)}")
    except Exception as e:
        print(f"Failed to fix {os.path.basename(filepath)}: {e}")

if __name__ == '__main__':
    # Fix all python files in Gop_app
    search_path = r'D:\Flood_SOS_app_mobile\Gop_app\*.py'
    for file in glob.glob(search_path):
        fix_mojibake(file)
