import glob
import os

def check_chars(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            bad = set()
            for c in text:
                if ord(c) > 255:
                    bad.add((c, ord(c)))
            if bad:
                print(f"File {os.path.basename(filepath)} has chars >255: {bad}")
    except Exception as e:
        print(e)

if __name__ == '__main__':
    search_path = r'D:\Flood_SOS_app_mobile\Gop_app\*.py'
    for file in glob.glob(search_path):
        if 'app6.py' in file or 'app_SOS_shelters7.py' in file:
            check_chars(file)
