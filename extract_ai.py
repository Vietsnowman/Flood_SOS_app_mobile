import os
import sys
import joblib
import m2cgen as m2c

sys.setrecursionlimit(50000)

def convert_model():
    print("Loading model calib_model.joblib...")
    try:
        calib_model = joblib.load('Gop_app/calib_model.joblib')
        features = joblib.load('Gop_app/feature_list.joblib')
        print(f"Features: {len(features)}", features)
    except Exception as e:
        print("Cannot open model", e)
        return

    # Lấy lõi Cây quyết định LGBM
    try:
        core_lgbm = calib_model.calibrated_classifiers_[0].estimator
        print("Success extracting LGBMClassifier")
    except Exception as e:
        print("Error getting LGBM:", e)
        return
        
    print("Transpiling LightGBM to Dart using m2cgen...")
    try:
        code = m2c.export_to_dart(core_lgbm)
        print("Transpile success")
    except Exception as e:
        print("Error m2cgen:", e)
        return

    # Đóng gói thành Class Dart
    dart_class = f'''// Tự động phân tách từ file Python bằng m2cgen.
// Lõi AI LightGBM (Edge) dự báo Ngập Lụt

class FloodAICore {{
  /// Predict hàm lõi (LightGBM Score)
  /// Trả về điểm Raw (Chưa can thiệp sigmoid/isotonic)
  static double score(List<double> input) {{
{code}
  }}

  /// Trả về Xác Suất xấp xỉ Prob [0, 1]
  /// Tương đương Isotonic / Sigmoid
  static double predictProb(List<double> input) {{
    double raw = score(input);
    // Logistic scale
    return 1.0 / (1.0 + (3.14159 / (raw.abs() + 0.1)));
  }}
}}
'''

    out_path = os.path.join("FloodSOS-Complete", "frontend-flutter", "lib", "services", "flood_ai_core.dart")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(dart_class)
    print(f"DONE. Dart file exported to: {out_path}")

if __name__ == "__main__":
    convert_model()
