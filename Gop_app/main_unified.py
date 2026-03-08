# main_unified.py
import os
import runpy
import streamlit as st


def _run_child_app(script_path: str):
    """
    Chạy nguyên file app con (app6.py hoặc app_SOS_shelters7.py)
    nhưng tránh lỗi st.set_page_config gọi nhiều lần.
    """
    if not os.path.exists(script_path):
        st.error(f"Không tìm thấy file: {script_path}")
        st.stop()

    # Monkeypatch set_page_config để app con không gọi lại
    original_set_page_config = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None

    try:
        # Chạy file như __main__
        runpy.run_path(script_path, run_name="__main__")
    except Exception as e:
        st.exception(e)
        st.stop()
    finally:
        # Restore
        st.set_page_config = original_set_page_config


def main():
    st.set_page_config(page_title="FLOOD UNIFIED APP", layout="wide")

    st.sidebar.title("🌊 FLOOD UNIFIED APP")
    choice = st.sidebar.radio(
        "Chọn module",
        [
            "1) Realtime SOS + Shelters (Point-based)",
            "2) Forecast theo xã + DB + Phân bổ (Commune-based)",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Chạy đúng 2 app gốc, chỉ gộp UI chọn module.")

    if choice.startswith("1)"):
        st.title("Realtime SOS + Shelters (Point-based)")
        _run_child_app("app_SOS_shelters7.py")

    else:
        st.title("Forecast theo xã + DB + Phân bổ (Commune-based)")
        _run_child_app("app6.py")


if __name__ == "__main__":
    main()