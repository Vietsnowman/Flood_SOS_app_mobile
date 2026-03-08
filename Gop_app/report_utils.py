import pandas as pd
from fpdf import FPDF
import os
from datetime import datetime

def export_csv(df, out_path):
    df.to_csv(out_path, index=False)
    return out_path

def export_pdf(top_df, summary_dict, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, "Flood Rescue Priority Report", ln=True)

    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Generated: {datetime.now().isoformat(timespec='seconds')}", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, "Summary", ln=True)
    pdf.set_font("Arial", size=10)

    for k, v in summary_dict.items():
        pdf.cell(0, 6, f"- {k}: {v}", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, "Top Communes", ln=True)
    pdf.set_font("Arial", size=9)

    cols = list(top_df.columns)
    # simple table
    for c in cols:
        pdf.cell(38, 6, str(c)[:15], border=1)
    pdf.ln()

    for _, row in top_df.iterrows():
        for c in cols:
            pdf.cell(38, 6, str(row[c])[:15], border=1)
        pdf.ln()

    pdf.output(out_path)
    return out_path