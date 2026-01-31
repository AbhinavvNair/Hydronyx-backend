"""Generate policy comparison PDF reports."""
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from typing import List, Dict, Any
from datetime import datetime


def generate_policy_comparison_pdf(
    baseline: Dict[str, Any],
    counterfactual: Dict[str, Any],
    params: Dict[str, Any],
    result: Dict[str, Any],
) -> bytes:
    """Generate a PDF report comparing baseline vs counterfactual policy scenario."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 2 * cm
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(HexColor("#06b6d4"))
    c.drawString(margin, y, "Hydronyx Policy Comparison Report")
    y -= 1.5 * cm

    c.setFont("Helvetica", 10)
    c.setFillColor(HexColor("#333333"))
    c.drawString(margin, y, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    y -= 1 * cm

    # Parameters
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Intervention Parameters")
    y -= 0.5 * cm
    c.setFont("Helvetica", 10)
    for k, v in params.items():
        c.drawString(margin + 0.5 * cm, y, f"  {k}: {v}")
        y -= 0.4 * cm
    y -= 0.5 * cm

    # Results
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Simulation Results")
    y -= 0.5 * cm
    c.setFont("Helvetica", 10)
    for k, v in result.items():
        if isinstance(v, float):
            c.drawString(margin + 0.5 * cm, y, f"  {k}: {v:.4f}")
        else:
            c.drawString(margin + 0.5 * cm, y, f"  {k}: {v}")
        y -= 0.4 * cm
    y -= 0.5 * cm

    # Trajectory summary
    bl_traj = baseline.get("baseline_trajectory", []) or baseline.get("trajectory", [])
    cf_traj = counterfactual.get("counterfactual_trajectory", []) or counterfactual.get("trajectory", [])
    if bl_traj and cf_traj:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Trajectory Comparison (first 6 months)")
        y -= 0.5 * cm
        c.setFont("Helvetica", 9)
        for i in range(min(6, len(bl_traj), len(cf_traj))):
            bl = bl_traj[i]
            cf = cf_traj[i]
            bl_gw = bl.get("groundwater", bl.get("gw", 0))
            cf_gw = cf.get("groundwater", cf.get("gw", 0))
            diff = cf_gw - bl_gw if isinstance(bl_gw, (int, float)) and isinstance(cf_gw, (int, float)) else 0
            c.drawString(margin + 0.5 * cm, y, f"  Month {i+1}: Baseline={bl_gw:.2f}m, Counterfactual={cf_gw:.2f}m, Diff={diff:+.2f}m")
            y -= 0.35 * cm
        y -= 0.5 * cm

    # Disclaimer
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(HexColor("#666666"))
    c.drawString(margin, y, "Disclaimer: This report is for decision support only. Verify with field data.")
    y -= 0.5 * cm

    c.save()
    buffer.seek(0)
    return buffer.getvalue()
