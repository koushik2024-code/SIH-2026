import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from config import settings

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class PDFReportGenerator:
    """
    Generates formal, multi-page PDF executive intelligence reports using ReportLab.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir or (settings.OUTPUT_DIR / "reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_pdf(
        self,
        summary: Dict[str, Any],
        monthly_data: List[Dict[str, Any]],
        quarterly_data: List[Dict[str, Any]],
        facility_risks: List[Dict[str, Any]],
        regional_data: List[Dict[str, Any]],
        filename: str = "fire_historical_report.pdf"
    ) -> Path:
        """
        Build and save an executive PDF report document.
        """
        output_path = self.output_dir / filename

        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab is not available. Generating text-based fallback PDF.")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"SIH 2026 / NTRO Executive Fire Report\nGenerated: {summary.get('generated_at')}\n")
                f.write(f"Total Fires: {summary.get('total_fires_analyzed')}\n")
            return output_path

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()
        normal_style = styles["Normal"]

        # Custom typography styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1a1a2e")
        )
        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#d32f2f")
        )
        meta_style = ParagraphStyle(
            "MetaText",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#555555")
        )
        h2_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1a1a2e"),
            spaceBefore=12,
            spaceAfter=6
        )
        cell_header_style = ParagraphStyle(
            "CellHeader",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white
        )
        cell_body_style = ParagraphStyle(
            "CellBody",
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#222222")
        )
        cell_bold_style = ParagraphStyle(
            "CellBold",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#111111")
        )

        story = []

        # 1. Header Banner
        story.append(Paragraph("NATIONAL THERMAL ANOMALY & INDUSTRIAL RISK INTELLIGENCE REPORT", title_style))
        story.append(Spacer(1, 3))
        story.append(Paragraph("SIH 2026 | NTRO CHALLENGE: PERSISTENT THERMAL SOURCE SURVEILLANCE", subtitle_style))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            f"Generated: {summary.get('generated_at', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))} | "
            f"Classification: RESTRICTED / OPERATIONAL DISPATCH | Multi-Sensor VIIRS & MODIS Feed",
            meta_style
        ))
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a1a2e"), spaceAfter=12))

        # 2. Executive Summary Metrics Table
        story.append(Paragraph("1. Executive Summary & Macro Observations", h2_style))
        summary_rows = [
            [
                Paragraph("<b>Total Fire Detections:</b>", cell_body_style),
                Paragraph(str(summary.get("total_fires_analyzed", 0)), cell_bold_style),
                Paragraph("<b>Total Thermal Radiative Energy:</b>", cell_body_style),
                Paragraph(f"{summary.get('total_thermal_energy_mwh', 0):,.1f} MW", cell_bold_style),
            ],
            [
                Paragraph("<b>Monitored Facilities:</b>", cell_body_style),
                Paragraph(str(summary.get("monitored_facilities_count", 0)), cell_bold_style),
                Paragraph("<b>Extreme / High Risk Facilities:</b>", cell_body_style),
                Paragraph(f"{summary.get('extreme_risk_facilities', 0)} Extreme, {summary.get('high_risk_facilities', 0)} High", cell_bold_style),
            ],
            [
                Paragraph("<b>Dominant Fire Category:</b>", cell_body_style),
                Paragraph(str(summary.get("dominant_fire_type", "Industrial")), cell_bold_style),
                Paragraph("<b>Monthly / Quarterly Cycles:</b>", cell_body_style),
                Paragraph(f"{summary.get('monthly_periods', 0)} Months, {summary.get('quarterly_periods', 0)} Quarters", cell_bold_style),
            ],
        ]
        sum_table = Table(summary_rows, colWidths=[130, 130, 140, 130])
        sum_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#dddddd")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e9ecef")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(sum_table)
        story.append(Spacer(1, 14))

        # 3. Top Facility Risk Leaderboard Table
        story.append(Paragraph("2. Critical Facility Risk Matrix (Top Priority Assets)", h2_style))
        fac_headers = ["Facility Name", "Type", "Nearby Fires (<5km)", "Peak FRP (MW)", "Risk Score", "Risk Tier"]
        fac_table_data = [[Paragraph(h, cell_header_style) for h in fac_headers]]

        top_facs = facility_risks[:8]
        for f in top_facs:
            tier_color = colors.HexColor(f.get("badge_color", "#333333"))
            tier_cell = Paragraph(f"<b>{f.get('risk_tier', 'LOW')}</b>", ParagraphStyle("TierCell", parent=cell_bold_style, textColor=tier_color))
            fac_table_data.append([
                Paragraph(f.get("facility_name", "Unknown")[:24], cell_body_style),
                Paragraph(f.get("facility_type", "industrial")[:16], cell_body_style),
                Paragraph(str(f.get("nearby_fire_count_5km", 0)), cell_body_style),
                Paragraph(f"{f.get('peak_frp_mw', 0):.1f}", cell_body_style),
                Paragraph(f"<b>{f.get('risk_score', 0):.1f} / 100</b>", cell_bold_style),
                tier_cell
            ])

        fac_table = Table(fac_table_data, colWidths=[140, 95, 85, 75, 75, 60])
        fac_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(fac_table)
        story.append(Spacer(1, 14))

        # 4. Monthly Breakdown Table
        story.append(Paragraph("3. Monthly Longitudinal Thermal Trend Analysis", h2_style))
        month_headers = ["Period", "Detections", "Avg Brightness", "Avg FRP", "Total Energy (MW)", "Near Industry", "MoM Velocity"]
        month_table_data = [[Paragraph(h, cell_header_style) for h in month_headers]]

        for m in monthly_data[:6]:
            mom_str = f"{m.get('mom_growth_percent', 0.0):+.1f}%" if m.get("mom_growth_percent") != 0.0 else "Baseline"
            mom_color = colors.HexColor("#d32f2f") if m.get('mom_growth_percent', 0.0) > 0 else colors.HexColor("#2e7d32")
            mom_cell = Paragraph(f"<b>{mom_str}</b>", ParagraphStyle("MomCell", parent=cell_bold_style, textColor=mom_color))
            month_table_data.append([
                Paragraph(m.get("period", ""), cell_body_style),
                Paragraph(str(m.get("total_fires", 0)), cell_body_style),
                Paragraph(f"{m.get('avg_brightness_k', 0):.1f} K", cell_body_style),
                Paragraph(f"{m.get('avg_frp_mw', 0):.1f} MW", cell_body_style),
                Paragraph(f"{m.get('total_frp_energy_mwh', 0):,.1f}", cell_body_style),
                Paragraph(str(m.get("near_industrial_count", 0)), cell_body_style),
                mom_cell
            ])

        month_table = Table(month_table_data, colWidths=[80, 70, 80, 75, 85, 70, 70])
        month_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(month_table)
        story.append(Spacer(1, 14))

        # 5. Regional Surveillance Matrix Table
        story.append(Paragraph("4. Regional Spatial Surveillance Quadrants", h2_style))
        reg_headers = ["Surveillance Zone", "Coverage States", "Total Fires", "Industrial", "Gas Flares", "Trend Vector"]
        reg_table_data = [[Paragraph(h, cell_header_style) for h in reg_headers]]

        for r in regional_data:
            reg_table_data.append([
                Paragraph(r.get("region_name", ""), cell_body_style),
                Paragraph(r.get("coverage_states", "")[:28], cell_body_style),
                Paragraph(str(r.get("total_fires", 0)), cell_body_style),
                Paragraph(str(r.get("industrial_fires", 0)), cell_body_style),
                Paragraph(str(r.get("gas_flares", 0)), cell_body_style),
                Paragraph(f"<b>{r.get('trend_status', 'Stable')}</b>", cell_bold_style)
            ])

        reg_table = Table(reg_table_data, colWidths=[130, 140, 60, 65, 65, 70])
        reg_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(reg_table)
        story.append(Spacer(1, 14))

        # 6. Confidentiality Footer
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc"), spaceBefore=10, spaceAfter=8))
        story.append(Paragraph(
            "CONFIDENTIALITY NOTICE: This automated report was produced by the AI-Based Thermal Source Detection System "
            "(SIH 2026 / NTRO). Intended for official defense and disaster response operations.",
            meta_style
        ))

        doc.build(story)
        logger.info(f"Generated publication-grade PDF report at {output_path}")
        return output_path
