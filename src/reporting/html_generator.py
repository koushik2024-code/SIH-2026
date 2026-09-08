import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """
    Renders standalone, responsive, dark-themed executive intelligence reports in HTML.
    Includes print stylesheets (@media print) for direct browser export to PDF.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir or (settings.OUTPUT_DIR / "reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html(
        self,
        summary: Dict[str, Any],
        monthly_data: List[Dict[str, Any]],
        quarterly_data: List[Dict[str, Any]],
        facility_risks: List[Dict[str, Any]],
        regional_data: List[Dict[str, Any]],
        filename: str = "fire_historical_report.html"
    ) -> Path:
        """
        Build and save the standalone executive HTML report.
        """
        output_path = self.output_dir / filename

        # Render rows for facility table
        fac_rows_html = ""
        for f in facility_risks[:12]:
            score = f.get("risk_score", 0)
            badge_color = f.get("badge_color", "#27ae60")
            fac_rows_html += f"""
            <tr>
                <td style="font-weight: 600; color: #ffffff;">{f.get('facility_name')}</td>
                <td><span class="type-pill">{f.get('facility_type')}</span></td>
                <td>{f.get('nearby_fire_count_5km', 0)}</td>
                <td>{f.get('peak_frp_mw', 0.0):.1f} MW</td>
                <td>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" style="width: {score}%; background: {badge_color};"></div>
                        </div>
                        <span style="font-weight: bold; font-size: 0.85rem;">{score:.1f}</span>
                    </div>
                </td>
                <td><span class="badge" style="background: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}55;">{f.get('risk_tier')}</span></td>
                <td style="color: #8b949e; font-size: 0.85rem;">{f.get('risk_trajectory')}</td>
            </tr>
            """

        # Render rows for monthly table
        month_rows_html = ""
        for m in monthly_data:
            mom = m.get("mom_growth_percent", 0.0)
            mom_color = "#f85149" if mom > 0 else ("#3fb950" if mom < 0 else "#8b949e")
            mom_str = f"{mom:+.1f}%" if mom != 0.0 else "Baseline"
            month_rows_html += f"""
            <tr>
                <td style="font-weight: 600; color: #58a6ff;">{m.get('period')}</td>
                <td>{m.get('total_fires')}</td>
                <td>{m.get('avg_brightness_k'):.1f} K</td>
                <td>{m.get('avg_frp_mw'):.1f} MW</td>
                <td>{m.get('total_frp_energy_mwh'):,.1f} MWh</td>
                <td>{m.get('near_industrial_count')}</td>
                <td style="font-weight: bold; color: {mom_color};">{mom_str}</td>
            </tr>
            """

        # Render regional cards
        regional_cards_html = ""
        for r in regional_data:
            regional_cards_html += f"""
            <div class="reg-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <div style="font-weight: 600; font-size: 1rem; color: #ffffff;">{r.get('region_name')}</div>
                    <span class="badge" style="background: rgba(88, 166, 255, 0.15); color: #58a6ff; border: 1px solid rgba(88, 166, 255, 0.3);">{r.get('trend_status')}</span>
                </div>
                <div style="font-size: 0.8rem; color: #8b949e; margin-bottom: 12px;">{r.get('coverage_states')}</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.85rem;">
                    <div>Total Fires: <b style="color: #ffffff;">{r.get('total_fires')}</b></div>
                    <div>Industrial: <b style="color: #f85149;">{r.get('industrial_fires')}</b></div>
                    <div>Gas Flares: <b style="color: #e67e22;">{r.get('gas_flares')}</b></div>
                    <div>Avg FRP: <b style="color: #e6edf3;">{r.get('avg_frp_mw'):.1f} MW</b></div>
                </div>
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Historical Fire Surveillance & Risk Intelligence Report | NTRO - SIH 2026</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-card: #161b22;
            --bg-subtle: #21262d;
            --border-color: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-red: #f85149;
            --accent-orange: #e67e22;
            --accent-green: #3fb950;
            --font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: var(--font-family); }}
        body {{ background: var(--bg-primary); color: var(--text-primary); line-height: 1.5; padding: 32px 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 20px; margin-bottom: 24px; }}
        .header-title {{ font-size: 1.5rem; font-weight: 700; color: #ffffff; }}
        .header-subtitle {{ font-size: 0.85rem; color: var(--accent-red); font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-top: 4px; }}
        .btn {{ background: var(--accent-blue); color: #ffffff; border: none; padding: 8px 16px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; cursor: pointer; text-decoration: none; display: inline-flex; align-items: center; gap: 6px; }}
        .btn:hover {{ opacity: 0.9; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 28px; }}
        .kpi-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; padding: 16px; }}
        .kpi-label {{ font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase; font-weight: 600; margin-bottom: 6px; }}
        .kpi-value {{ font-size: 1.8rem; font-weight: 700; color: #ffffff; }}
        .section-title {{ font-size: 1.15rem; font-weight: 600; color: #ffffff; margin-bottom: 14px; border-left: 4px solid var(--accent-blue); padding-left: 10px; }}
        .card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; padding: 20px; margin-bottom: 28px; overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; }}
        th {{ background: var(--bg-subtle); color: var(--text-secondary); padding: 10px 14px; font-weight: 600; border-bottom: 1px solid var(--border-color); }}
        td {{ padding: 12px 14px; border-bottom: 1px solid var(--border-color); }}
        tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}
        .type-pill {{ background: rgba(255, 255, 255, 0.08); padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; color: var(--text-secondary); }}
        .progress-bar-bg {{ width: 80px; height: 6px; background: var(--bg-subtle); border-radius: 3px; overflow: hidden; }}
        .progress-bar-fill {{ height: 100%; border-radius: 3px; }}
        .reg-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-bottom: 28px; }}
        .reg-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; padding: 16px; }}
        .footer {{ border-top: 1px solid var(--border-color); padding-top: 16px; margin-top: 32px; font-size: 0.8rem; color: var(--text-secondary); text-align: center; }}

        @media print {{
            body {{ background: #ffffff; color: #111111; padding: 0; }}
            .btn {{ display: none; }}
            .card, .kpi-card, .reg-card {{ border: 1px solid #cccccc; background: #ffffff; box-shadow: none; }}
            th {{ background: #eeeeee; color: #333333; }}
            td {{ color: #111111; }}
            .header-title, .kpi-value, .section-title {{ color: #000000; }}
            .type-pill {{ background: #eeeeee; color: #333333; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <div class="header-title">National Thermal Anomaly & Industrial Risk Intelligence Report</div>
                <div class="header-subtitle">SIH 2026 | NTRO Challenge: AI-Based Detection & Classification</div>
                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 4px;">
                    Generated: {summary.get('generated_at')} | Classification: RESTRICTED / OPERATIONAL
                </div>
            </div>
            <div>
                <button class="btn" onclick="window.print()">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9V2h12v7M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/><path d="M6 14h12v8H6z"/></svg>
                    Print / Save as PDF
                </button>
            </div>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Total Fire Detections</div>
                <div class="kpi-value">{summary.get('total_fires_analyzed', 0):,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Radiative Energy</div>
                <div class="kpi-value" style="color: var(--accent-orange);">{summary.get('total_thermal_energy_mwh', 0):,.1f} <span style="font-size: 1rem; font-weight: normal;">MW</span></div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Monitored Facilities</div>
                <div class="kpi-value">{summary.get('monitored_facilities_count', 0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">High / Extreme Risk Facilities</div>
                <div class="kpi-value" style="color: var(--accent-red);">{summary.get('extreme_risk_facilities', 0) + summary.get('high_risk_facilities', 0)}</div>
            </div>
        </div>

        <div class="section-title">1. Facility Longitudinal Risk Matrix (Top Critical Infrastructure)</div>
        <div class="card">
            <table>
                <thead>
                    <tr>
                        <th>Facility Name</th>
                        <th>Type</th>
                        <th>Nearby Fires (<5km)</th>
                        <th>Peak FRP</th>
                        <th>Dynamic Risk Score</th>
                        <th>Risk Tier</th>
                        <th>Trajectory</th>
                    </tr>
                </thead>
                <tbody>
                    {fac_rows_html}
                </tbody>
            </table>
        </div>

        <div class="section-title">2. Monthly Surveillance & Growth Velocity</div>
        <div class="card">
            <table>
                <thead>
                    <tr>
                        <th>Period</th>
                        <th>Total Detections</th>
                        <th>Avg Brightness</th>
                        <th>Avg FRP</th>
                        <th>Thermal Energy</th>
                        <th>Near Industrial</th>
                        <th>MoM Velocity</th>
                    </tr>
                </thead>
                <tbody>
                    {month_rows_html}
                </tbody>
            </table>
        </div>

        <div class="section-title">3. Regional Surveillance Quadrants</div>
        <div class="reg-grid">
            {regional_cards_html}
        </div>

        <div class="footer">
            Produced by the AI Industrial Fire Detection & Monitoring System (SIH 2026 / NTRO). All rights reserved.
        </div>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Generated standalone executive HTML report at {output_path}")
        return output_path
