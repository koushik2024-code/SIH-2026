import os
import sys
import json
import shutil
import jinja2

# Ensure project root & src/web are in sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src", "web"))
sys.path.insert(0, BASE_DIR)

from src.web.app import get_data, _analytics_engine
from src.web.map_generator import MapGenerator
from src.reporting.report_engine import ReportEngine
import re
from src.pipeline_automation.database import FireMonitoringDatabase


def adapt_links_for_static(html: str) -> str:
    """Transform server-side absolute routes into static relative filenames for GitHub Pages & static hosting."""
    # 1. API route replacements
    html = html.replace('href="/api/reports/export/pdf"', 'href="static/reports/fire_historical_report.pdf" download="fire_historical_report.pdf"')
    html = html.replace('href="/api/reports/export/html"', 'href="static/reports/fire_historical_report.html" target="_blank"')
    html = html.replace('href="/api/analytics"', 'href="static/reports/analytics_summary.json" target="_blank"')

    # 2. Query string routes (e.g. /alerts?severity=CRITICAL, /?search=XYZ)
    html = re.sub(r'href="/alerts\?', 'href="alerts.html?', html)
    html = re.sub(r'href="/reports\?', 'href="reports.html?', html)
    html = re.sub(r'href="/analytics\?', 'href="analytics.html?', html)
    html = re.sub(r'href="/\?', 'href="index.html?', html)

    # 3. Standard page routes
    replacements = [
        ('href="/"', 'href="index.html"'),
        ("href='/'", "href='index.html'"),
        ('href="/analytics"', 'href="analytics.html"'),
        ('href="/alerts"', 'href="alerts.html"'),
        ('href="/reports"', 'href="reports.html"'),
        ('href="/map"', 'href="map.html"'),
        ('href="/static/css/theme.css"', 'href="static/css/theme.css"'),
        ('mapFrame.src = "/map" + window.location.search;', 'mapFrame.src = "map.html" + window.location.search;'),
        ('mapFrame.src = "/map"', 'mapFrame.src = "map.html"'),
        ('href="map.html"', 'href="map.html"'),
    ]
    for old, new in replacements:
        html = html.replace(old, new)
    return html


def build():
    print("=" * 60)
    print("Building Static Deployment Suite for GitHub Pages & Local Hosting")
    print("=" * 60)

    # 1. Setup asset directories
    os.makedirs(os.path.join(BASE_DIR, "static", "css"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "docs", "static", "css"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "static", "reports"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "docs", "static", "reports"), exist_ok=True)
    
    theme_src = os.path.join(BASE_DIR, "src", "web", "static", "css", "theme.css")
    if os.path.exists(theme_src):
        shutil.copy(theme_src, os.path.join(BASE_DIR, "static", "css", "theme.css"))
        shutil.copy(theme_src, os.path.join(BASE_DIR, "docs", "static", "css", "theme.css"))
        print("[+] Synced theme.css to static/ and docs/static/")

    # Sync pre-generated report documents
    rep_pdf = os.path.join(BASE_DIR, "output", "reports", "fire_historical_report.pdf")
    rep_html = os.path.join(BASE_DIR, "output", "reports", "fire_historical_report.html")
    analytics_json = os.path.join(BASE_DIR, "output", "analytics_summary.json")

    for src_file in [rep_pdf, rep_html, analytics_json]:
        if os.path.exists(src_file):
            shutil.copy(src_file, os.path.join(BASE_DIR, "static", "reports"))
            shutil.copy(src_file, os.path.join(BASE_DIR, "docs", "static", "reports"))
            print(f"[+] Synced {os.path.basename(src_file)} to static/reports/ and docs/static/reports/")

    # 2. Setup Jinja environment
    template_dir = os.path.join(BASE_DIR, "src", "web", "templates")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
    env.filters['tojson'] = lambda val: json.dumps(val)

    # 3. Fetch comprehensive pipeline data
    print("[+] Loading fire detections, infrastructure, and alert history...")
    fire_df, facs_gdf = get_data()

    # Format fires data for JSON embedding
    df_copy = fire_df.copy()
    if 'datetime' in df_copy.columns:
        df_copy['datetime'] = df_copy['datetime'].astype(str)
    fires_list = df_copy.to_dict(orient='records')

    # Format facilities data for JSON embedding
    fac_copy = facs_gdf.copy()
    if 'geometry' in fac_copy.columns:
        fac_copy = fac_copy.drop(columns=['geometry'])
    facs_list = fac_copy.to_dict(orient='records')

    map_gen = MapGenerator()
    stats = map_gen.get_fire_statistics(fire_df)
    all_fire_types = list(MapGenerator.FIRE_TYPE_COLORS.keys())

    # Fetch alerts from database
    db = FireMonitoringDatabase()
    alerts_list = db.get_alerts(limit=100)
    alert_stats = db.get_alert_statistics()

    # Enrich stats with unified near_industrial_count and alert metrics
    stats["near_industrial_count"] = stats.get("industrial_count", 0)
    stats["active_alerts"] = alert_stats.get("active_alerts", 0)
    stats["critical_active_alerts"] = alert_stats.get("critical_active_alerts", 0)

    # Save metrics.json for static client fetching
    metrics_json_str = json.dumps(stats, indent=2)
    with open(os.path.join(BASE_DIR, "static", "reports", "metrics.json"), "w", encoding="utf-8") as f:
        f.write(metrics_json_str)
    with open(os.path.join(BASE_DIR, "docs", "static", "reports", "metrics.json"), "w", encoding="utf-8") as f:
        f.write(metrics_json_str)
    print("[+] Generated static/reports/metrics.json and docs/static/reports/metrics.json")

    # -------------------------------------------------------------
    # 4. Build Page 1: Dashboard (index.html & docs/index.html)
    # -------------------------------------------------------------
    print("[+] Rendering Dashboard (index.html)...")
    t_dash = env.get_template("dashboard.html")
    dash_html = t_dash.render(
        fires_data=fires_list,
        facilities_data=facs_list,
        stats=stats,
        all_fire_types=all_fire_types,
        selected_types=all_fire_types,
        fire_type_colors=MapGenerator.FIRE_TYPE_COLORS,
        map_html="",
        alert_stats=alert_stats,
        recent_alerts=alerts_list[:10],
        crit_count=alert_stats.get("critical_active_alerts", 0),
        search_query="",
        start_date="",
        end_date="",
        min_confidence=0,
    )
    dash_static = adapt_links_for_static(dash_html)
    with open(os.path.join(BASE_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(dash_static)
    with open(os.path.join(BASE_DIR, "docs", "index.html"), "w", encoding="utf-8") as f:
        f.write(dash_static)
    print("    -> Wrote index.html and docs/index.html")

    # -------------------------------------------------------------
    # 5. Build Page 2: Alerts Incident Center (alerts.html & docs/alerts.html)
    # -------------------------------------------------------------
    print("[+] Rendering Alerts Incident Center (alerts.html)...")
    t_alerts = env.get_template("alerts.html")
    alerts_html = t_alerts.render(
        alerts=alerts_list,
        stats=alert_stats,
        current_status="ALL",
        current_severity="ALL",
    )
    alerts_static = adapt_links_for_static(alerts_html)
    with open(os.path.join(BASE_DIR, "alerts.html"), "w", encoding="utf-8") as f:
        f.write(alerts_static)
    with open(os.path.join(BASE_DIR, "docs", "alerts.html"), "w", encoding="utf-8") as f:
        f.write(alerts_static)
    print("    -> Wrote alerts.html and docs/alerts.html")

    # -------------------------------------------------------------
    # 6. Build Page 3: Reports & Longitudinal Dossier (reports.html & docs/reports.html)
    # -------------------------------------------------------------
    print("[+] Rendering Executive Reports Dossier (reports.html)...")
    rep_engine = ReportEngine()
    analysis = rep_engine.run_full_analysis(fire_df, facs_gdf)
    t_reports = env.get_template("reports.html")
    reports_html = t_reports.render(
        summary=analysis["summary"],
        monthly_data=analysis["monthly_data"],
        quarterly_data=analysis["quarterly_data"],
        facility_risks=analysis["facility_risks"],
        regional_trends=analysis["regional_trends"],
    )
    reports_static = adapt_links_for_static(reports_html)
    with open(os.path.join(BASE_DIR, "reports.html"), "w", encoding="utf-8") as f:
        f.write(reports_static)
    with open(os.path.join(BASE_DIR, "docs", "reports.html"), "w", encoding="utf-8") as f:
        f.write(reports_static)
    print("    -> Wrote reports.html and docs/reports.html")

    # -------------------------------------------------------------
    # 7. Build Page 4: Spatial Analytics Panel (analytics.html & docs/analytics.html)
    # -------------------------------------------------------------
    print("[+] Rendering Spatial Analytics (analytics.html)...")
    analytics_data = _analytics_engine.generate_full_analytics(fire_df, facs_gdf)

    # Save analytics.json for static client fetching
    analytics_json_str = json.dumps(analytics_data, indent=2)
    with open(os.path.join(BASE_DIR, "static", "reports", "analytics.json"), "w", encoding="utf-8") as f:
        f.write(analytics_json_str)
    with open(os.path.join(BASE_DIR, "docs", "static", "reports", "analytics.json"), "w", encoding="utf-8") as f:
        f.write(analytics_json_str)
    print("    -> Generated static/reports/analytics.json and docs/static/reports/analytics.json")

    t_analytics = env.get_template("analytics.html")
    analytics_html = t_analytics.render(
        **analytics_data,
        start_date="",
        end_date="",
        min_confidence=0,
        search_query="",
        total_fires=len(fire_df),
    )
    analytics_static = adapt_links_for_static(analytics_html)
    with open(os.path.join(BASE_DIR, "analytics.html"), "w", encoding="utf-8") as f:
        f.write(analytics_static)
    with open(os.path.join(BASE_DIR, "docs", "analytics.html"), "w", encoding="utf-8") as f:
        f.write(analytics_static)
    print("    -> Wrote analytics.html and docs/analytics.html")

    print("=" * 60)
    print("All 4 production pages built successfully for both root and docs/!")
    print("=" * 60)


if __name__ == "__main__":
    build()
