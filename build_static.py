import os
import sys
import json
import jinja2

# Ensure src/web in path
sys.path.insert(0, os.path.join(os.path.abspath("."), "src", "web"))
sys.path.insert(0, os.path.abspath("."))

from demo_data import DemoDataGenerator
from map_generator import MapGenerator

def build():
    print("Generating demo data for static deployment...")
    gen = DemoDataGenerator()
    facs_gdf = gen.generate_facilities()
    fire_df = gen.generate_fire_data(500, 3)
    
    # Format fire dataframe
    df_copy = fire_df.copy()
    if 'datetime' in df_copy.columns:
        df_copy['datetime'] = df_copy['datetime'].astype(str)
    fires_list = df_copy.to_dict(orient='records')
    
    # Format facilities dataframe
    fac_copy = facs_gdf.copy()
    if 'geometry' in fac_copy.columns:
        fac_copy = fac_copy.drop(columns=['geometry'])
    facs_list = fac_copy.to_dict(orient='records')
    
    map_gen = MapGenerator()
    stats = map_gen.get_fire_statistics(fire_df)
    all_fire_types = list(MapGenerator.FIRE_TYPE_COLORS.keys())
    
    # Copy theme.css for static hosting
    import shutil
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("docs/static/css", exist_ok=True)
    shutil.copy("src/web/static/css/theme.css", "static/css/theme.css")
    shutil.copy("src/web/static/css/theme.css", "docs/static/css/theme.css")
    
    # Load template
    template_path = os.path.join("src", "web", "templates", "dashboard.html")
    with open(template_path, "r", encoding="utf-8") as f:
        template_content = f.read()
    
    env = jinja2.Environment()
    # Add tojson filter simulation
    env.filters['tojson'] = lambda val: json.dumps(val)
    template = env.from_string(template_content)

    alert_stats = {"critical_active_alerts": 12, "total_alerts": 48}
    recent_alerts = [{
        "alert_id": "ALT-935401CC",
        "title": "Severe Thermal Flare Hazard",
        "severity": "CRITICAL",
        "facility_name": "Jamnagar Petrochemical Complex",
        "timestamp": "2026-09-09 18:30"
    }]
    
    rendered = template.render(
        fires_data=fires_list,
        facilities_data=facs_list,
        stats=stats,
        all_fire_types=all_fire_types,
        selected_types=all_fire_types,
        fire_type_colors=MapGenerator.FIRE_TYPE_COLORS,
        map_html="",
        alert_stats=alert_stats,
        recent_alerts=recent_alerts,
        crit_count=12,
        search_query="",
        start_date="",
        end_date="",
        min_confidence=0,
    )
    
    # Write to root index.html
    root_rendered = rendered.replace(
        '<link rel="stylesheet" href="/static/css/theme.css">',
        '<link rel="stylesheet" href="static/css/theme.css">\n    <link rel="stylesheet" href="/static/css/theme.css">'
    )
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(root_rendered)
    print("Written index.html")
    
    # Write to docs/index.html
    docs_rendered = rendered.replace(
        '<link rel="stylesheet" href="/static/css/theme.css">',
        '<link rel="stylesheet" href="static/css/theme.css">\n    <link rel="stylesheet" href="../src/web/static/css/theme.css">'
    )
    os.makedirs("docs", exist_ok=True)
    with open(os.path.join("docs", "index.html"), "w", encoding="utf-8") as f:
        f.write(docs_rendered)
    print("Written docs/index.html")
    print("Static build successful!")

if __name__ == "__main__":
    build()
