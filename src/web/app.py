import os
import sys
import json
import logging
from flask import Flask, render_template, request, jsonify

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.web.demo_data import DemoDataGenerator
from src.web.map_generator import MapGenerator

logger = logging.getLogger(__name__)

# Module-level data cache
_fire_df = None
_facilities_gdf = None
_map_gen = MapGenerator()
_demo_gen = DemoDataGenerator()

def get_data():
    """Get fire and facility data, prioritizing persistent SQLite database."""
    global _fire_df, _facilities_gdf
    if _fire_df is None:
        try:
            from src.pipeline_automation.database import FireMonitoringDatabase
            db = FireMonitoringDatabase()
            db_fires = db.get_all_fires(limit=3000)
            if not db_fires.empty:
                _fire_df = db_fires
                logger.info(f"Loaded {len(_fire_df)} fire records from SQLite persistent storage.")
        except Exception as e:
            logger.warning(f"Could not load from SQLite database ({e}).")

    if _facilities_gdf is None:
        try:
            import geopandas as gpd
            from config import settings
            fac_path = settings.PROCESSED_DATA_DIR / "industrial_facilities.geojson"
            if fac_path.exists():
                _facilities_gdf = gpd.read_file(fac_path)
        except Exception:
            pass

    if _facilities_gdf is None or _facilities_gdf.empty:
        logger.info("Generating facility data...")
        _facilities_gdf = _demo_gen.generate_facilities()

    if _fire_df is None or _fire_df.empty:
        logger.info("Database empty; generating demo fire data for dashboard preview...")
        _fire_df = _demo_gen.generate_fire_data(n_fires=500, days_back=3)

    return _fire_df, _facilities_gdf

def create_app():
    """Flask application factory."""
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    app.config['SECRET_KEY'] = 'sih-2026-industrial-fire-detection'
    
    @app.route('/')
    def index():
        """Main dashboard page."""
        fire_df, facilities_gdf = get_data()
        
        # Get filter parameters
        selected_types = request.args.getlist('fire_type')
        if not selected_types:
            selected_types = list(MapGenerator.FIRE_TYPE_COLORS.keys())
        
        # Generate map
        map_html = _map_gen.create_dashboard_map(fire_df, facilities_gdf, selected_types)
        
        # Get statistics
        stats = _map_gen.get_fire_statistics(fire_df)
        
        # Get all available fire types for filter checkboxes
        all_fire_types = list(MapGenerator.FIRE_TYPE_COLORS.keys())
        
        return render_template('dashboard.html',
                             map_html=map_html,
                             stats=stats,
                             all_fire_types=all_fire_types,
                             selected_types=selected_types,
                             fire_type_colors=MapGenerator.FIRE_TYPE_COLORS)
    
    @app.route('/map')
    def render_map():
        """Render standalone Folium map."""
        fire_df, facilities_gdf = get_data()
        selected_types = request.args.getlist('fire_type')
        if not selected_types:
            selected_types = list(MapGenerator.FIRE_TYPE_COLORS.keys())
        return _map_gen.create_dashboard_map(fire_df, facilities_gdf, selected_types)
    
    @app.route('/api/stats')
    def api_stats():
        """Return fire statistics as JSON."""
        fire_df, _ = get_data()
        stats = _map_gen.get_fire_statistics(fire_df)
        return jsonify(stats)
    
    @app.route('/api/fires')
    def api_fires():
        """Return fire data as JSON."""
        fire_df, _ = get_data()
        # Convert datetime to string for JSON serialization
        df_copy = fire_df.copy()
        if 'datetime' in df_copy.columns:
            df_copy['datetime'] = df_copy['datetime'].astype(str)
        return jsonify(df_copy.to_dict(orient='records'))
    
    @app.route('/api/facilities')
    def api_facilities():
        """Return facility data as JSON."""
        _, facilities_gdf = get_data()
        if facilities_gdf is not None and not facilities_gdf.empty:
            df_copy = facilities_gdf.copy()
            if 'geometry' in df_copy.columns:
                df_copy = df_copy.drop(columns=['geometry'])
            return jsonify(df_copy.to_dict(orient='records'))
        return jsonify([])
    
    @app.route('/api/refresh', methods=['GET', 'POST'])
    def api_refresh():
        """Regenerate / reload data cache."""
        global _fire_df, _facilities_gdf
        _fire_df = None
        _facilities_gdf = None
        get_data()
        return jsonify({"status": "ok", "message": "Data refreshed from persistent storage/source"})

    @app.route('/api/pipeline/status')
    def api_pipeline_status():
        """Return automated pipeline status and SQLite database metrics."""
        try:
            from src.pipeline_automation.database import FireMonitoringDatabase
            db = FireMonitoringDatabase()
            stats = db.get_statistics()
            recent_runs = db.get_recent_pipeline_runs(limit=5)
            return jsonify({
                "status": "active",
                "database_path": str(db.db_path),
                "statistics": stats,
                "recent_runs": recent_runs
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/api/pipeline/run', methods=['POST', 'GET'])
    def api_pipeline_run():
        """Trigger an on-demand incremental pipeline run."""
        try:
            from src.pipeline_automation.automated_pipeline import AutomatedPipeline
            simulate = request.args.get('simulate', 'false').lower() in ['true', '1']
            pipeline = AutomatedPipeline()
            result = pipeline.run_pipeline(simulate=simulate)
            
            # Invalidate cache so next page load shows new fires
            global _fire_df, _facilities_gdf
            _fire_df = None
            _facilities_gdf = None
            
            return jsonify(result)
        except Exception as e:
            return jsonify({"status": "failed", "error": str(e)}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
