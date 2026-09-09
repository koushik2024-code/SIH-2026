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
    """Get or generate fire and facility data."""
    global _fire_df, _facilities_gdf
    if _fire_df is None:
        logger.info("Generating demo data...")
        _facilities_gdf = _demo_gen.generate_facilities()
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
        
        # Generate folium map HTML for fallback
        map_html = _map_gen.create_dashboard_map(fire_df, facilities_gdf, selected_types)
        
        # Get statistics
        stats = _map_gen.get_fire_statistics(fire_df)
        
        # Available fire types
        all_fire_types = list(MapGenerator.FIRE_TYPE_COLORS.keys())
        
        # Prepare JSON serializable fire and facility data for real-time frontend interactivity
        df_copy = fire_df.copy()
        if 'datetime' in df_copy.columns:
            df_copy['datetime'] = df_copy['datetime'].astype(str)
        fires_list = df_copy.to_dict(orient='records')
        
        facilities_list = []
        if facilities_gdf is not None and not facilities_gdf.empty:
            fac_copy = facilities_gdf.copy()
            if 'geometry' in fac_copy.columns:
                fac_copy = fac_copy.drop(columns=['geometry'])
            facilities_list = fac_copy.to_dict(orient='records')
        
        return render_template(
            'dashboard.html',
            map_html=map_html,
            stats=stats,
            all_fire_types=all_fire_types,
            selected_types=selected_types,
            fire_type_colors=MapGenerator.FIRE_TYPE_COLORS,
            fires_data=fires_list,
            facilities_data=facilities_list
        )
    
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
    
    @app.route('/api/alerts')
    def api_alerts():
        """Return high-priority alerts."""
        fire_df, _ = get_data()
        df_copy = fire_df.copy()
        if 'datetime' in df_copy.columns:
            df_copy['datetime'] = df_copy['datetime'].astype(str)
        # Filter high priority fires
        alerts = df_copy[
            (df_copy['is_near_industrial'] == 1) | 
            (df_copy['frp'] > 50) | 
            (df_copy['confidence'] == 'high')
        ].sort_values(by='frp', ascending=False).head(30)
        return jsonify(alerts.to_dict(orient='records'))
    
    @app.route('/api/refresh', methods=['GET', 'POST'])
    def api_refresh():
        """Regenerate demo data."""
        global _fire_df, _facilities_gdf
        _fire_df = None
        _facilities_gdf = None
        get_data()
        return jsonify({"status": "ok", "message": "Data refreshed"})
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
