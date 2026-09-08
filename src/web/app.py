import os
import sys
import json
import logging
import pandas as pd
from typing import Optional, List, Any, Dict
from flask import Flask, render_template, request, jsonify

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.web.demo_data import DemoDataGenerator
from src.web.map_generator import MapGenerator
from src.visualization.analytics_panels import AnalyticsEngine

logger = logging.getLogger(__name__)

# Module-level data cache
_fire_df = None
_facilities_gdf = None
_map_gen = MapGenerator()
_demo_gen = DemoDataGenerator()
_analytics_engine = AnalyticsEngine()

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

def filter_fire_dataframe(
    df: pd.DataFrame,
    selected_types: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_confidence: Optional[Any] = None,
    search: Optional[str] = None,
) -> pd.DataFrame:
    """Filter fire dataframe by type, date range, confidence threshold, and search query."""
    if df is None or df.empty:
        return pd.DataFrame()

    filtered = df.copy()

    # 1. Fire type filter
    if selected_types and "fire_type" in filtered.columns:
        filtered = filtered[filtered["fire_type"].isin(selected_types)]

    # 2. Date range filter
    if start_date and str(start_date).strip() and "acq_date" in filtered.columns:
        filtered = filtered[filtered["acq_date"].astype(str) >= str(start_date).strip()]
    if end_date and str(end_date).strip() and "acq_date" in filtered.columns:
        filtered = filtered[filtered["acq_date"].astype(str) <= str(end_date).strip()]

    # 3. Confidence threshold filter
    if min_confidence is not None and str(min_confidence).strip() and "confidence" in filtered.columns:
        try:
            min_c = float(min_confidence)
            if min_c > 0:
                def to_num(val):
                    try:
                        return float(val)
                    except Exception:
                        s = str(val).lower().strip()
                        if s in ["h", "high"]:
                            return 85
                        elif s in ["n", "nominal", "medium", "med"]:
                            return 60
                        elif s in ["l", "low"]:
                            return 30
                        return 50
                conf_series = filtered["confidence"].apply(to_num)
                filtered = filtered[conf_series >= min_c]
        except (ValueError, TypeError):
            pass

    # 4. Search query filter
    if search and str(search).strip():
        q = str(search).lower().strip()
        conditions = []
        if "nearest_facility_name" in filtered.columns:
            conditions.append(filtered["nearest_facility_name"].astype(str).str.lower().str.contains(q, na=False))
        if "detection_id" in filtered.columns:
            conditions.append(filtered["detection_id"].astype(str).str.lower().str.contains(q, na=False))
        if "fire_type" in filtered.columns:
            conditions.append(filtered["fire_type"].astype(str).str.lower().str.contains(q, na=False))
        if conditions:
            combined = conditions[0]
            for c in conditions[1:]:
                combined = combined | c
            filtered = filtered[combined]

    return filtered

def create_app():
    """Flask application factory."""
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    app.config['SECRET_KEY'] = 'sih-2026-industrial-fire-detection'
    
    @app.route('/')
    def index():
        """Main dashboard page with interactive controls."""
        fire_df, facilities_gdf = get_data()
        
        # Get filter parameters
        selected_types = request.args.getlist('fire_type')
        if not selected_types:
            selected_types = list(MapGenerator.FIRE_TYPE_COLORS.keys())
        
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        min_confidence = request.args.get('min_confidence', '0')
        search_query = request.args.get('search', '')

        # Filter fire dataset for analytics & visualization
        filtered_fires = filter_fire_dataframe(
            fire_df,
            selected_types=selected_types,
            start_date=start_date,
            end_date=end_date,
            min_confidence=min_confidence,
            search=search_query,
        )
        
        # Generate map
        map_html = _map_gen.create_dashboard_map(filtered_fires, facilities_gdf, selected_types)
        
        # Get statistics (reflecting filtered view or baseline)
        stats = _map_gen.get_fire_statistics(filtered_fires if not filtered_fires.empty else fire_df)
        
        # Overall dataset bounds
        overall_start = str(fire_df['acq_date'].min()) if not fire_df.empty and 'acq_date' in fire_df.columns else ""
        overall_end = str(fire_df['acq_date'].max()) if not fire_df.empty and 'acq_date' in fire_df.columns else ""

        # Extract facility names for autocomplete datalist
        facility_names = []
        if facilities_gdf is not None and not (hasattr(facilities_gdf, 'empty') and facilities_gdf.empty):
            if 'name' in facilities_gdf.columns:
                facility_names = [str(n).strip() for n in facilities_gdf['name'].dropna().unique() if str(n).strip()]

        # Get all available fire types for filter checkboxes
        all_fire_types = list(MapGenerator.FIRE_TYPE_COLORS.keys())
        
        return render_template('dashboard.html',
                             map_html=map_html,
                             stats=stats,
                             all_fire_types=all_fire_types,
                             selected_types=selected_types,
                             fire_type_colors=MapGenerator.FIRE_TYPE_COLORS,
                             start_date=start_date or overall_start,
                             end_date=end_date or overall_end,
                             min_confidence=min_confidence,
                             search_query=search_query,
                             facility_names=facility_names,
                             total_unfiltered=len(fire_df),
                             total_filtered=len(filtered_fires))
    
    @app.route('/map')
    def render_map():
        """Render standalone Folium map with optional query filters."""
        fire_df, facilities_gdf = get_data()
        selected_types = request.args.getlist('fire_type')
        if not selected_types:
            selected_types = list(MapGenerator.FIRE_TYPE_COLORS.keys())

        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        min_confidence = request.args.get('min_confidence', '0')
        search_query = request.args.get('search', '')

        filtered_fires = filter_fire_dataframe(
            fire_df,
            selected_types=selected_types,
            start_date=start_date,
            end_date=end_date,
            min_confidence=min_confidence,
            search=search_query,
        )

        return _map_gen.create_dashboard_map(filtered_fires, facilities_gdf, selected_types)
    
    @app.route('/analytics')
    def analytics():
        """Render comprehensive spatial intelligence and fire analytics panel."""
        fire_df, facilities_gdf = get_data()

        selected_types = request.args.getlist('fire_type')
        if not selected_types:
            selected_types = list(MapGenerator.FIRE_TYPE_COLORS.keys())

        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        min_confidence = request.args.get('min_confidence', '0')
        search_query = request.args.get('search', '')

        filtered_fires = filter_fire_dataframe(
            fire_df,
            selected_types=selected_types,
            start_date=start_date,
            end_date=end_date,
            min_confidence=min_confidence,
            search=search_query,
        )

        analytics_data = _analytics_engine.generate_full_analytics(
            filtered_fires if not filtered_fires.empty else fire_df,
            facilities_gdf
        )

        return render_template(
            'analytics.html',
            **analytics_data,
            start_date=start_date,
            end_date=end_date,
            min_confidence=min_confidence,
            search_query=search_query,
            total_fires=len(filtered_fires),
        )

    @app.route('/api/analytics')
    def api_analytics():
        """Return full structured fire analytics payload as JSON."""
        fire_df, facilities_gdf = get_data()
        selected_types = request.args.getlist('fire_type')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        min_confidence = request.args.get('min_confidence')
        search_query = request.args.get('search')

        filtered_fires = filter_fire_dataframe(
            fire_df,
            selected_types=selected_types if selected_types else None,
            start_date=start_date,
            end_date=end_date,
            min_confidence=min_confidence,
            search=search_query,
        )

        payload = _analytics_engine.generate_full_analytics(
            filtered_fires if not filtered_fires.empty else fire_df,
            facilities_gdf
        )
        return jsonify(payload)

    @app.route('/api/analytics/export', methods=['POST', 'GET'])
    def api_analytics_export():
        """Export static charts and standalone HTML report to disk."""
        fire_df, facilities_gdf = get_data()
        from config import settings
        out_dir = settings.BASE_DIR / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "analytics_dashboard.html"
        _analytics_engine.generate_standalone_report_html(fire_df, facilities_gdf, report_path)
        figures = _analytics_engine.export_matplotlib_charts(fire_df, facilities_gdf, out_dir)
        return jsonify({
            "status": "success",
            "report_path": str(report_path),
            "figures": {k: str(v) for k, v in figures.items()}
        })
    
    @app.route('/api/stats')
    def api_stats():
        """Return fire statistics as JSON with optional filters."""
        fire_df, _ = get_data()
        selected_types = request.args.getlist('fire_type')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        min_confidence = request.args.get('min_confidence')
        search_query = request.args.get('search')

        filtered_fires = filter_fire_dataframe(
            fire_df,
            selected_types=selected_types if selected_types else None,
            start_date=start_date,
            end_date=end_date,
            min_confidence=min_confidence,
            search=search_query,
        )
        stats = _map_gen.get_fire_statistics(filtered_fires)
        return jsonify(stats)
    
    @app.route('/api/fires')
    def api_fires():
        """Return fire data as JSON with optional filters."""
        fire_df, _ = get_data()
        selected_types = request.args.getlist('fire_type')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        min_confidence = request.args.get('min_confidence')
        search_query = request.args.get('search')

        filtered_fires = filter_fire_dataframe(
            fire_df,
            selected_types=selected_types if selected_types else None,
            start_date=start_date,
            end_date=end_date,
            min_confidence=min_confidence,
            search=search_query,
        )

        df_copy = filtered_fires.copy()
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
