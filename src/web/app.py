import os
import sys
import json
import logging
import pandas as pd
from typing import Optional, List, Any, Dict
from flask import Flask, render_template, request, jsonify, Response

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.web.demo_data import DemoDataGenerator
from src.web.map_generator import MapGenerator
from src.visualization.analytics_panels import AnalyticsEngine
from src.monitoring import get_alert_dispatcher, AlertEngine, AlertSeverity

logger = logging.getLogger(__name__)

# Module-level data cache
_fire_df = None
_facilities_gdf = None
_map_gen = MapGenerator()
_demo_gen = DemoDataGenerator()
_analytics_engine = AnalyticsEngine()
_alert_dispatcher = get_alert_dispatcher()
_alert_engine = AlertEngine(dispatcher=_alert_dispatcher)


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

        # Retrieve alert statistics and recent alerts for dashboard HUD
        alert_stats = {"total_alerts": 0, "active_alerts": 0, "critical_active_alerts": 0}
        recent_alerts = []
        try:
            from src.pipeline_automation.database import FireMonitoringDatabase
            db = FireMonitoringDatabase()
            _alert_dispatcher.set_database(db)
            alert_stats = db.get_alert_statistics()
            recent_alerts = db.get_alerts(limit=10)
        except Exception as e:
            logger.debug(f"Could not load alert stats from DB: {e}")
            recent_alerts = _alert_dispatcher.dashboard_channel.get_recent_alerts(limit=10)
            alert_stats = {
                "total_alerts": len(recent_alerts),
                "active_alerts": len([a for a in recent_alerts if a.get("status") == "ACTIVE"]),
                "critical_active_alerts": len([a for a in recent_alerts if a.get("status") == "ACTIVE" and a.get("severity") == "CRITICAL"]),
            }
        
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
                             total_filtered=len(filtered_fires),
                             alert_stats=alert_stats,
                             recent_alerts=recent_alerts)
    
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

    # =========================================================================
    # Part 5.2: Multi-Channel Alert System Web & API Routes
    # =========================================================================

    @app.route('/alerts')
    def alerts_center():
        """Render dedicated Alert Incident Management & Operations Center UI."""
        status_filter = request.args.get('status')
        severity_filter = request.args.get('severity')
        
        alerts_list = []
        stats = {"total_alerts": 0, "active_alerts": 0, "critical_active_alerts": 0, "by_severity": {}, "by_trigger": {}}

        try:
            from src.pipeline_automation.database import FireMonitoringDatabase
            db = FireMonitoringDatabase()
            _alert_dispatcher.set_database(db)
            alerts_list = db.get_alerts(limit=100, severity=severity_filter, status=status_filter)
            stats = db.get_alert_statistics()
        except Exception as e:
            logger.debug(f"Could not load alerts from DB ({e}); using in-memory channel.")
            alerts_list = _alert_dispatcher.dashboard_channel.get_recent_alerts(limit=50)

        return render_template(
            'alerts.html',
            alerts=alerts_list,
            stats=stats,
            current_status=status_filter or 'ALL',
            current_severity=severity_filter or 'ALL',
        )

    @app.route('/api/alerts')
    def api_alerts():
        """Return JSON list of alerts with optional filtering."""
        limit = int(request.args.get('limit', 50))
        severity = request.args.get('severity')
        status = request.args.get('status')
        trigger_type = request.args.get('trigger_type')

        try:
            from src.pipeline_automation.database import FireMonitoringDatabase
            db = FireMonitoringDatabase()
            _alert_dispatcher.set_database(db)
            alerts = db.get_alerts(limit=limit, severity=severity, status=status, trigger_type=trigger_type)
            return jsonify(alerts)
        except Exception as e:
            logger.warning(f"Falling back to in-memory alerts ({e})")
            raw_alerts = _alert_dispatcher.dashboard_channel.get_recent_alerts(limit=limit)
            if severity:
                raw_alerts = [a for a in raw_alerts if a.get("severity", "").upper() == severity.upper()]
            if status:
                raw_alerts = [a for a in raw_alerts if a.get("status", "").upper() == status.upper()]
            return jsonify(raw_alerts)

    @app.route('/api/alerts/stats')
    def api_alert_stats():
        """Return aggregate summary metrics of alerts."""
        try:
            from src.pipeline_automation.database import FireMonitoringDatabase
            db = FireMonitoringDatabase()
            _alert_dispatcher.set_database(db)
            stats = db.get_alert_statistics()
            return jsonify(stats)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/alerts/<alert_id>/ack', methods=['POST'])
    def api_acknowledge_alert(alert_id: str):
        """Acknowledge an active alert by alert_id."""
        user = "operator"
        if request.is_json and request.json:
            user = request.json.get("user", "operator")
        try:
            from src.pipeline_automation.database import FireMonitoringDatabase
            db = FireMonitoringDatabase()
            success = db.acknowledge_alert(alert_id, acknowledged_by=user)
            return jsonify({"status": "success" if success else "not_found", "alert_id": alert_id})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/api/alerts/test', methods=['POST', 'GET'])
    def api_trigger_test_alert():
        """Trigger an immediate simulated test alert across all channels."""
        try:
            from src.pipeline_automation.database import FireMonitoringDatabase
            db = FireMonitoringDatabase()
            _alert_dispatcher.set_database(db)
            test_alert = _alert_engine.trigger_test_alert()
            return jsonify({
                "status": "dispatched",
                "alert": test_alert.to_dict(),
                "channels": test_alert.channels_dispatched
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/api/alerts/stream')
    def api_alerts_stream():
        """
        Server-Sent Events (SSE) real-time streaming endpoint.
        Pushes new thermal anomaly alerts directly to client browsers.
        """
        return Response(
            _alert_dispatcher.dashboard_channel.sse_event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    @app.route('/reports')
    def reports_console():
        """Historical Fire Analytics & Executive Reporting Console (Part 5.4)."""
        fires_df, facilities_gdf = get_data()
        from src.reporting.report_engine import ReportEngine
        engine = ReportEngine()
        analysis = engine.run_full_analysis(fires_df, facilities_gdf)
        return render_template(
            "reports.html",
            summary=analysis["summary"],
            monthly_data=analysis["monthly_data"],
            quarterly_data=analysis["quarterly_data"],
            facility_risks=analysis["facility_risks"],
            regional_trends=analysis["regional_trends"]
        )

    @app.route('/health')
    def web_health():
        """Health and readiness probe endpoint (Part 5.5)."""
        from src.monitoring.health import SystemHealthManager
        manager = SystemHealthManager()
        readiness = manager.get_readiness()
        code = 200 if readiness.get("status") == "ready" else 503
        return jsonify(readiness), code

    @app.route('/api/health/deep')
    def web_deep_health():
        """Deep diagnostic health inspection endpoint (Part 5.5)."""
        from src.monitoring.health import SystemHealthManager
        manager = SystemHealthManager()
        return jsonify(manager.get_deep_diagnostics())

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
