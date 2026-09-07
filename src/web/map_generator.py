import folium
from folium.plugins import HeatMap, MarkerCluster, MiniMap
import pandas as pd
try:
    import geopandas as gpd
except ImportError:
    gpd = None
import logging

logger = logging.getLogger(__name__)

class MapGenerator:
    """Generates interactive Folium maps for fire visualization."""
    
    FIRE_TYPE_COLORS = {
        "Industrial Fire": "#e74c3c",    # Red
        "Gas Flare": "#e67e22",          # Orange
        "Forest Fire": "#27ae60",        # Green
        "Agricultural Burning": "#f1c40f", # Yellow
        "Mining Activity": "#7f8c8d",    # Gray
        "Other/Unknown": "#95a5a6",      # Light gray
    }
    
    FACILITY_ICONS = {
        "oil_refinery": ("tint", "red"),
        "thermal_power_plant": ("bolt", "orange"),
        "steel_plant": ("industry", "darkred"),
        "mining": ("gem", "gray"),
        "petrochemical": ("flask", "purple"),
        "gas_flare": ("fire", "orange"),
        "lng_terminal": ("ship", "blue"),
        "petroleum_well": ("oil-can", "black"),
    }
    
    def create_dashboard_map(self, fire_df: pd.DataFrame, facilities_gdf: gpd.GeoDataFrame, 
                              selected_types: list = None) -> str:
        """Create the main dashboard map and return as HTML string."""
        
        # Center on India
        m = folium.Map(
            location=[22.5, 78.5],
            zoom_start=5,
            tiles=None,
            prefer_canvas=True
        )
        
        # Add tile layers
        folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri', name='Satellite View'
        ).add_to(m)
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
            attr='CartoDB', name='Dark Mode'
        ).add_to(m)
        
        # Filter by selected types if provided
        if selected_types and 'fire_type' in fire_df.columns:
            display_df = fire_df[fire_df['fire_type'].isin(selected_types)]
        else:
            display_df = fire_df
        
        # --- Fire Detection Layer (Clustered) ---
        fire_cluster = MarkerCluster(name="🔥 Fire Detections", show=True)
        
        for _, row in display_df.iterrows():
            fire_type = row.get('fire_type', 'Other/Unknown')
            color = self.FIRE_TYPE_COLORS.get(fire_type, '#95a5a6')
            
            popup_html = f"""
            <div style='font-family: Arial; min-width: 200px;'>
                <h4 style='color: {color}; margin-bottom: 5px;'>🔥 {fire_type}</h4>
                <hr style='margin: 5px 0;'>
                <b>Coordinates:</b> {row['latitude']:.4f}, {row['longitude']:.4f}<br>
                <b>Brightness:</b> {row.get('brightness', 'N/A')} K<br>
                <b>FRP:</b> {row.get('frp', 'N/A')} MW<br>
                <b>Confidence:</b> {row.get('confidence', 'N/A')}<br>
                <b>Date:</b> {row.get('acq_date', 'N/A')}<br>
                <b>Time:</b> {str(row.get('acq_time', 'N/A')).zfill(4)}<br>
                <b>Day/Night:</b> {row.get('daynight', 'N/A')}<br>
                <b>Nearest Industry:</b> {row.get('nearest_facility_name', 'N/A')}<br>
                <b>Distance:</b> {row.get('distance_to_nearest_industrial', 'N/A'):.1f} km<br>
            </div>
            """
            
            # Scale marker radius by FRP
            frp = row.get('frp', 5)
            radius = min(max(frp / 10, 3), 15)
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{fire_type} | FRP: {frp:.1f} MW"
            ).add_to(fire_cluster)
        
        fire_cluster.add_to(m)
        
        # --- Industrial Facilities Layer ---
        if facilities_gdf is not None and not facilities_gdf.empty:
            facility_group = folium.FeatureGroup(name="🏭 Industrial Facilities", show=True)
            
            for _, row in facilities_gdf.iterrows():
                ftype = row.get('facility_type', 'other')
                icon_name, icon_color = self.FACILITY_ICONS.get(ftype, ('question', 'gray'))
                
                popup_html = f"""
                <div style='font-family: Arial; min-width: 180px;'>
                    <h4 style='margin-bottom: 5px;'>🏭 {row.get('name', 'Unknown')}</h4>
                    <hr style='margin: 5px 0;'>
                    <b>Type:</b> {ftype.replace('_', ' ').title()}<br>
                    <b>Location:</b> {row['latitude']:.4f}, {row['longitude']:.4f}<br>
                </div>
                """
                
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_html, max_width=280),
                    tooltip=f"{row.get('name', 'Facility')} ({ftype.replace('_',' ').title()})",
                    icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa')
                ).add_to(facility_group)
            
            facility_group.add_to(m)
        
        # --- Heatmap Layer ---
        if not display_df.empty and 'frp' in display_df.columns:
            heat_data = []
            for _, row in display_df.iterrows():
                frp_norm = min(row.get('frp', 5) / 100.0, 1.0)
                heat_data.append([row['latitude'], row['longitude'], frp_norm])
            
            HeatMap(
                heat_data, 
                name="🌡️ Thermal Heatmap", 
                min_opacity=0.3, 
                radius=18, 
                blur=15,
                gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'},
                show=False
            ).add_to(m)
        
        # --- Industrial Buffer Zones ---
        if facilities_gdf is not None and not facilities_gdf.empty:
            buffer_group = folium.FeatureGroup(name="⭕ Industrial Buffer (2km)", show=False)
            for _, row in facilities_gdf.iterrows():
                folium.Circle(
                    location=[row['latitude'], row['longitude']],
                    radius=2000,  # 2km in meters
                    color='red',
                    fill=True,
                    fillOpacity=0.05,
                    weight=1,
                    dash_array='5'
                ).add_to(buffer_group)
            buffer_group.add_to(m)
        
        # Add controls
        folium.LayerControl(collapsed=False).add_to(m)
        MiniMap(toggle_display=True).add_to(m)
        
        # Return HTML string
        return m._repr_html_()
    
    def get_fire_statistics(self, fire_df: pd.DataFrame) -> dict:
        """Calculate summary statistics from fire data."""
        stats = {
            "total_fires": len(fire_df),
            "avg_brightness": round(fire_df['brightness'].mean(), 1) if 'brightness' in fire_df.columns else 0,
            "avg_frp": round(fire_df['frp'].mean(), 1) if 'frp' in fire_df.columns else 0,
            "high_confidence": len(fire_df[fire_df['confidence'] == 'high']) if 'confidence' in fire_df.columns else 0,
            "industrial_count": len(fire_df[fire_df.get('is_near_industrial', pd.Series(dtype=int)) == 1]) if 'is_near_industrial' in fire_df.columns else 0,
            "daytime_fires": len(fire_df[fire_df['is_daytime'] == 1]) if 'is_daytime' in fire_df.columns else 0,
            "nighttime_fires": len(fire_df[fire_df['is_daytime'] == 0]) if 'is_daytime' in fire_df.columns else 0,
        }
        
        # Fire type distribution
        if 'fire_type' in fire_df.columns:
            type_counts = fire_df['fire_type'].value_counts().to_dict()
            stats['fire_type_distribution'] = type_counts
        else:
            stats['fire_type_distribution'] = {}
        
        # Facility type distribution of nearest facilities
        if 'nearest_facility_type' in fire_df.columns:
            stats['nearest_facility_distribution'] = fire_df['nearest_facility_type'].value_counts().to_dict()
        
        # Date range
        if 'acq_date' in fire_df.columns:
            stats['date_range'] = {
                'start': str(fire_df['acq_date'].min()),
                'end': str(fire_df['acq_date'].max())
            }
        
        return stats
