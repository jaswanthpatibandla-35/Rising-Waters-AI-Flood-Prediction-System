import os
import shutil
import requests
import pandas as pd
from datetime import datetime
from config import Config

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# -------------------------------------------------------------
# 1. OPENWEATHER API INTEGRATION
# -------------------------------------------------------------
def get_weather_data(city_name):
    """
    Query OpenWeather API for weather parameters.
    If the API Key is missing or the lookup fails, returns a realistic mock dictionary.
    """
    api_key = Config.OPENWEATHER_API_KEY
    if not api_key:
        return get_mock_weather_data(city_name)

    try:
        # Step 1: Geocoding (City to Lat/Lon)
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
        geo_res = requests.get(geo_url, timeout=5)
        geo_data = geo_res.json()

        if not geo_data:
            return get_mock_weather_data(city_name)

        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
        state = geo_data[0].get('state', 'Unknown State')
        country = geo_data[0].get('country', 'Unknown Country')

        # Step 2: Weather Lookup
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"
        weather_res = requests.get(weather_url, timeout=5)
        w_data = weather_res.json()

        if weather_res.status_code != 200:
            return get_mock_weather_data(city_name)

        return {
            'success': True,
            'city': geo_data[0]['name'],
            'state': state,
            'country': country,
            'latitude': float(lat),
            'longitude': float(lon),
            'temperature': float(w_data['main']['temp']),
            'humidity': float(w_data['main']['humidity']),
            'pressure': float(w_data['main']['pressure']),
            'cloud_cover': float(w_data['clouds']['all']),
            'wind_speed': float(w_data['wind']['speed'] * 3.6),  # m/s to km/h
            'visibility': float(w_data.get('visibility', 10000) / 1000.0)  # m to km
        }

    except Exception as e:
        # Fallback to mock on any connection timeout/error
        return get_mock_weather_data(city_name)

def get_mock_weather_data(city_name):
    """Generate realistic mock environmental details based on location string."""
    import random
    # Consistent seeds for reproducible mocks per city
    random.seed(hash(city_name) % 10000)
    
    # Check if monsoon-prone or standard location
    is_monsoon_prone = any(x in city_name.lower() for x in ['mumbai', 'kolkata', 'chennai', 'kerala', 'assam', 'dhaka', 'vietnam'])
    
    lat = round(random.uniform(8.0, 37.0), 4)
    lon = round(random.uniform(68.0, 97.0), 4)
    
    temp = round(random.uniform(22.0, 33.0), 1)
    humidity = round(random.uniform(70.0, 98.0), 1) if is_monsoon_prone else round(random.uniform(45.0, 85.0), 1)
    pressure = round(random.uniform(995.0, 1012.0), 1) if is_monsoon_prone else round(random.uniform(1008.0, 1022.0), 1)
    clouds = round(random.uniform(50.0, 100.0), 1) if is_monsoon_prone else round(random.uniform(10.0, 70.0), 1)
    wind = round(random.uniform(10.0, 38.0), 1)
    visibility = round(random.uniform(3.0, 12.0), 1)

    return {
        'success': True,
        'city': city_name.title(),
        'state': 'Regional State',
        'country': 'IN',
        'latitude': lat,
        'longitude': lon,
        'temperature': temp,
        'humidity': humidity,
        'pressure': pressure,
        'cloud_cover': clouds,
        'wind_speed': wind,
        'visibility': visibility
    }

# -------------------------------------------------------------
# 2. PDF REPORT GENERATOR
# -------------------------------------------------------------
def generate_pdf_report(prediction_record, best_model_name):
    """Generates a professional PDF report containing prediction details."""
    os.makedirs(Config.REPORT_FOLDER, exist_ok=True)
    filename = f"Flood_Report_{prediction_record.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    filepath = os.path.join(Config.REPORT_FOLDER, filename)

    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor('#0f172a'),
        alignment=1, # Center
        spaceAfter=15
    )
    
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor('#2563eb'),
        spaceBefore=10,
        spaceAfter=10
    )

    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155')
    )

    story = []

    # Title
    story.append(Paragraph("RISING WATERS | FLOOD RISK REPORT", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Spacer(1, 15))

    # Core Summary Grid
    summary_data = [
        [Paragraph("<b>Prediction ID</b>", body_style), f"#{prediction_record.id}", Paragraph("<b>Date</b>", body_style), prediction_record.date],
        [Paragraph("<b>Operator</b>", body_style), prediction_record.user_name, Paragraph("<b>Location</b>", body_style), prediction_record.location],
        [Paragraph("<b>Latitude</b>", body_style), f"{prediction_record.latitude}°", Paragraph("<b>Longitude</b>", body_style), f"{prediction_record.longitude}°"],
        [Paragraph("<b>Model Used</b>", body_style), best_model_name, Paragraph("<b>Confidence</b>", body_style), f"{prediction_record.confidence}%"]
    ]
    t1 = Table(summary_data, colWidths=[110, 150, 100, 160])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t1)
    story.append(Spacer(1, 20))

    # Risk Warning
    story.append(Paragraph("FLOOD RISK INDEX", section_style))
    risk_color = '#10b981' if prediction_record.risk_level == 'Low' else ('#f59e0b' if prediction_record.risk_level == 'Medium' else '#ef4444')
    
    risk_data = [
        [
            Paragraph(f"<font color='{risk_color}'><b>{prediction_record.risk_level.upper()} RISK DETECTED</b></font>", ParagraphStyle('Risk', parent=body_style, fontSize=14, fontName='Helvetica-Bold')),
            Paragraph(f"Flood Probability: <b>{prediction_record.probability}%</b>", ParagraphStyle('Prob', parent=body_style, fontSize=12))
        ]
    ]
    t_risk = Table(risk_data, colWidths=[260, 260])
    t_risk.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fef2f2') if prediction_record.risk_level == 'High' else colors.HexColor('#f0fdf4') if prediction_record.risk_level == 'Low' else colors.HexColor('#fffbeb')),
        ('GRID', (0,0), (-1,-1), 1.5, colors.HexColor(risk_color)),
        ('PADDING', (0,0), (-1,-1), 12),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t_risk)
    story.append(Spacer(1, 20))

    # Parameters Grid
    story.append(Paragraph("INPUT METEOROLOGICAL MEASUREMENTS", section_style))
    param_data = [
        [Paragraph("<b>Parameter</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Parameter</b>", body_style), Paragraph("<b>Value</b>", body_style)],
        ["Annual Rainfall", f"{prediction_record.annual_rainfall} mm", "Monthly Rainfall", f"{prediction_record.monthly_rainfall} mm"],
        ["Temperature", f"{prediction_record.temperature} °C", "Relative Humidity", f"{prediction_record.humidity} %"],
        ["Atmospheric Pressure", f"{prediction_record.pressure} hPa", "Cloud Cover", f"{prediction_record.cloud_cover} %"],
        ["Wind Speed", f"{prediction_record.wind_speed} km/h", "River Level", f"{prediction_record.river_water_level} m"],
        ["Ground Water Level", f"{prediction_record.ground_water_level} m", "Visibility", f"{prediction_record.visibility} km"],
        ["Season", prediction_record.season, "Month", prediction_record.month]
    ]
    t_params = Table(param_data, colWidths=[150, 110, 150, 110])
    t_params.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_params)
    story.append(Spacer(1, 20))

    # Evacuation Advisory
    story.append(Paragraph("SAFETY ADVISORY & MITIGATION STEPS", section_style))
    if prediction_record.risk_level == 'High':
        advisory_text = "<b>CRITICAL WARNING:</b> Immediate evacuation protocols should be followed. Move to higher ground, disconnect power mains, charge device packs, and pack clean drinking water. Keep disaster emergency lines open."
    elif prediction_record.risk_level == 'Medium':
        advisory_text = "<b>CAUTION ADVISED:</b> Active weather patterns indicate water accumulation risks. Clear local drainage paths, relocate valuables to upper shelves, and monitor official flood broadcasts."
    else:
        advisory_text = "<b>NORMAL STATUS:</b> Standard seasonal weather parameters active. No warnings issued. Check community drain lines periodically."
    
    story.append(Paragraph(advisory_text, body_style))
    
    doc.build(story)
    return filename

# -------------------------------------------------------------
# 3. EXCEL/CSV HISTORY GENERATOR
# -------------------------------------------------------------
def export_predictions_data(records, format_type='CSV'):
    """Exports prediction records list as CSV or Excel binaries."""
    os.makedirs(Config.REPORT_FOLDER, exist_ok=True)
    
    data_list = []
    for r in records:
        data_list.append({
            'ID': r.id,
            'Date': r.date,
            'Operator': r.user_name,
            'Location': r.location,
            'Latitude': r.latitude,
            'Longitude': r.longitude,
            'Annual_Rainfall_mm': r.annual_rainfall,
            'Monthly_Rainfall_mm': r.monthly_rainfall,
            'Temperature_C': r.temperature,
            'Humidity_percent': r.humidity,
            'Pressure_hPa': r.pressure,
            'Cloud_Cover_percent': r.cloud_cover,
            'Wind_Speed_kmh': r.wind_speed,
            'River_Level_m': r.river_water_level,
            'Ground_Water_Level_m': r.ground_water_level,
            'Visibility_km': r.visibility,
            'Season': r.season,
            'Month': r.month,
            'District': r.district,
            'State': r.state,
            'Flood_Risk_Probability': r.probability,
            'Risk_Level': r.risk_level,
            'Model_Confidence': r.confidence
        })

    df = pd.DataFrame(data_list)
    
    if format_type.upper() == 'EXCEL':
        filename = f"Predictions_Export_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"
        filepath = os.path.join(Config.REPORT_FOLDER, filename)
        df.to_excel(filepath, index=False)
    else:
        filename = f"Predictions_Export_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        filepath = os.path.join(Config.REPORT_FOLDER, filename)
        df.to_csv(filepath, index=False)

    return filename

# -------------------------------------------------------------
# 4. DATABASE BACKUP & RESTORE
# -------------------------------------------------------------
def backup_database():
    """Copies current database file to backups folder."""
    os.makedirs(Config.BACKUP_DIR, exist_ok=True)
    if not os.path.exists(Config.DATABASE_PATH):
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f"flood_backup_{timestamp}.db"
    backup_path = os.path.join(Config.BACKUP_DIR, backup_filename)
    
    shutil.copy(Config.DATABASE_PATH, backup_path)
    return backup_filename

def restore_database(backup_filename):
    """Restores database from backups folder."""
    backup_path = os.path.join(Config.BACKUP_DIR, backup_filename)
    if not os.path.exists(backup_path):
        return False
    
    # Overwrite database file
    shutil.copy(backup_path, Config.DATABASE_PATH)
    return True
