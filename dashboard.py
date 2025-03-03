import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import random
import joblib
import time
import base64
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# โหลดโมเดล Machine Learning และ Label Encoder
# ถ้าไม่มีโมเดลอยู่แล้ว เราจะสร้างโมเดลและเทรนง่ายๆ ขึ้นมาใช้
try:
    loaded_model = joblib.load("machine_failure_model.pkl")
    loaded_le = joblib.load("label_encoder.pkl")
    print("โหลดโมเดลที่มีอยู่แล้วสำเร็จ")
except:
    print("ไม่พบโมเดลที่มีอยู่แล้ว จะสร้างโมเดลใหม่...")
    # สร้างข้อมูลจำลองสำหรับเทรนโมเดล
    def create_synthetic_data(n_samples=1000):
        data = []
        failure_types = ["Normal", "Bearing Failure", "Motor Overheating", 
                         "Misalignment", "Loose Components", "Excessive Load"]
        
        for _ in range(n_samples):
            temperature = random.uniform(50, 130)
            vibration = random.uniform(0.1, 2.5)
            machine_age = random.randint(1, 10)
            humidity = random.randint(25, 75)
            rpm = random.randint(1000, 5500)
            operating_hours = random.randint(1000, 8500)
            
            # กำหนดเงื่อนไขความผิดปกติ (คล้ายกับ rule-based แต่เพิ่มความซับซ้อน)
            if temperature > 105 and vibration > 1.2:
                failure = "Motor Overheating"
            elif vibration > 1.8 and rpm > 4000:
                failure = "Misalignment"
            elif humidity < 30 and vibration > 1.0:
                failure = "Bearing Failure"
            elif machine_age > 8 and operating_hours > 7000:
                failure = "Loose Components"
            elif rpm > 4800 or (rpm > 4200 and temperature > 95):
                failure = "Excessive Load"
            else:
                failure = "Normal"
                
            # เพิ่มความหลากหลายและ noise ให้ข้อมูล
            if random.random() < 0.15:  # สุ่ม 15% ของข้อมูลให้แตกต่างจากกฎ
                failure = random.choice(failure_types)
                
            data.append({
                "Temperature": temperature,
                "Vibration": vibration,
                "Machine_Age": machine_age,
                "Humidity": humidity,
                "RPM": rpm,
                "Operating_Hours": operating_hours,
                "Failure_Type": failure
            })
        
        return pd.DataFrame(data)
    
    # สร้างข้อมูลสำหรับเทรนโมเดล
    train_data = create_synthetic_data(2000)
    
    # แปลงคลาสให้เป็นตัวเลขด้วย LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(train_data["Failure_Type"])
    X = train_data.drop("Failure_Type", axis=1)
    
    # สร้างและเทรนโมเดล RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # บันทึกโมเดลและ LabelEncoder
    joblib.dump(model, "machine_failure_model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    
    # กำหนดตัวแปรให้กับโค้ดที่เหลือ
    loaded_model = model
    loaded_le = le
    print("สร้างและบันทึกโมเดลใหม่เรียบร้อย")

# ฟังก์ชันจำลองค่าจากเซ็นเซอร์
def generate_sensor_data():
    return {
        "Temperature": round(random.uniform(50, 120), 2),  # °C
        "Vibration": round(random.uniform(0.1, 2.0), 2),  # G-force
        "Machine_Age": random.randint(1, 10),             # Years
        "Humidity": random.randint(30, 70),               # %
        "RPM": random.randint(1000, 5000),               # รอบต่อนาที
        "Operating_Hours": random.randint(1000, 8000)    # ชั่วโมง
    }

# ตั้งค่าเริ่มต้นของกราฟ
sensor_data_history = pd.DataFrame(columns=["Time", "Temperature", "Vibration", "RPM"])

# ค่าเริ่มต้นสำหรับการแสดงข้อมูลเซ็นเซอร์
last_sensor_data = {
    "Temperature": 0,
    "Vibration": 0,
    "Machine_Age": 0,
    "Humidity": 0,
    "RPM": 0,
    "Operating_Hours": 0
}

# สร้าง DataFrame สำหรับเก็บประวัติการตรวจพบความผิดปกติ
abnormality_history = pd.DataFrame(columns=["Timestamp", "Abnormality_Type", "Temperature", "Vibration", "RPM", "Humidity"])

# กำหนด External Stylesheets และ custom CSS
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# สร้างโฟลเดอร์ assets ถ้ายังไม่มี
import os
if not os.path.exists("assets"):
    os.makedirs("assets")

# Custom CSS (เหมือนเดิม ไม่มีการเปลี่ยนแปลง จึงละไว้)
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>AI Predictive Maintenance Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Roboto', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                color: #ffffff;
                background-color: #121212; /* fallback if image fails */
                background-image: url('/assets/background.jpg');
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
                min-height: 100vh;
            }
            
            /* Overlay to make background darker for better readability */
            body::before {
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.7); /* Dark overlay */
                z-index: -1;
            }
            
            .header {
                background: rgba(0, 0, 0, 0.7);
                color: #00bcd4;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                margin-bottom: 20px;
                border-bottom: 2px solid #00bcd4;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
            }
            
            .dashboard-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 20px 20px 20px;
            }
            
            .card {
                background-color: rgba(30, 30, 30, 0.8);
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
                padding: 20px;
                margin-bottom: 20px;
                border: 1px solid #333333;
                backdrop-filter: blur(5px);
                -webkit-backdrop-filter: blur(5px);
            }
            
            .card-title {
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 15px;
                color: #00bcd4;
                border-bottom: 1px solid #444444;
                padding-bottom: 8px;
            }
            
            .sensor-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .sensor-value {
                display: flex;
                align-items: center;
                justify-content: space-between;
                background-color: rgba(40, 40, 40, 0.9);
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
                border-left: 3px solid #00bcd4;
                transition: all 0.3s ease;
            }
            
            .sensor-value:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 188, 212, 0.4);
            }
            
            .sensor-label {
                font-weight: 500;
                color: #bbbbbb;
            }
            
            .sensor-reading {
                font-size: 22px;
                font-weight: 600;
                color: #ffffff;
                text-shadow: 0 0 5px rgba(0, 188, 212, 0.5);
            }
            
            .sensor-icon {
                font-size: 24px;
                margin-right: 10px;
                color: #00bcd4;
            }
            
            .normal {
                background-color: rgba(27, 94, 32, 0.8);
                color: #e8f5e9;
                border: 1px solid #2e7d32;
            }
            
            .warning {
                background-color: rgba(230, 81, 0, 0.8);
                color: #fff8e1;
                border: 1px solid #ff8f00;
            }
            
            .danger {
                background-color: rgba(183, 28, 28, 0.8);
                color: #ffebee;
                border: 1px solid #c62828;
            }
            
            .status-indicator {
                font-size: 24px;
                font-weight: 600;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
                backdrop-filter: blur(5px);
                -webkit-backdrop-filter: blur(5px);
            }
            
            .status-icon {
                font-size: 28px;
                margin-right: 10px;
            }
            
            .info-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
            }
            
            .info-item {
                display: flex;
                flex-direction: column;
                padding: 10px;
                background-color: rgba(40, 40, 40, 0.9);
                border-radius: 5px;
            }
            
            @media (max-width: 600px) {
                .info-grid {
                    grid-template-columns: 1fr;
                }
            }
            
            /* สีเรืองแสงสำหรับข้อมูล */
            .glow-text {
                text-shadow: 0 0 5px #00bcd4, 0 0 10px rgba(0, 188, 212, 0.5);
            }
            
            /* เพิ่มเอฟเฟ็กต์เคลื่อนไหว */
            @keyframes pulse {
                0% {
                    box-shadow: 0 0 0 0 rgba(0, 188, 212, 0.4);
                }
                70% {
                    box-shadow: 0 0 0 10px rgba(0, 188, 212, 0);
                }
                100% {
                    box-shadow: 0 0 0 0 rgba(0, 188, 212, 0);
                }
            }
            
            .pulse {
                animation: pulse 2s infinite;
            }
            
            /* สไตล์สำหรับกราฟ */
            .plot-container {
                border-radius: 8px;
                overflow: hidden;
            }
            
            /* สไตล์สำหรับตารางประวัติ */
            .history-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }
            
            .history-table th {
                background-color: rgba(0, 188, 212, 0.2);
                color: #00bcd4;
                padding: 12px 10px;
                text-align: left;
                font-weight: 500;
                border-bottom: 2px solid #444;
            }
            
            .history-table td {
                padding: 12px 10px;
                border-bottom: 1px solid #444;
                color: #e0e0e0;
            }
            
            .history-table tr:hover {
                background-color: rgba(0, 188, 212, 0.1);
            }
            
            .abnormality-tag {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: 500;
                background-color: rgba(183, 28, 28, 0.8);
                color: #fff;
                margin-right: 5px;
                margin-bottom: 5px;
            }
            
            .abnormality-container {
                display: flex;
                flex-wrap: wrap;
            }
            
            .history-empty {
                text-align: center;
                padding: 20px;
                color: #888;
                font-style: italic;
            }
            
            .history-container {
                max-height: 300px;
                overflow-y: auto;
                scrollbar-width: thin;
                scrollbar-color: #00bcd4 #2a2a2a;
            }
            
            .history-container::-webkit-scrollbar {
                width: 8px;
            }
            
            .history-container::-webkit-scrollbar-track {
                background: #2a2a2a;
                border-radius: 4px;
            }
            
            .history-container::-webkit-scrollbar-thumb {
                background-color: #00bcd4;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# สร้างไฟล์ HTML สำหรับแจ้งเตือนเพื่อวางรูปภาพพื้นหลัง (เหมือนเดิม)
background_notice = """
<html>
<body>
    <h2>คำแนะนำการใช้รูปภาพพื้นหลัง</h2>
    <p>กรุณาจัดเตรียมรูปภาพพื้นหลังโดยทำตามขั้นตอนดังนี้:</p>
    <ol>
        <li>สร้างโฟลเดอร์ชื่อ "assets" ในโฟลเดอร์เดียวกับไฟล์ Python นี้</li>
        <li>วางรูปภาพลงในโฟลเดอร์ assets และตั้งชื่อเป็น "background.jpg"</li>
    </ol>
    <p>หากต้องการใช้ชื่อไฟล์อื่น ให้แก้ไขที่ <code>background-image: url('/assets/background.jpg');</code> ในโค้ด CSS</p>
</body>
</html>
"""
# ส่วนประกอบ UI ของแอป (เหมือนเดิม)
app.layout = html.Div([
    # Header
    html.Div([
        html.H1([html.I(className="fas fa-cogs", style={"marginRight": "15px"}), "AI Predictive Maintenance Dashboard"], 
                style={"margin": "0", "textAlign": "center"})
    ], className="header"),
    
    # Main container
    html.Div([
        # Status indicator
        html.Div(id="prediction-output", className="status-indicator normal pulse"),
        
        # Sensor readings
        html.Div([
            html.Div([
                html.Div("Real-Time Sensor Readings", className="card-title")
            ]),
            html.Div(id="sensor-readings", className="sensor-grid")
        ], className="card"),
        
        # Graphs
        html.Div([
            html.Div("Sensor Data Trends", className="card-title"),
            dcc.Graph(id="sensor-graph", className="plot-container")
        ], className="card"),
        
        # ประวัติความผิดปกติ
        html.Div([
            html.Div("ประวัติการตรวจพบความผิดปกติ", className="card-title"),
            html.Div(id="abnormality-history", className="history-container")
        ], className="card"),
        
        # รายละเอียดเพิ่มเติม
        html.Div([
            html.Div("Machine Information", className="card-title"),
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("Machine ID:", className="sensor-label"),
                        html.Div("M-7842", className="glow-text", style={"fontWeight": "600", "color": "#ffffff"})
                    ]),
                    html.Div([
                        html.Div("Location:", className="sensor-label"),
                        html.Div("Factory Line B - Station 3", style={"fontWeight": "600", "color": "#ffffff"})
                    ])
                ], className="info-item"),
                html.Div([
                    html.Div([
                        html.Div("Last Maintenance:", className="sensor-label"),
                        html.Div("2024-12-15", style={"fontWeight": "600", "color": "#ffffff"})
                    ]),
                    html.Div([
                        html.Div("Next Scheduled:", className="sensor-label"),
                        html.Div("2025-04-15", className="glow-text", style={"fontWeight": "600", "color": "#ffffff"})
                    ])
                ], className="info-item")
            ], className="info-grid")
        ], className="card"),
        
        # อัปเดตข้อมูลทุก 1 วินาที
        dcc.Interval(id="interval-update", interval=5000, n_intervals=0),
        
        # เพิ่มข้อมูลแสดงสถานะของโมเดล
        html.Div([
            html.Div("Model Information", className="card-title"),
            html.Div(id="model-info", className="info-item")
        ], className="card")
    ], className="dashboard-container")
])

#ใช้โมเดล Machine Learning จริงในการทำนาย
def predict_with_ml_model(sensor_data):
    # แปลงข้อมูลเซ็นเซอร์เป็น DataFrame
    input_df = pd.DataFrame([sensor_data])
    
    # ทำนายด้วยโมเดล
    prediction_idx = loaded_model.predict(input_df)[0]
    
    # แปลงกลับเป็นชื่อประเภทความผิดปกติ
    prediction = loaded_le.inverse_transform([prediction_idx])[0]
    
    # คำนวณความน่าจะเป็นสำหรับแต่ละคลาส
    probabilities = loaded_model.predict_proba(input_df)[0]
    
    # จัดลำดับความน่าจะเป็นจากมากไปน้อย
    sorted_indices = np.argsort(probabilities)[::-1]  # เรียงจากมากไปน้อย
    
    # สร้างรายการความผิดปกติที่มีความน่าจะเป็นสูงกว่าเกณฑ์
    threshold = 0.15  # ความน่าจะเป็นขั้นต่ำที่จะรายงาน
    abnormalities = []
    
    for idx in sorted_indices:
        prob = probabilities[idx]
        if prob >= threshold:
            abnormality = loaded_le.inverse_transform([idx])[0]
            abnormalities.append(abnormality)
    
    # ถ้ามี Normal และความผิดปกติอื่นๆ ให้ตัด Normal ออก
    if "Normal" in abnormalities and len(abnormalities) > 1:
        abnormalities.remove("Normal")
    
    # ถ้าไม่มีความผิดปกติที่เกินเกณฑ์ ให้เป็น Normal
    if not abnormalities:
        abnormalities = ["Normal"]
    
    return abnormalities, probabilities

# ฟังก์ชันอัปเดตข้อมูลแบบเรียลไทม์ (ปรับปรุงให้ใช้โมเดล ML)
@app.callback(
    [Output("sensor-graph", "figure"),
     Output("prediction-output", "children"),
     Output("prediction-output", "className"),
     Output("sensor-readings", "children"),
     Output("abnormality-history", "children"),
     Output("model-info", "children")],
    [Input("interval-update", "n_intervals")]
)
def update_dashboard(n):
    global sensor_data_history, last_sensor_data, abnormality_history
    
    # รับค่าจากเซ็นเซอร์ (จำลอง)
    new_data = generate_sensor_data()
    last_sensor_data = new_data  # เก็บค่าล่าสุดไว้
    timestamp = time.strftime("%H:%M:%S")  # เวลา
    full_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # วันที่และเวลาเต็มรูปแบบ

    # ทำนายด้วยโมเดล Machine Learning
    abnormalities, probabilities = predict_with_ml_model(new_data)

    # เพิ่มข้อมูลใหม่ลงใน DataFrame
    new_entry = pd.DataFrame({"Time": [timestamp], 
                              "Temperature": [new_data["Temperature"]],
                              "Vibration": [new_data["Vibration"]],
                              "RPM": [new_data["RPM"]]})
    
    # แก้ไขเพื่อหลีกเลี่ยง FutureWarning
    if sensor_data_history.empty:
        sensor_data_history = new_entry
    else:
        sensor_data_history = pd.concat([sensor_data_history, new_entry], ignore_index=True)
        
    # จำกัดขนาดให้เก็บแค่ 15 จุดล่าสุด
    if len(sensor_data_history) > 15:
        sensor_data_history = sensor_data_history.tail(15)

    # บันทึกประวัติเมื่อตรวจพบความผิดปกติ
    if abnormalities[0] != "Normal":
        # บันทึกข้อมูลความผิดปกติแต่ละประเภท
        for abnormality in abnormalities:
            abnormality_entry = pd.DataFrame({
                "Timestamp": [full_timestamp],
                "Abnormality_Type": [abnormality],
                "Temperature": [new_data["Temperature"]],
                "Vibration": [new_data["Vibration"]],
                "RPM": [new_data["RPM"]],
                "Humidity": [new_data["Humidity"]]
            })
            
            if abnormality_history.empty:
                abnormality_history = abnormality_entry
            else:
                abnormality_history = pd.concat([abnormality_history, abnormality_entry], ignore_index=True)
        
        # จำกัดประวัติให้แสดงแค่ 20 รายการล่าสุด
        if len(abnormality_history) > 10:
            abnormality_history = abnormality_history.tail(10)

    # สร้างกราฟ (เหมือนเดิม)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sensor_data_history["Time"], y=sensor_data_history["Temperature"], 
                             mode="lines+markers", name="Temperature (°C)", line=dict(color="#ff5722", width=3)))
    fig.add_trace(go.Scatter(x=sensor_data_history["Time"], y=sensor_data_history["Vibration"], 
                             mode="lines+markers", name="Vibration (G-force)", line=dict(color="#00bcd4", width=3)))
    fig.add_trace(go.Scatter(x=sensor_data_history["Time"], y=sensor_data_history["RPM"], 
                             mode="lines+markers", name="RPM", line=dict(color="#76ff03", width=3)))

    fig.update_layout(
        title=None,
        xaxis_title="Time",
        yaxis_title="Value",
        plot_bgcolor="rgba(30, 30, 30, 0.8)",
        paper_bgcolor="rgba(30, 30, 30, 0)",
        font=dict(family="Roboto, sans-serif", color="#e0e0e0"),
        legend=dict(orientation="h", y=1.1, font=dict(color="#e0e0e0"), bgcolor="rgba(30, 30, 30, 0.7)"),
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)", tickfont=dict(color="#e0e0e0")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)", tickfont=dict(color="#e0e0e0")),
        hovermode="x unified"
    )

    # กำหนดสถานะและไอคอน
    status_class = "status-indicator normal pulse" if abnormalities[0] == "Normal" else "status-indicator danger pulse"
    status_icon = html.I(className="fas fa-check-circle status-icon") if abnormalities[0] == "Normal" else html.I(className="fas fa-exclamation-triangle status-icon")
    
    # แสดงผลการทำนาย
    if abnormalities[0] == "Normal":
        prediction_text = [status_icon, "ระบบทำงานปกติ"]
    else:
        # สร้าง tags สำหรับแต่ละความผิดปกติ
        abnormality_tags = html.Div([
            html.Span(abnormality, className="abnormality-tag") for abnormality in abnormalities
        ], className="abnormality-container")
        
        prediction_text = [
            status_icon,
            html.Div([
                html.Div("ตรวจพบความผิดปกติ:", style={"marginBottom": "8px"}),
                abnormality_tags
            ])
        ]
    
    # กำหนดเงื่อนไขสำหรับตรวจสอบค่าผิดปกติของเซ็นเซอร์แต่ละตัว
    is_temp_abnormal = new_data["Temperature"] > 100  # อุณหภูมิสูงเกินไป
    is_vibration_abnormal = new_data["Vibration"] > 1.5  # ความสั่นสะเทือนสูงเกินไป
    is_rpm_abnormal = new_data["RPM"] > 4500  # รอบต่อนาทีสูงเกินไป
    is_humidity_abnormal = new_data["Humidity"] < 35  # ความชื้นต่ำเกินไป
    is_machine_age_abnormal = new_data["Machine_Age"] > 8 and new_data["Operating_Hours"] > 7000  # อายุเครื่องและชั่วโมงทำงานสูง
    is_operating_hours_abnormal = new_data["Operating_Hours"] > 7000  # ชั่วโมงทำงานสูงเกินไป
    
    # สร้าง UI สำหรับแสดงค่าเซ็นเซอร์พร้อมสีแดงสำหรับค่าผิดปกติ (เหมือนเดิม)
    sensor_readings = [
        html.Div([
            html.Div([html.I(className="fas fa-thermometer-half sensor-icon"), "อุณหภูมิ"], className="sensor-label"),
            html.Div(f"{new_data['Temperature']} °C", className="sensor-reading")
        ], className="sensor-value danger" if is_temp_abnormal else "sensor-value"),
        
        html.Div([
            html.Div([html.I(className="fas fa-vibration sensor-icon"), "แรงสั่นสะเทือน"], className="sensor-label"),
            html.Div(f"{new_data['Vibration']} G", className="sensor-reading")
        ], className="sensor-value danger" if is_vibration_abnormal else "sensor-value"),
        
        html.Div([
            html.Div([html.I(className="fas fa-tachometer-alt sensor-icon"), "RPM"], className="sensor-label"),
            html.Div(f"{new_data['RPM']}", className="sensor-reading")
        ], className="sensor-value danger" if is_rpm_abnormal else "sensor-value"),
        
        html.Div([
            html.Div([html.I(className="fas fa-tint sensor-icon"), "ความชื้น"], className="sensor-label"),
            html.Div(f"{new_data['Humidity']}%", className="sensor-reading")
        ], className="sensor-value danger" if is_humidity_abnormal else "sensor-value"),
        
        html.Div([
            html.Div([html.I(className="fas fa-calendar-alt sensor-icon"), "อายุเครื่อง"], className="sensor-label"),
            html.Div(f"{new_data['Machine_Age']} ปี", className="sensor-reading")
        ], className="sensor-value danger" if is_machine_age_abnormal else "sensor-value"),
        
        html.Div([
            html.Div([html.I(className="fas fa-clock sensor-icon"), "ชั่วโมงทำงาน"], className="sensor-label"),
            html.Div(f"{new_data['Operating_Hours']} ชม.", className="sensor-reading")
        ], className="sensor-value danger" if is_operating_hours_abnormal else "sensor-value"),
    ]
    
    # สร้างตารางแสดงประวัติความผิดปกติ (เหมือนเดิม)
    if abnormality_history.empty:
        abnormality_table = html.Div("ยังไม่มีประวัติการตรวจพบความผิดปกติ", className="history-empty")
    else:
        # เรียงข้อมูลจากใหม่ไปเก่า
        sorted_history = abnormality_history.sort_values(by="Timestamp", ascending=False)
        
        # สร้างตาราง
        abnormality_table = html.Table([
            html.Thead(
                html.Tr([
                    html.Th("เวลา"),
                    html.Th("ประเภทความผิดปกติ"),
                    html.Th("อุณหภูมิ (°C)"),
                    html.Th("แรงสั่นสะเทือน (G)"),
                    html.Th("RPM"),
                    html.Th("ความชื้น (%)")
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(row["Timestamp"]),
                    html.Td(html.Span(row["Abnormality_Type"], className="abnormality-tag")),
                    html.Td(f"{row['Temperature']:.2f}"),
                    html.Td(f"{row['Vibration']:.2f}"),
                    html.Td(f"{row['RPM']}"),
                    html.Td(f"{row['Humidity']}"),
                ]) for _, row in sorted_history.iterrows()
            ])
        ], className="history-table")
    
    # เพิ่มข้อมูลแสดงโมเดล
    # สร้างข้อมูลความน่าจะเป็นของแต่ละประเภทความผิดปกติเพื่อแสดงผล
    class_probabilities = []
    for i, prob in enumerate(probabilities):
        class_name = loaded_le.inverse_transform([i])[0]
        if prob >= 0.05:  # แสดงเฉพาะคลาสที่มีความน่าจะเป็นมากกว่า 5%
            class_probabilities.append(html.Div([
                html.Span(f"{class_name}: ", style={"fontWeight": "500", "color": "#bbbbbb"}),
                html.Span(f"{prob*100:.1f}%", style={"fontWeight": "600", "color": "#ffffff"})
            ], style={"marginBottom": "5px"}))
    
    model_info = html.Div([
        html.Div([
            html.Div("โมเดลที่ใช้:", className="sensor-label"),
            html.Div("RandomForest Classifier", className="glow-text", style={"fontWeight": "600", "color": "#ffffff", "marginBottom": "10px"})
        ]),
        html.Div([
            html.Div("ความน่าจะเป็นของแต่ละประเภท:", className="sensor-label", style={"marginBottom": "8px"}),
            html.Div(class_probabilities)
        ])
    ])

    return fig, prediction_text, status_class, sensor_readings, abnormality_table, model_info

# รันแอป
if __name__ == "__main__":
    app.run_server(debug=True)