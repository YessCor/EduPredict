import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as plotly_ex
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

st.set_page_config(
    page_title="EduPredict Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
    --primary: #3b82f6;
    --primary-dark: #2563eb;
    --success: #10b981;
    --danger: #ef4444;
    --background: #f0f2f5;
    --card-bg: #ffffff;
    --text-main: #1e293b;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --radius-lg: 16px;
    --radius-md: 12px;
    --shadow: 0 1px 3px rgba(0,0,0,0.1);
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

[data-testid="stAppViewContainer"] {
    background-color: var(--background);
}
[data-testid="stMainBlockContainer"] {
    background-color: var(--background);
    padding-top: 1rem;
}
[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
}

/* Hide Streamlit elements to keep it clean */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr {
    border-color: #334155 !important;
}

/* Sidebar Brand */
.ep-brand {
    padding: 32px 20px;
    text-align: center;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.ep-brand-icon {
    width: 64px;
    height: 64px;
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 16px auto;
    font-size: 32px;
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.3);
}

.ep-brand h2 {
    font-size: 24px;
    font-weight: 800;
    margin: 0;
    color: #ffffff;
    letter-spacing: -0.5px;
}

.ep-brand p {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 4px;
}

/* Modern Radio Navigation */
[data-testid="stSidebar"] .stRadio {
    padding: 20px 12px;
}

/* Hide the radio dot/circle */
[data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"] {
    display: none;
}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border: none !important;
    padding: 12px 16px !important;
    border-radius: 12px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
}

/* Hide the circular indicator */
[data-testid="stSidebar"] .stRadio label [data-testid="stBaseButton-radio"] {
    display: none !important;
}

/* Style the text label */
[data-testid="stSidebar"] .stRadio label div[data-testid="stMarkdownContainer"] p {
    color: #94a3b8 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    margin: 0 !important;
}

/* Hover state */
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255, 255, 255, 0.05) !important;
}

[data-testid="stSidebar"] .stRadio label:hover div[data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
}

/* Active/Checked state */
[data-testid="stSidebar"] .stRadio label[data-selected="true"] {
    background: rgba(59, 130, 246, 0.15) !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
}

[data-testid="stSidebar"] .stRadio label[data-selected="true"] div[data-testid="stMarkdownContainer"] p {
    color: #3b82f6 !important;
    font-weight: 600 !important;
}

/* Custom indicator for active state */
[data-testid="stSidebar"] .stRadio label[data-selected="true"]::before {
    content: '';
    position: absolute;
    left: 0;
    top: 25%;
    bottom: 25%;
    width: 3px;
    background: #3b82f6;
    border-radius: 0 4px 4px 0;
}

.ep-sidebar-footer {
    padding: 24px;
    text-align: center;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: auto;
}

.ep-sidebar-footer p {
    color: #475569;
    font-size: 12px;
}

/* Titles */
.ep-title {
    font-size: 32px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 8px;
    letter-spacing: -0.025em;
}

.ep-sub {
    font-size: 16px;
    color: #64748b;
    margin-bottom: 32px;
}

.ep-section-title {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-main);
    margin: 32px 0 16px 0;
}

/* Metric cards */
.ep-metric {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow);
    transition: all 0.25s ease;
}

.ep-metric:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
}

.ep-metric-accent {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    border-radius: 16px 16px 0 0;
}

.ep-metric-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 10px;
}

.ep-metric-value {
    font-size: 38px;
    font-weight: 700;
    color: var(--text-main);
    line-height: 1;
}

.ep-metric-sub {
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 10px;
}

/* Cards */
.ep-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
    box-shadow: var(--shadow);
    margin-bottom: 20px;
}

.ep-card-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 20px;
}

.ep-insight-card {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border: 1px solid #bae6fd;
    border-radius: var(--radius-md);
    padding: 20px;
}

.ep-insight-title {
    font-size: 13px;
    font-weight: 600;
    color: #0f172a;
    margin: 0 0 12px 0;
}

.ep-insight-item {
    margin-bottom: 12px;
}

.ep-insight-label {
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 2px;
}

.ep-insight-text {
    font-size: 12px;
    color: #475569;
    margin: 0;
}

/* Chart container */
.ep-chart-container {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 16px;
    margin-bottom: 16px;
}

/* Table */
.ep-table-container {
    background: var(--card-bg);
    border-radius: var(--radius-md);
    border: 1px solid var(--border);
    overflow: hidden;
    margin-top: 10px;
}

.ep-table {
    width: 100%;
    border-collapse: collapse;
    text-align: left;
}

.ep-table thead {
    background: #f8fafc;
}

.ep-table th {
    padding: 16px;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 2px solid var(--border);
}

.ep-table td {
    padding: 16px;
    font-size: 14px;
    color: var(--text-main);
    border-bottom: 1px solid #f1f5f9;
}

.ep-table tr:last-child td {
    border-bottom: none;
}

.ep-table tr:hover {
    background: #f8fafc;
}

/* Badges */
.badge {
    padding: 5px 12px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
    display: inline-block;
}

.badge-risk {
    background: #fce7f3;
    color: #be185d;
}

.badge-safe {
    background: #dcfce7;
    color: #15803d;
}

/* Result box */
.ep-result-box {
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin-top: 16px;
    transition: all 0.3s ease;
}

.ep-result-box.danger {
    background: linear-gradient(135deg, #fff1f2, #fce7f3);
    border: 2px solid #fda4af;
}

.ep-result-box.safe {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid #86efac;
}

.ep-result-icon {
    font-size: 48px;
    margin-bottom: 12px;
}

.ep-result-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
}

.ep-result-prob {
    font-size: 15px;
    color: var(--text-muted);
    margin-top: 8px;
}

.ep-progress-bar {
    height: 12px;
    background: #e2e8f0;
    border-radius: 6px;
    overflow: hidden;
    margin-top: 16px;
}

.ep-progress-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
}

/* Streamlit overrides */
.stSelectbox label, .stSlider label {
    color: var(--text-main) !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = {
        "Promedio": [2.5, 3.8, 4.2, 2.8, 3.6, 2.7, 4.5, 3.1, 4.0, 2.4],
        "Fallas": [18, 4, 1, 15, 5, 17, 0, 10, 2, 20],
        "Horas_Estudio": [1, 5, 6, 1, 4, 1, 7, 2, 5, 0],
        "Internet": ["No", "Si", "Si", "No", "Si", "No", "Si", "Si", "Si", "No"],
        "Trabaja": ["Si", "No", "No", "Si", "No", "Si", "No", "Si", "No", "Si"],
        "Deserta": [1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    df_display = df.copy()
    le_internet = LabelEncoder()
    le_trabaja = LabelEncoder()
    df["Internet"] = le_internet.fit_transform(df["Internet"])
    df["Trabaja"] = le_trabaja.fit_transform(df["Trabaja"])
    return df, df_display

df, df_display = load_data()

@st.cache_resource
def train_models(df):
    X = df.drop("Deserta", axis=1)
    y = df["Deserta"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model_rf = RandomForestClassifier(random_state=42)
    model_rf.fit(X_train, y_train)
    return model_rf, X, y, X_test, y_test

model_rf, X, y, X_test, y_test = train_models(df)

with st.sidebar:
    st.markdown("""
        <div class="ep-brand">
            <div class="ep-brand-icon">🎓</div>
            <h2>EduPredict AI</h2>
            <p>Dashboard Académico</p>
        </div>
    """, unsafe_allow_html=True)
    page = st.radio(
        "Navegación",
        ["Inicio", "Análisis Exploratorio", "Simulador de Riesgo", "Métricas del Modelo"],
        label_visibility="collapsed"
    )
    st.markdown("""
        <div class="ep-sidebar-footer">
            <p>EduPredict v2.0</p>
        </div>
    """, unsafe_allow_html=True)

def metric_card(label, value, subtext, color="#3b82f6"):
    st.markdown(f"""
        <div class="ep-metric">
            <div class="ep-metric-accent" style="background: {color}"></div>
            <div class="ep-metric-label">{label}</div>
            <div class="ep-metric-value">{value}</div>
            <div class="ep-metric-sub">{subtext}</div>
        </div>
    """, unsafe_allow_html=True)

if page == "Inicio":
    st.markdown("<div class='ep-title'>Resumen General</div>", unsafe_allow_html=True)
    st.markdown("<div class='ep-sub'>Detección de riesgo de deserción con inteligencia artificial</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Total estudiantes", len(df), "en el dataset actual", "#3b82f6")
    with col2:
        tasa = (df['Deserta'].sum() / len(df)) * 100
        metric_card("Tasa de deserción", f"{tasa:.0f}%", f"{df['Deserta'].sum()} de {len(df)} estudiantes", "#f43f5e")
    with col3:
        metric_card("Variables clave", "5", "analizadas por el modelo AI", "#10b981")
        
    st.markdown("<div class='ep-section-title'>Dataset de Estudiantes</div>", unsafe_allow_html=True)
    
    html_table = """
    <div class="ep-table-container">
    <table class="ep-table">
        <thead>
            <tr>
                <th>Promedio</th>
                <th>Fallas</th>
                <th>Hrs/Sem</th>
                <th>Internet</th>
                <th>Trabaja</th>
                <th>Estado</th>
            </tr>
        </thead>
        <tbody>"""
    for _, row in df_display.iterrows():
        badge = "<span class='badge badge-risk'>Riesgo</span>" if row['Deserta'] else "<span class='badge badge-safe'>Estable</span>"
        html_table += f"""
            <tr>
                <td><strong>{row['Promedio']:.1f}</strong></td>
                <td>{row['Fallas']}</td>
                <td>{row['Horas_Estudio']}</td>
                <td>{row['Internet']}</td>
                <td>{row['Trabaja']}</td>
                <td>{badge}</td>
            </tr>"""
    html_table += "</tbody></table></div>"
    st.markdown(html_table, unsafe_allow_html=True)

elif page == "Análisis Exploratorio":
    st.markdown("<div class='ep-title'>Análisis de Tendencias</div>", unsafe_allow_html=True)
    st.markdown("<div class='ep-sub'>Relaciones críticas entre el rendimiento y la permanencia</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='ep-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ep-card-title'>📈 Promedio vs Deserción</div>", unsafe_allow_html=True)
        fig1 = plotly_ex.scatter(df_display, x="Promedio", y="Deserta", color="Deserta", 
                                 color_discrete_map={0: "#10b981", 1: "#f43f5e"},
                                 labels={"Deserta": "Estado"},)
        fig1.update_layout(
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font_family="Inter, sans-serif",
            font_color="#1e293b",
            margin=dict(l=10, r=10, t=10, b=30),
            height=300,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0)",
                font=dict(size=11)
            ),
            xaxis=dict(
                title="Promedio Académico",
                gridcolor='#f1f5f9',
                linecolor='#e2e8f0'
            ),
            yaxis=dict(
                title="Deserción",
                gridcolor='#f1f5f9',
                linecolor='#e2e8f0',
                tickvals=[0, 1],
                ticktext=['No Deserta', 'Deserta']
            )
        )
        fig1.update_traces(marker=dict(size=16, line=dict(width=2, color='#ffffff'), opacity=0.9))
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("""
            <div style="font-size: 12px; color: #64748b; text-align: center; margin-top: 8px;">
                Estudiantes con promedio bajo (< 3.0) tienen mayor riesgo de deserción
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='ep-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ep-card-title'>📉 Fallas vs Deserción</div>", unsafe_allow_html=True)
        fig2 = plotly_ex.scatter(df_display, x="Fallas", y="Deserta", color="Deserta", 
                                 color_discrete_map={0: "#10b981", 1: "#f43f5e"},
                                 labels={"Deserta": "Estado"})
        fig2.update_layout(
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font_family="Inter, sans-serif",
            font_color="#1e293b",
            margin=dict(l=10, r=10, t=10, b=30),
            height=300,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0)",
                font=dict(size=11)
            ),
            xaxis=dict(
                title="Número de Fallas",
                gridcolor='#f1f5f9',
                linecolor='#e2e8f0'
            ),
            yaxis=dict(
                title="Deserción",
                gridcolor='#f1f5f9',
                linecolor='#e2e8f0',
                tickvals=[0, 1],
                ticktext=['No Deserta', 'Deserta']
            )
        )
        fig2.update_traces(marker=dict(size=16, line=dict(width=2, color='#ffffff'), opacity=0.9))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""
            <div style="font-size: 12px; color: #64748b; text-align: center; margin-top: 8px;">
                Más de 10 fallas incrementan significativamente el riesgo
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ep-card' style='margin-top: 20px;'>", unsafe_allow_html=True)
    st.markdown("<div class='ep-card-title'>📊 Distribución de Horas de Estudio</div>", unsafe_allow_html=True)
    
    col_h1, col_h2 = st.columns([1, 1])
    with col_h1:
        fig3 = plotly_ex.histogram(df_display, x="Horas_Estudio", color="Deserta", barmode="group",
                                    color_discrete_map={0: "#10b981", 1: "#f43f5e"},
                                    nbins=8)
        fig3.update_layout(
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font_family="Inter, sans-serif",
            font_color="#1e293b",
            margin=dict(l=10, r=10, t=10, b=30),
            height=280,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="right",
                x=1,
                font=dict(size=11)
            ),
            xaxis=dict(
                title="Horas de Estudio/Semana",
                gridcolor='#f1f5f9',
                linecolor='#e2e8f0'
            ),
            yaxis=dict(
                title="Cantidad de Estudiantes",
                gridcolor='#f1f5f9',
                linecolor='#e2e8f0'
            ),
            bargap=0.2
        )
        fig3.update_traces(opacity=0.85)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col_h2:
        st.markdown("""
            <div style="padding: 20px; background: linear-gradient(135deg, #f0f9ff, #e0f2fe); border-radius: 12px; height: 100%;">
                <h4 style="color: #0f172a; margin: 0 0 16px 0; font-size: 14px;">📋 Hallazgos Clave</h4>
                <div style="margin-bottom: 12px;">
                    <span style="color: #10b981; font-weight: 600;">✓ Estudiantes estables</span>
                    <p style="color: #475569; font-size: 13px; margin: 4px 0 0 0;">Estudian 4+ horas semanales con bajo número de fallas</p>
                </div>
                <div style="margin-bottom: 12px;">
                    <span style="color: #f43f5e; font-weight: 600;">✗ Estudiantes en riesgo</span>
                    <p style="color: #475569; font-size: 13px; margin: 4px 0 0 0;">Estudian menos de 2 horas y tienen múltiples fallas</p>
                </div>
                <div>
                    <span style="color: #6366f1; font-weight: 600;">💡 Insights</span>
                    <p style="color: #475569; font-size: 13px; margin: 4px 0 0 0;">El tiempo de estudio es el factor más diferenciador</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Simulador de Riesgo":
    st.markdown("<div class='ep-title'>Predicción en Tiempo Real</div>", unsafe_allow_html=True)
    st.markdown("<div class='ep-sub'>Ajuste los parámetros del estudiante para evaluar el riesgo</div>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='ep-card'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            promedio = st.slider("Promedio Académico", 0.0, 5.0, 3.0, 0.1)
            fallas = st.slider("Número de Fallas", 0, 30, 5)
            horas = st.slider("Horas de Estudio/Semana", 0, 20, 4)
        with c2:
            internet = st.selectbox("¿Tiene acceso a Internet?", ["Si", "No"])
            trabaja = st.selectbox("¿Trabaja actualmente?", ["No", "Si"])
            
            input_df = pd.DataFrame({
                "Promedio": [promedio],
                "Fallas": [fallas],
                "Horas_Estudio": [horas],
                "Internet": [1 if internet == "Si" else 0],
                "Trabaja": [1 if trabaja == "Si" else 0]
            })
            prob = model_rf.predict_proba(input_df)[0][1]
            is_risk = prob >= 0.5
            
            box_class = "danger" if is_risk else "safe"
            color = "#ef4444" if is_risk else "#10b981"
            status = "ALTO RIESGO" if is_risk else "RIESGO BAJO"
            icon = "⚠️" if is_risk else "✅"
            
            st.markdown(f"""
                <div class="ep-result-box {box_class}">
                    <div class="ep-result-icon">{icon}</div>
                    <div class="ep-result-title" style="color: {color};">{status}</div>
                    <div class="ep-result-prob">Probabilidad: <strong>{prob*100:.0f}%</strong></div>
                    <div class="ep-progress-bar">
                        <div class="ep-progress-fill" style="width: {prob*100}%; background: {color};"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Métricas del Modelo":
    st.markdown("<div class='ep-title'>Rendimiento de la Inteligencia Artificial</div>", unsafe_allow_html=True)
    st.markdown("<div class='ep-sub'>Evaluación técnica del modelo Random Forest</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        acc = accuracy_score(y_test, model_rf.predict(X_test))
        st.markdown(f"""
            <div class="ep-card" style="text-align: center;">
                <div class="ep-card-title">Accuracy General</div>
                <div style="font-size: 56px; font-weight: 700; color: #10b981; line-height: 1.2;">{acc*100:.0f}%</div>
                <div class="ep-metric-sub">Basado en el conjunto de prueba</div>
            </div>
        """, unsafe_allow_html=True)
        
        cv_scores = cross_val_score(model_rf, X, y, cv=5)
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='ep-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ep-card-title'>Validación Cruzada (5 Folds)</div>", unsafe_allow_html=True)
        st.bar_chart(cv_scores, color="#3b82f6")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='ep-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ep-card-title'>Importancia de Variables</div>", unsafe_allow_html=True)
        importances = pd.Series(model_rf.feature_importances_, index=X.columns).sort_values(ascending=True)
        fig_imp = plotly_ex.bar(importances, orientation='h', 
                                title="<b>Factores que más pesan</b>",
                               color_discrete_sequence=["#3b82f6"])
        fig_imp.update_layout(
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font_family="Inter, sans-serif",
            font_color="#1e293b",
            title_font_size=14,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            height=280
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)