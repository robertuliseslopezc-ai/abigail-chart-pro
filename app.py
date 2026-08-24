import os
import json
import warnings
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import joblib

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Configuración de página ultra-compacta
st.set_page_config(page_title="Abigail Chart Pro", layout="centered", initial_sidebar_state="collapsed")

# ============================================================
# ESTADO INICIAL
# ============================================================

if "valores" not in st.session_state:
    st.session_state.valores = []

if "aciertos" not in st.session_state:
    st.session_state.aciertos = 0

if "fallos" not in st.session_state:
    st.session_state.fallos = 0

if "hilo" not in st.session_state:
    st.session_state.hilo = []

if "ultima_senal" not in st.session_state:
    st.session_state.ultima_senal = None

if "resultado" not in st.session_state:
    st.session_state.resultado = None

if "model_ready" not in st.session_state:
    st.session_state.model_ready = False

if "model_version" not in st.session_state:
    st.session_state.model_version = 0

if "model_last_train_size" not in st.session_state:
    st.session_state.model_last_train_size = 0

if "modelo_ai" not in st.session_state:
    st.session_state.modelo_ai = None

if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = None

if "historial" not in st.session_state:
    st.session_state.historial = pd.DataFrame(
        columns=[
            "rango",
            "color",
            "conteo_hi_lo",
            "fuerza",
            "senal",
            "resultado"
        ]
    )

# ============================================================
# ESTILOS CSS CUSTOM
# ============================================================

st.markdown("""
    <style>
    header, footer {visibility: hidden;}
    
    .block-container {
        padding-top: 0.2rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 520px !important;
    }
    
    .stApp {
        background-color: #050505;
    }
    
    div.stButton > button {
        padding: 4px 2px !important;
        font-size: 11px !important;
        font-weight: bold !important;
        border-radius: 6px !important;
        background-color: #161b22 !important;
        color: #58a6ff !important;
        border: 1px solid #30363d !important;
        margin: 0px !important;
        height: 36px !important;
    }
    
    div.stButton > button:hover {
        border-color: #00ff66 !important;
        color: #ffffff !important;
        background-color: #1f242d !important;
        box-shadow: 0px 0px 10px rgba(0, 255, 102, 0.5);
    }

    [data-testid="column"] {
        padding: 1px !important;
    }

    .title-abigail {
        text-align: center;
        font-family: 'Trebuchet MS', 'Impact', sans-serif;
        font-size: 20px;
        font-weight: 900;
        letter-spacing: 1.5px;
        background: linear-gradient(90deg, #00ff66, #00e5ff, #ff007f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 5px;
        margin-bottom: 2px;
        text-transform: uppercase;
        text-shadow: 0px 0px 12px rgba(0, 255, 102, 0.3);
    }

    .rondas-badge {
        text-align: center;
        background-color: #11161d;
        border: 1px solid #238636;
        border-radius: 6px;
        color: #00ff66;
        font-size: 11px;
        font-weight: bold;
        padding: 3px 0px;
        margin-top: 4px;
        margin-bottom: 4px;
        letter-spacing: 1px;
        box-shadow: inset 0px 0px 6px rgba(0, 255, 102, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

historial = st.session_state.historial

# ============================================================
# TABLA OFICIAL DE FUERZA
# ============================================================

rangos = {
    "1.00 / 1.09": ("red", -5),
    "1.10 / 1.29": ("red", -4),
    "1.30 / 1.49": ("red", -3),
    "1.50 / 1.79": ("white", -2),
    "1.80 / 1.99": ("white", -1),
    "2.00 / 3.99": ("green", 1),
    "4.00 / 5.99": ("green", 2),
    "6.00 / 7.99": ("yellow", 3),
    "8.00 / 9.99": ("yellow", 4),
    "10+": ("#ff00ff", 5)
}

# ============================================================
# FILTRO ANTI-TRAMPA BAJISTA (LÓGICA INTERNA INVISIBLE)
# ============================================================

def evaluar_filtro_anti_trampa(historial):
    if len(historial) < 5:
        return True

    fuerzas = pd.to_numeric(historial["fuerza"], errors="coerce").dropna().tolist()
    if len(fuerzas) < 5:
        return True

    ventana_10 = fuerzas[-10:]
    suma_fuerza_10 = sum(ventana_10)

    # Entorno Alcista/Positivo: Disparo normal
    if suma_fuerza_10 >= 0:
        return True

    # Entorno Bajista: Aplicar reglas estrictas
    fuerza_actual = fuerzas[-1]
    fuerza_previa = fuerzas[-2] if len(fuerzas) >= 2 else 0

    # Bloqueo de trampa tras caída aplastante
    if fuerza_previa <= -4 and fuerza_actual < 3:
        return False

    # Exigir continuidad de 2 verdes o verde fuerte (>=3)
    consecutivos_verdes = 0
    for f in reversed(fuerzas):
        if f > 0:
            consecutivos_verdes += 1
        else:
            break

    if consecutivos_verdes < 2 and fuerza_actual < 3:
        return False

    return True

# ============================================================
# FUNCIONES DE LECTURA
# ============================================================

def lectura_fuerza(fuerzas):
    if not fuerzas:
        return None, 0
    actual = fuerzas[-1]
    if actual > 0:
        tendencia = "alcista"
    elif actual < 0:
        tendencia = "bajista"
    else:
        tendencia = "neutral"
    intensidad = abs(actual)
    return tendencia, intensidad

def lectura_continuidad(fuerzas):
    if not fuerzas:
        return False, 0, "neutral"
    actual = fuerzas[-1]
    if actual > 0:
        direccion = "alcista"
    elif actual < 0:
        direccion = "bajista"
    else:
        direccion = "neutral"

    consecutivos = 1
    for i in range(len(fuerzas) - 2, -1, -1):
        anterior = fuerzas[i]
        if actual > 0 and anterior > 0:
            consecutivos += 1
        elif actual < 0 and anterior < 0:
            consecutivos += 1
        else:
            break
    continuidad = consecutivos >= 3
    return continuidad, consecutivos, direccion

def lectura_impulso(fuerzas):
    if not fuerzas:
        return False, "neutral"
    actual = fuerzas[-1]
    if actual >= 3:
        return True, "alcista"
    if actual <= -3:
        return True, "bajista"
    return False, "neutral"

def lectura_retroceso(fuerzas):
    if len(fuerzas) < 2:
        return False, "ninguno"
    actual = fuerzas[-1]
    anteriores = fuerzas[:-1]
    max_anterior = max(anteriores)
    min_anterior = min(anteriores)
    if actual < 0 and max_anterior > 0:
        return True, "bajista"
    if actual > 0 and min_anterior < 0:
        return True, "alcista"
    return False, "ninguno"

# ============================================================
# FEATURES PARA IA
# ============================================================

def extraer_limite(rango):
    try:
        if rango == "10+":
            return 10.0
        return float(str(rango).split("/")[0].strip())
    except:
        return np.nan

def construir_features(historial):
    if historial is None or len(historial) < 5:
        return None

    df = historial.copy()
    df["fuerza"] = pd.to_numeric(df["fuerza"], errors="coerce")
    df["conteo_hi_lo"] = pd.to_numeric(df["conteo_hi_lo"], errors="coerce")
    df = df.dropna(subset=["fuerza", "conteo_hi_lo", "rango"]).reset_index(drop=True)

    if len(df) < 5:
        return None

    df["limite"] = df["rango"].apply(extraer_limite)
    df["movimiento_real"] = df["limite"].apply(lambda x: 1 if pd.notna(x) and x >= 2.0 else 0)

    df["fuerza_prev"] = df["fuerza"].shift(1)
    df["fuerza_prev2"] = df["fuerza"].shift(2)
    df["conteo_prev"] = df["conteo_hi_lo"].shift(1)
    df["conteo_prev2"] = df["conteo_hi_lo"].shift(2)
    df["delta_fuerza"] = df["fuerza"] - df["fuerza_prev"]
    df["media_fuerza_3"] = df["fuerza"].rolling(3).mean()
    df["std_fuerza_5"] = df["fuerza"].rolling(5).std()
    df["suma_3"] = df["fuerza"].rolling(3).sum()
    df["suma_5"] = df["fuerza"].rolling(5).sum()

    df["color_red"] = (df["color"] == "red").astype(int)
    df["color_white"] = (df["color"] == "white").astype(int)
    df["color_green"] = (df["color"] == "green").astype(int)
    df["color_yellow"] = (df["color"] == "yellow").astype(int)

    df["abs_fuerza"] = df["fuerza"].abs()
    df["signo_fuerza"] = np.sign(df["fuerza"])
    df["cambio_signo"] = (df["signo_fuerza"] != df["signo_fuerza"].shift(1)).astype(int)

    feature_cols = [
        "fuerza", "fuerza_prev", "fuerza_prev2", "conteo_hi_lo", "conteo_prev",
        "conteo_prev2", "delta_fuerza", "media_fuerza_3", "std_fuerza_5",
        "suma_3", "suma_5", "color_red", "color_white", "color_green",
        "color_yellow", "abs_fuerza", "signo_fuerza", "cambio_signo"
    ]

    X = df[feature_cols].copy()
    X = X.fillna(0)
    y = df["movimiento_real"].copy()

    return X, y, feature_cols, df

# ============================================================
# MOTOR ANALÍTICO BASE
# ============================================================

def motor_analitico(historial):
    if len(historial) < 5:
        return None

    fuerzas = pd.to_numeric(historial["fuerza"], errors="coerce").dropna().tolist()

    if len(fuerzas) < 5:
        return None

    tendencia_fuerza, intensidad = lectura_fuerza(fuerzas)
    continuidad, consecutivos, direccion_continuidad = lectura_continuidad(fuerzas)
    impulso, direccion_impulso = lectura_impulso(fuerzas)
    retroceso, direccion_retroceso = lectura_retroceso(fuerzas)

    condiciones = 0

    if tendencia_fuerza == "alcista" and intensidad >= 1:
        condiciones += 1

    if continuidad and direccion_continuidad == "alcista":
        condiciones += 1

    if impulso and direccion_impulso == "alcista":
        condiciones += 1

    if not retroceso:
        condiciones += 1

    if condiciones >= 3:
        return "analisis_alineado"

    return None

# ============================================================
# IA INTERNA
# ============================================================

def entrenar_modelo_si_aplica(historial):
    try:
        res = construir_features(historial)
        if res is None:
            st.session_state.model_ready = False
            return None
            
        X, y, feature_cols, df = res
        if len(X) < 20 or len(set(y.tolist())) < 2:
            st.session_state.model_ready = False
            return None

        if st.session_state.modelo_ai is None or len(df) > st.session_state.model_last_train_size:
            modelo = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear"
                ))
            ])

            modelo.fit(X, y)

            st.session_state.modelo_ai = modelo
            st.session_state.feature_columns = feature_cols
            st.session_state.model_ready = True
            st.session_state.model_last_train_size = len(df)
            st.session_state.model_version += 1

        return st.session_state.modelo_ai
    except:
        st.session_state.model_ready = False
        return None

def prediccion_ai(historial):
    modelo = entrenar_modelo_si_aplica(historial)
    if modelo is None:
        return None, None

    res = construir_features(historial)
    if res is None:
        return None, None
        
    X, y, feature_cols, df = res
    if X is None or len(X) == 0:
        return None, None

    ultimo = X.iloc[[-1]].copy()

    try:
        prob = modelo.predict_proba(ultimo)[0][1]
    except:
        return None, None

    if prob >= 0.65:
        return "analisis_alineado", float(prob)

    if prob <= 0.42:
        return None, float(prob)

    return None, float(prob)

# ============================================================
# PROCESAR JUGADA (INTEGRADO CON FILTRO ANTI-TRAMPA)
# ============================================================

def procesar_jugada(rango, color, fuerza):
    st.session_state.valores.append((rango, color))

    limite = None

    try:
        limite = float(rango.split("/")[0])
        movimiento = (+1 if limite >= 2.0 else -1)
    except:
        if rango == "10+":
            limite = 10.0
            movimiento = +1
        else:
            movimiento = 0

    st.session_state.hilo.append(movimiento)

    if (
        st.session_state.ultima_senal == "analisis_alineado"
        and st.session_state.resultado is None
        and limite is not None
    ):
        if limite >= 2.0:
            st.session_state.aciertos += 1
            st.session_state.resultado = "acierto"
            st.session_state.ultima_senal = None
        else:
            st.session_state.resultado = "segunda oportunidad"
            st.session_state.ultima_senal = "analisis_segunda"

    elif (
        st.session_state.ultima_senal == "analisis_segunda"
        and st.session_state.resultado == "segunda oportunidad"
        and limite is not None
    ):
        if limite >= 2.0:
            st.session_state.aciertos += 1
            st.session_state.resultado = "acierto"
        else:
            st.session_state.fallos += 1
            st.session_state.resultado = "fallo"

        st.session_state.ultima_senal = None

    conteo_total = sum(st.session_state.hilo)

    nueva_jugada = {
        "rango": rango,
        "color": color,
        "conteo_hi_lo": conteo_total,
        "fuerza": fuerza,
        "senal": st.session_state.ultima_senal,
        "resultado": st.session_state.resultado
    }

    st.session_state.historial = pd.concat(
        [
            st.session_state.historial,
            pd.DataFrame([nueva_jugada])
        ],
        ignore_index=True
    )

    historial = st.session_state.historial

    if st.session_state.resultado is None:
        # EVALUAR FILTRO ANTI-TRAMPA ANTES DE EMITIR SEÑAL
        if evaluar_filtro_anti_trampa(historial):
            decision_base = motor_analitico(historial)
            decision_ai, prob_ai = prediccion_ai(historial)

            if decision_ai == "analisis_alineado":
                st.session_state.ultima_senal = "analisis_alineado"
            else:
                st.session_state.ultima_senal = decision_base
        else:
            st.session_state.ultima_senal = None

def fn_retroceder():
    if st.session_state.valores:
        st.session_state.valores.pop()

    if st.session_state.hilo:
        st.session_state.hilo.pop()

    if not st.session_state.historial.empty:
        st.session_state.historial = (
            st.session_state.historial.iloc[:-1]
            .reset_index(drop=True)
        )

    st.session_state.ultima_senal = None
    st.session_state.resultado = None

def fn_reiniciar():
    st.session_state.valores = []
    st.session_state.aciertos = 0
    st.session_state.fallos = 0
    st.session_state.hilo = []
    st.session_state.ultima_senal = None
    st.session_state.resultado = None
    st.session_state.historial = pd.DataFrame(
        columns=[
            "rango",
            "color",
            "conteo_hi_lo",
            "fuerza",
            "senal",
            "resultado"
        ]
    )
    st.session_state.model_ready = False
    st.session_state.model_version = 0
    st.session_state.model_last_train_size = 0
    st.session_state.modelo_ai = None
    st.session_state.feature_columns = None

# ============================================================
# RELOJ INTERACTIVO (+2 MIN ESTÁTICO & ALERTA VERDE 10 SEG)
# ============================================================

components.html("""
    <div style="display: flex; gap: 8px; align-items: center;">
        <div id="clock" onclick="setTargetTime()" style="
            background-color: #1a1a1a;
            color: #ffffff;
            border: 1px solid #333333;
            font-family: monospace;
            font-size: 11px;
            padding: 3px 8px;
            border-radius: 4px;
            width: max-content;
            cursor: pointer;
            user-select: none;
            transition: all 0.2s ease;
        " title="Haz clic para fijar marca estática a +2 minutos">00:00:00</div>

        <div id="clock_target" style="
            display: none;
            background-color: #1a1a1a;
            color: #58a6ff;
            border: 1px solid #30363d;
            font-family: monospace;
            font-size: 11px;
            padding: 3px 8px;
            border-radius: 4px;
            width: max-content;
            user-select: none;
        ">00:00:00</div>
    </div>

    <script>
        let targetTimestamp = null;

        function setTargetTime() {
            const now = new Date();
            targetTimestamp = now.getTime() + (2 * 60 * 1000);
            
            const targetDate = new Date(targetTimestamp);
            const targetStr = targetDate.toTimeString().split(' ')[0];
            
            const targetEl = document.getElementById('clock_target');
            targetEl.innerText = targetStr;
            targetEl.style.display = 'block';
        }

        function updateClock() {
            const now = new Date();
            const currentTimestamp = now.getTime();
            const timeString = now.toTimeString().split(' ')[0];
            
            const clockEl = document.getElementById('clock');
            const targetEl = document.getElementById('clock_target');
            
            clockEl.innerText = timeString;

            if (targetTimestamp !== null) {
                const diffSeconds = (targetTimestamp - currentTimestamp) / 1000;

                if (diffSeconds <= 10 && diffSeconds > 0) {
                    clockEl.style.color = '#00ff66';
                    clockEl.style.borderColor = '#00ff66';
                    clockEl.style.boxShadow = '0px 0px 8px rgba(0, 255, 102, 0.6)';
                } else if (diffSeconds <= 0) {
                    targetTimestamp = null;
                    targetEl.style.display = 'none';
                    clockEl.style.color = '#ffffff';
                    clockEl.style.borderColor = '#333333';
                    clockEl.style.boxShadow = 'none';
                } else {
                    clockEl.style.color = '#ffffff';
                    clockEl.style.borderColor = '#333333';
                    clockEl.style.boxShadow = 'none';
                }
            } else {
                clockEl.style.color = '#ffffff';
                clockEl.style.borderColor = '#333333';
                clockEl.style.boxShadow = 'none';
            }
        }

        setInterval(updateClock, 1000);
        updateClock();
    </script>
""", height=30)

# ============================================================
# LIENZO GRÁFICO (PLOTLY PRO CON LÍNEA DE CUADRÍCULA RESALTADA)
# ============================================================

x_full = list(range(len(st.session_state.valores)))
y_full = []
conteo = 0

for movimiento in st.session_state.hilo:
    conteo += movimiento
    y_full.append(conteo)

colors_full = [c for _, c in st.session_state.valores]

# Ventana deslizante (máximo 80 puntos visibles)
CAPACIDAD_MAX = 80
if len(x_full) > CAPACIDAD_MAX:
    x = x_full[-CAPACIDAD_MAX:]
    y = y_full[-CAPACIDAD_MAX:]
    colors = colors_full[-CAPACIDAD_MAX:]
else:
    x = x_full
    y = y_full
    colors = colors_full

if y_full:
    ultimo_y = y_full[-1]
    color_marco = "#00ff66" if ultimo_y > 0 else ("#ff3333" if ultimo_y < 0 else "#888888")
else:
    ultimo_y = 0
    color_marco = "#00ff66"

fig = go.Figure()

# Definición de la lista de formas (shapes)
shapes_list = [
    # Marco exterior
    dict(
        type="rect",
        xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color=color_marco, width=3)
    )
]

# Si hay puntos ingresados, dibujamos la línea horizontal que resalta sobre la cuadrícula en 'ultimo_y'
if y_full:
    shapes_list.append(
        dict(
            type="line",
            xref="paper",
            yref="y",
            x0=0,
            x1=1,
            y0=ultimo_y,
            y1=ultimo_y,
            line=dict(
                color="#00e5ff",  # Azul neón que destaca sobre el fondo gris
                width=2
            ),
            layer="below"
        )
    )

if y:
    # Línea blanca de tendencia principal
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines+markers',
        line=dict(color='white', width=2),
        marker=dict(size=8, color=colors, line=dict(width=1, color='white')),
        hoverinfo='skip'
    ))

    centro_x = (min(x) + max(x)) / 2
    centro_y = (min(y) + max(y)) / 2

    texto_cartel = None
    color_cartel = "#00ff66"

    if st.session_state.ultima_senal == "analisis_alineado":
        texto_cartel = "ENTRADA"
        color_cartel = "#00ff66"
    elif st.session_state.resultado == "acierto":
        texto_cartel = "ACIERTO"
        color_cartel = "#00ff66"
        st.session_state.resultado = None
    elif st.session_state.resultado == "fallo":
        texto_cartel = "FALLO"
        color_cartel = "#ff3333"
        st.session_state.resultado = None
    elif st.session_state.resultado == "segunda oportunidad":
        texto_cartel = "SEGUNDA OPORTUNIDAD"
        color_cartel = "#ffcc00"

    if texto_cartel:
        fig.add_annotation(
            x=centro_x, y=centro_y,
            text=texto_cartel,
            showarrow=False,
            font=dict(size=16, color=color_cartel, family="sans-serif"),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor=color_cartel,
            borderwidth=1
        )

fig.update_layout(
    margin=dict(l=35, r=10, t=10, b=10),
    height=250,
    paper_bgcolor='#000000',
    plot_bgcolor='#000000',
    xaxis=dict(
        showticklabels=False,
        showgrid=True,
        gridcolor='#222222',
        zeroline=False
    ),
    yaxis=dict(
        showticklabels=True,
        tickfont=dict(color='white', size=10),
        showgrid=True,
        gridcolor='#222222',
        zeroline=False
    ),
    shapes=shapes_list
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ============================================================
# BOTONERA REEMPLAZADA CON CALLBACKS INSTANTÁNEOS
# ============================================================

cols1 = st.columns(5, gap="small")
cols2 = st.columns(5, gap="small")

for i, (rango, datos) in enumerate(list(rangos.items())[:5]):
    color, fuerza = datos
    cols1[i].button(rango, key=f"btn_{i}", on_click=procesar_jugada, args=(rango, color, fuerza), use_container_width=True)

for i, (rango, datos) in enumerate(list(rangos.items())[5:]):
    color, fuerza = datos
    cols2[i].button(rango, key=f"btn_{i+5}", on_click=procesar_jugada, args=(rango, color, fuerza), use_container_width=True)

# ============================================================
# BARRA DE RONDAS
# ============================================================

total_rondas = len(st.session_state.historial)
st.markdown(f"<div class='rondas-badge'>⚡ RONDAS REGISTRADAS: {total_rondas} ⚡</div>", unsafe_allow_html=True)

# ============================================================
# CONTROLES Y MÉTRICAS INFERIORES
# ============================================================

cols_control = st.columns(4, gap="small")

cols_control[0].button("Retroceder", on_click=fn_retroceder, use_container_width=True)
cols_control[1].button("Reiniciar", on_click=fn_reiniciar, use_container_width=True)

conteo_total = sum(st.session_state.hilo) if st.session_state.hilo else 0

cols_control[2].markdown(f"<p style='color:white; font-size:11px; text-align:center; margin-top:6px;'>📊 <b>Hi-Lo:</b> {conteo_total}</p>", unsafe_allow_html=True)
cols_control[3].markdown(f"<p style='color:white; font-size:11px; text-align:center; margin-top:6px;'>✅ {st.session_state.aciertos} | ❌ {st.session_state.fallos}</p>", unsafe_allow_html=True)

# ============================================================
# NOMBRE DEL PROYECTO
# ============================================================

st.markdown("<div class='title-abigail'>⚡ ABIGAIL CHART PRO ⚡</div>", unsafe_allow_html=True)
