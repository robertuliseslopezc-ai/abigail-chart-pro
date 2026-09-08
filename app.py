import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Configuración de página ancha para ajustarse a dispositivos móviles y escritorios
st.set_page_config(
    page_title="Abigail Chart Pro",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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

if "acumulado_fuerza" not in st.session_state:
    st.session_state.acumulado_fuerza = 0

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

if "modo_vista" not in st.session_state:
    st.session_state.modo_vista = "Ambos"

if "historial" not in st.session_state:
    st.session_state.historial = pd.DataFrame(
        columns=[
            "rango",
            "color",
            "conteo_hi_lo",
            "fuerza",
            "open",
            "high",
            "low",
            "close",
            "senal",
            "resultado",
        ]
    )

# ============================================================
# ESTILOS CSS
# ============================================================
st.markdown(
    """
    <style>
    header, footer {visibility: hidden;}
    
    .block-container {
        padding-top: 0.2rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }
    
    .stApp {
        background-color: #000000 !important;
    }

    [data-testid="stPlotlyChart"] {
        background-color: #000000 !important;
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
""",
    unsafe_allow_html=True,
)

# ============================================================
# TABLA DE FUERZA
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
    "10+": ("#ff00ff", 5),
}

# ============================================================
# MOTOR ANALÍTICO & VENTANA MÓVIL CON LECTURA DE FUERZA
# ============================================================
def verificar_permitir_senales(historial):
    if historial is None or len(historial) < 10:
        return True
    
    ultimas_10 = historial.tail(10)

    def es_mayor_igual_2x(rango):
        try:
            val_str = str(rango).strip()
            if val_str == "10+":
                return True
            limite = float(val_str.split("/")[0].strip())
            return limite >= 2.0
        except:
            return False

    conteo_positivas = ultimas_10["rango"].apply(es_mayor_igual_2x).sum()
    if conteo_positivas < 5:
        return False

    fuerzas_10 = pd.to_numeric(ultimas_10["fuerza"], errors="coerce").fillna(0)
    
    if fuerzas_10.sum() < 0:
        return False
        
    fuerzas_ultimas_3 = fuerzas_10.tail(3).sum()
    if fuerzas_ultimas_3 <= -3:
        return False

    return True

def evaluar_filtro_ema(historial, periodo=20):
    if len(historial) < 2:
        return True
    df_close = pd.to_numeric(historial["close"], errors="coerce")
    if len(df_close) < periodo:
        ema = df_close.ewm(span=len(df_close), adjust=False).mean()
    else:
        ema = df_close.ewm(span=periodo, adjust=False).mean()
    
    precio_actual = df_close.iloc[-1]
    ema_actual = ema.iloc[-1]
    return precio_actual >= ema_actual

def lectura_fuerza(fuerzas):
    if not fuerzas:
        return None, 0
    actual = fuerzas[-1]
    tendencia = "alcista" if actual > 0 else ("bajista" if actual < 0 else "neutral")
    return tendencia, abs(actual)

def lectura_continuidad(fuerzas):
    if not fuerzas:
        return False, 0, "neutral"
    actual = fuerzas[-1]
    direccion = "alcista" if actual > 0 else ("bajista" if actual < 0 else "neutral")
    consecutivos = 1
    for i in range(len(fuerzas) - 2, -1, -1):
        anterior = fuerzas[i]
        if (actual > 0 and anterior > 0) or (actual < 0 and anterior < 0):
            consecutivos += 1
        else:
            break
    return consecutivos >= 3, consecutivos, direccion

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
    if actual < 0 and max(anteriores) > 0:
        return True, "bajista"
    if actual > 0 and min(anteriores) < 0:
        return True, "alcista"
    return False, "ninguno"

def motor_analitico_c2(historial):
    if len(historial) < 5:
        return None
    fuerzas = pd.to_numeric(historial["fuerza"], errors="coerce").dropna().tolist()
    if len(fuerzas) < 5:
        return None

    tendencia_fuerza, intensidad = lectura_fuerza(fuerzas)
    continuidad, _, dir_cont = lectura_continuidad(fuerzas)
    impulso, dir_imp = lectura_impulso(fuerzas)
    retroceso, _ = lectura_retroceso(fuerzas)

    condiciones = 0
    if tendencia_fuerza == "alcista" and intensidad >= 1:
        condiciones += 1
    if continuidad and dir_cont == "alcista":
        condiciones += 1
    if impulso and dir_imp == "alcista":
        condiciones += 1
    if not retroceso:
        condiciones += 1

    return "analisis_alineado" if condiciones >= 3 else None

def extraer_limite(rango):
    try:
        if str(rango) == "10+":
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
        "conteo_prev2", "delta_fuerza", "media_fuerza_3", "std_fuerza_5", "suma_3",
        "suma_5", "color_red", "color_white", "color_green", "color_yellow",
        "abs_fuerza", "signo_fuerza", "cambio_signo"
    ]
    return df[feature_cols].fillna(0), df["movimiento_real"], feature_cols, df

def entrenar_modelo_si_aplica(historial):
    try:
        res = construir_features(historial)
        if res is None:
            return None
        X, y, feature_cols, df = res
        if len(X) < 20 or len(set(y.tolist())) < 2:
            return None
        if st.session_state.modelo_ai is None or len(df) > st.session_state.model_last_train_size:
            modelo = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))
            ])
            modelo.fit(X, y)
            st.session_state.modelo_ai = modelo
            st.session_state.feature_columns = feature_cols
            st.session_state.model_ready = True
            st.session_state.model_last_train_size = len(df)
            st.session_state.model_version += 1
        return st.session_state.modelo_ai
    except:
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
    try:
        prob = modelo.predict_proba(X.iloc[[-1]])[0][1]
    except:
        return None, None
    return ("analisis_alineado", float(prob)) if prob >= 0.65 else (None, float(prob))

# ============================================================
# LOGICA DE JUGADA
# ============================================================
def procesar_jugada(rango, color, fuerza):
    st.session_state.valores.append((rango, color))

    limite = None
    try:
        limite = float(rango.split("/")[0])
        movimiento = +1 if limite >= 2.0 else -1
    except:
        if rango == "10+":
            limite = 10.0
            movimiento = +1
        else:
            movimiento = 0

    st.session_state.hilo.append(movimiento)
    conteo_total = sum(st.session_state.hilo)

    open_p = st.session_state.acumulado_fuerza
    close_p = open_p + fuerza
    st.session_state.acumulado_fuerza = close_p

    if fuerza > 0:
        high_p = close_p + (fuerza * 0.15)
        low_p = open_p - 0.2
    elif fuerza < 0:
        high_p = open_p + 0.2
        low_p = close_p - (abs(fuerza) * 0.15)
    else:
        high_p = open_p + 0.5
        low_p = open_p - 0.5

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

    nueva_jugada = {
        "rango": rango,
        "color": color,
        "conteo_hi_lo": conteo_total,
        "fuerza": fuerza,
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close_p,
        "senal": st.session_state.ultima_senal,
        "resultado": st.session_state.resultado,
    }

    st.session_state.historial = pd.concat(
        [st.session_state.historial, pd.DataFrame([nueva_jugada])],
        ignore_index=True,
    )

    historial = st.session_state.historial
    if st.session_state.resultado is None:
        if verificar_permitir_senales(historial) and evaluar_filtro_ema(historial, periodo=20):
            decision_base = motor_analitico_c2(historial)
            decision_ai, _ = prediccion_ai(historial)
            if decision_ai == "analisis_alineado" or decision_base == "analisis_alineado":
                st.session_state.ultima_senal = "analisis_alineado"
            else:
                st.session_state.ultima_senal = None
        else:
            st.session_state.ultima_senal = None

def fn_retroceder():
    if st.session_state.valores:
        st.session_state.valores.pop()
    if st.session_state.hilo:
        st.session_state.hilo.pop()
    if not st.session_state.historial.empty:
        ultimo = st.session_state.historial.iloc[-1]
        st.session_state.acumulado_fuerza = ultimo["open"]
        st.session_state.historial = st.session_state.historial.iloc[:-1].reset_index(drop=True)

    st.session_state.ultima_senal = None
    st.session_state.resultado = None

def fn_reiniciar():
    st.session_state.valores = []
    st.session_state.aciertos = 0
    st.session_state.fallos = 0
    st.session_state.hilo = []
    st.session_state.acumulado_fuerza = 0
    st.session_state.ultima_senal = None
    st.session_state.resultado = None
    st.session_state.historial = pd.DataFrame(
        columns=[
            "rango", "color", "conteo_hi_lo", "fuerza",
            "open", "high", "low", "close", "senal", "resultado"
        ]
    )

# ============================================================
# RELOJ INTERACTIVO
# ============================================================
components.html(
    """
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
        ">00:00:00</div>
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
        ">00:00:00</div>
    </div>
    <script>
        let targetTimestamp = null;
        function setTargetTime() {
            const now = new Date();
            targetTimestamp = now.getTime() + (2 * 60 * 1000);
            const targetStr = new Date(targetTimestamp).toTimeString().split(' ')[0];
            const targetEl = document.getElementById('clock_target');
            targetEl.innerText = targetStr;
            targetEl.style.display = 'block';
        }
        function updateClock() {
            const now = new Date();
            const currentTimestamp = now.getTime();
            document.getElementById('clock').innerText = now.toTimeString().split(' ')[0];
            if (targetTimestamp !== null && (targetTimestamp - currentTimestamp) / 1000 <= 0) {
                targetTimestamp = null;
                document.getElementById('clock_target').style.display = 'none';
            }
        }
        setInterval(updateClock, 1000);
        updateClock();
    </script>
""",
    height=30,
)

# ============================================================
# SELECTOR DE VISTA
# ============================================================
st.session_state.modo_vista = st.radio(
    "Vista",
    options=["Tendencia", "Velas", "Ambos"],
    index=["Tendencia", "Velas", "Ambos"].index(st.session_state.modo_vista),
    horizontal=True,
    label_visibility="collapsed",
)

# ============================================================
# GRAFICADOR UNIFICADO CON AJUSTE DE POSICIÓN EN MODO "AMBOS"
# ============================================================
@st.fragment
def renderizar_graficos_estilo_imagen():
    hist_df = st.session_state.historial
    modo = st.session_state.modo_vista

    fig = go.Figure()
    shapes_list = []

    # --- DATOS DE VELAS ---
    if not hist_df.empty and modo in ["Velas", "Ambos"]:
        df_v = hist_df.tail(60).reset_index(drop=True)
        x_v = list(range(len(df_v)))

        fig.add_trace(
            go.Candlestick(
                x=x_v,
                open=df_v["open"],
                high=df_v["high"],
                low=df_v["low"],
                close=df_v["close"],
                increasing_line_color="#00ff66",
                decreasing_line_color="#ff3333",
                increasing_fillcolor="#00ff66",
                decreasing_fillcolor="#ff3333",
                name="Velas",
            )
        )

        df_close = pd.to_numeric(hist_df["close"], errors="coerce")
        ema_vals = df_close.ewm(span=min(len(df_close), 20), adjust=False).mean().tail(len(df_v))
        
        fig.add_trace(
            go.Scatter(
                x=x_v,
                y=ema_vals,
                mode="lines",
                line=dict(color="#00e5ff", width=1.5),
                name="EMA 20",
                hoverinfo="skip",
            )
        )

        ultimo_close = df_v["close"].iloc[-1]
        shapes_list.append(
            dict(
                type="line",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=ultimo_close,
                y1=ultimo_close,
                line=dict(color="#ff00ff", width=1.2, dash="dash"),
            )
        )

    # --- DATOS DE TENDENCIA ---
    if modo in ["Tendencia", "Ambos"]:
        x_full = list(range(len(st.session_state.valores)))
        y_full = []
        conteo = 0
        for m in st.session_state.hilo:
            conteo += m
            y_full.append(conteo)

        colors_full = [c for _, c in st.session_state.valores]

        if modo == "Ambos":
            fig.add_trace(
                go.Scatter(
                    x=x_full[-40:],
                    y=y_full[-40:],
                    mode="lines+markers",
                    line=dict(color="white", width=1.5),
                    marker=dict(size=6, color=colors_full[-40:]),
                    xaxis="x2",
                    yaxis="y2",
                    name="Tendencia PIP",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x_full[-60:],
                    y=y_full[-60:],
                    mode="lines+markers",
                    line=dict(color="white", width=2),
                    marker=dict(size=8, color=colors_full[-60:]),
                )
            )
            if y_full:
                ultimo_y = y_full[-1]
                shapes_list.append(
                    dict(
                        type="line",
                        xref="paper",
                        yref="y",
                        x0=0,
                        x1=1,
                        y0=ultimo_y,
                        y1=ultimo_y,
                        line=dict(color="#00e5ff", width=1.5, dash="dash"),
                    )
                )

    # --- Carteles de Señales de Entrada ---
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
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            text=texto_cartel,
            showarrow=False,
            font=dict(size=18, color=color_cartel, family="sans-serif"),
            bgcolor="rgba(0,0,0,0.9)",
            bordercolor=color_cartel,
            borderwidth=1.5,
        )

    layout_args = dict(
        template="plotly_dark",
        margin=dict(l=30, r=10, t=10, b=10),
        height=320,
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        showlegend=False,
        shapes=shapes_list,
        xaxis=dict(
            showticklabels=False,
            gridcolor="#111111",
            zeroline=False,
            showgrid=True,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            showticklabels=True,
            tickfont=dict(color="white", size=10),
            gridcolor="#111111",
            zeroline=False,
            showgrid=True,
        ),
    )

    # RE-UBICACIÓN DEL SUB-GRAFICO (PARTE SUPERIOR DERECHA)
    if modo == "Ambos":
        layout_args["xaxis2"] = dict(
            domain=[0.62, 0.98],
            anchor="y2",
            showticklabels=False,
            gridcolor="#111111",
        )
        layout_args["yaxis2"] = dict(
            domain=[0.68, 0.98],
            anchor="x2",
            showticklabels=False,
            gridcolor="#111111",
        )

    fig.update_layout(**layout_args)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

renderizar_graficos_estilo_imagen()

# ============================================================
# BOTONERA PRINCIPAL
# ============================================================
cols1 = st.columns(5, gap="small")
cols2 = st.columns(5, gap="small")

for i, (rango, datos) in enumerate(list(rangos.items())[:5]):
    color, fuerza = datos
    cols1[i].button(
        rango,
        key=f"btn_{i}",
        on_click=procesar_jugada,
        args=(rango, color, fuerza),
        use_container_width=True,
    )

for i, (rango, datos) in enumerate(list(rangos.items())[5:]):
    color, fuerza = datos
    cols2[i].button(
        rango,
        key=f"btn_{i+5}",
        on_click=procesar_jugada,
        args=(rango, color, fuerza),
        use_container_width=True,
    )

# ============================================================
# BARRA DE RONDAS Y CONTROLES
# ============================================================
total_rondas = len(st.session_state.historial)
st.markdown(
    f"<div class='rondas-badge'>⚡ RONDAS REGISTRADAS: {total_rondas} ⚡</div>",
    unsafe_allow_html=True,
)

cols_control = st.columns(4, gap="small")
cols_control[0].button("Retroceder", on_click=fn_retroceder, use_container_width=True)
cols_control[1].button("Reiniciar", on_click=fn_reiniciar, use_container_width=True)

conteo_total = sum(st.session_state.hilo) if st.session_state.hilo else 0

cols_control[2].markdown(
    f"<p style='color:white; font-size:11px; text-align:center; margin-top:6px;'>📊 <b>Hi-Lo:</b> {conteo_total}</p>",
    unsafe_allow_html=True,
)
cols_control[3].markdown(
    f"<p style='color:white; font-size:11px; text-align:center; margin-top:6px;'>✅ {st.session_state.aciertos} | ❌ {st.session_state.fallos}</p>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='title-abigail'>⚡ ABIGAIL CHART PRO ⚡</div>",
    unsafe_allow_html=True,
)
