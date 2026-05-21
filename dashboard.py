"""
Dashboard web interactivo — Asistente de Inversión
Ejecutar: python -m streamlit run dashboard.py
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modulos.mi_portafolio import (
    cargar_portafolio, valor_total_cop,
    obtener_retornos_portafolio_real,
    construir_retornos_para_analisis,
    PORTAFOLIO_SUGERIDO,
)
from modulos.metricas import (
    retorno_anualizado, volatilidad_anualizada,
    sharpe_ratio, maximo_drawdown,
)
from modulos.montecarlo import simular_montecarlo, estadisticas_montecarlo
from modulos.optimizador import optimizar_sharpe, frontera_eficiente
from modulos.datos import descargar_precios, calcular_retornos_diarios
import yfinance as yf


# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Asistente de Inversión | David",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORES_ACTIVOS = ["#00d4ff", "#00a8cc", "#ffaa00", "#00ff88", "#aa88ff", "#ff8844"]
BG_DARK   = "#1a1d27"
BG_DARKER = "#0f1117"

_LAYOUT_BASE = dict(
    paper_bgcolor=BG_DARK,
    plot_bgcolor=BG_DARK,
    font=dict(color="white"),
    legend=dict(font=dict(color="white"), bgcolor=BG_DARKER),
)


def dark_fig(**kwargs):
    """Plotly figure con fondo oscuro."""
    layout = {**_LAYOUT_BASE, **kwargs}
    fig = go.Figure()
    fig.update_layout(**layout)
    return fig


def dark_axes(fig):
    fig.update_xaxes(color="gray", gridcolor="#2e3148")
    fig.update_yaxes(color="gray", gridcolor="#2e3148")
    return fig


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _retornos_real(periodo: str) -> pd.DataFrame:
    port = cargar_portafolio()
    return obtener_retornos_portafolio_real(port, periodo)


@st.cache_data(ttl=3600, show_spinner=False)
def _retornos_analisis(periodo: str):
    port = cargar_portafolio()
    return construir_retornos_para_analisis(port, periodo)


@st.cache_data(ttl=3600, show_spinner=False)
def _retornos_opciones(periodo: str):
    port = cargar_portafolio()
    todos = port.get("opciones_inversion", []) + port.get("alternativas_sugeridas", [])
    tickers = list(dict.fromkeys(
        op["ticker_yfinance"] for op in todos if op.get("ticker_yfinance")
    ))
    mapa = {op["ticker_yfinance"]: op["nombre_corto"]
            for op in todos if op.get("ticker_yfinance")}
    ids_opcion = {op["ticker_yfinance"]
                  for op in port.get("opciones_inversion", [])
                  if op.get("ticker_yfinance")}
    precios  = descargar_precios(tickers, periodo)
    retornos = calcular_retornos_diarios(precios)
    return precios, retornos, mapa, ids_opcion


@st.cache_data(ttl=3600, show_spinner=False)
def _optimizar(periodo: str):
    ret_df, pw_act, pw_sug = _retornos_analisis(periodo)
    return optimizar_sharpe(ret_df), pw_act, pw_sug, ret_df


@st.cache_data(ttl=3600, show_spinner=False)
def _frontera(periodo: str):
    ret_df, _, _ = _retornos_analisis(periodo)
    return frontera_eficiente(ret_df, n_portafolios=5000)


@st.cache_data(ttl=3600, show_spinner=False)
def _montecarlo(periodo: str, capital: int, dias: int, n_sim: int):
    ret_df, pw_act, pw_sug = _retornos_analisis(periodo)
    activos = list(ret_df.columns)
    w_act = np.array([pw_act.get(a, 0) for a in activos])
    w_sug = np.array([pw_sug.get(a, 0) for a in activos])
    r_act = (ret_df @ w_act).dropna()
    r_sug = (ret_df @ w_sug).dropna()
    tray_act = simular_montecarlo(r_act, capital, dias, n_sim, semilla=42)
    tray_sug = simular_montecarlo(r_sug, capital, dias, n_sim, semilla=42)
    return tray_act, tray_sug, pw_act, pw_sug, activos, ret_df


@st.cache_data(ttl=3600, show_spinner=False)
def _descargar_ticker(ticker: str, periodo: str) -> pd.Series | None:
    """Descarga precios de cierre para un ticker individual. Cacheado por ticker+periodo."""
    try:
        data = yf.download(ticker, period=periodo, auto_adjust=True, progress=False)
        if data.empty or len(data) < 30:
            return None
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.dropna()
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Parámetros")
    periodo = st.selectbox("Período histórico", ["1y", "2y", "5y"], index=1)
    st.divider()
    st.subheader("Monte Carlo")
    anos_mc = st.slider("Años a simular", 1, 5, 2)
    n_sim   = st.select_slider("Simulaciones", [500, 1000, 3000], value=1000)
    st.divider()
    st.caption("📡 Datos: Yahoo Finance")
    st.caption("💱 USD/COP: ticker COP=X")
    st.caption("*Fondos AVC: retornos sintéticos (GBM)")

portafolio = cargar_portafolio()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Mi Portafolio",
    "🔍 Opciones de Inversión",
    "🎲 Monte Carlo",
    "⚙️ Optimizador Markowitz",
    "🧪 Explorador",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Mi Portafolio
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header(portafolio["nombre"])

    total      = valor_total_cop(portafolio)
    disponible = portafolio["disponible_cop"]
    comision   = portafolio["comision_por_transaccion_cop"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Invertido",  f"${total:,.0f} COP")
    c2.metric("🏦 Disponible", f"${disponible:,.0f} COP")
    c3.metric("📦 Total",      f"${total + disponible:,.0f} COP")
    c4.metric("📈 Activos",    str(len(portafolio["activos"])))

    st.divider()

    # Composición
    nombres = [a["nombre_corto"] for a in portafolio["activos"]]
    valores  = [a["valor_cop"]    for a in portafolio["activos"]]

    pie_col, bar_col = st.columns(2)

    with pie_col:
        fig_pie = go.Figure(go.Pie(
            labels=nombres, values=valores, hole=0.55,
            marker_colors=COLORES_ACTIVOS[:len(nombres)],
            textinfo="percent+label",
        ))
        fig_pie.update_layout(
            **_LAYOUT_BASE,
            title=dict(text="Composición del portafolio", font=dict(color="white")),
            margin=dict(t=50, b=10, l=10, r=10),
            showlegend=False,
            annotations=[dict(
                text=f"${total/1e6:.2f}M<br>COP",
                x=0.5, y=0.5, font_size=16, font_color="white", showarrow=False,
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with bar_col:
        df_bar = pd.DataFrame({"Activo": nombres, "Valor COP": valores,
                                "Pct": [v / total for v in valores]})
        fig_bar = px.bar(
            df_bar, x="Valor COP", y="Activo", orientation="h",
            title="Valor por activo (COP)",
            color="Valor COP", color_continuous_scale="Teal",
            text=df_bar["Pct"].map(lambda p: f"{p:.1%}"),
        )
        fig_bar.update_layout(
            **_LAYOUT_BASE, yaxis_title="",
            coloraxis_showscale=False,
            xaxis=dict(tickformat="$,.0f", color="gray", gridcolor="#2e3148"),
            yaxis=dict(color="gray"),
        )
        fig_bar.update_traces(textposition="outside", textfont_color="white")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Métricas históricas
    st.subheader(f"Métricas históricas ({periodo})")

    with st.spinner("Descargando datos históricos..."):
        retornos_real = _retornos_real(periodo)

    pesos_r  = {a["id"]: a["valor_cop"] / total for a in portafolio["activos"]}
    info_map = {a["id"]: a for a in portafolio["activos"]}

    filas_m = []
    for id_act in retornos_real.columns:
        s   = retornos_real[id_act].dropna()
        r   = retorno_anualizado(s)
        v   = volatilidad_anualizada(s)
        sh  = sharpe_ratio(s)
        dd  = (1 + s).cumprod().pipe(lambda p: (p - p.cummax()) / p.cummax()).min()
        inf = info_map.get(id_act, {})
        filas_m.append({
            "Activo":       inf.get("nombre_corto", id_act),
            "Peso":         f"{pesos_r.get(id_act, 0):.1%}",
            "Ret. Anual":   f"{r:+.1%}",
            "Volatilidad":  f"{v:.1%}",
            "Sharpe":       round(sh, 2),
            "Max Drawdown": f"{dd:.1%}",
            "Fuente":       "yfinance" if inf.get("ticker_yfinance") else "sintético*",
        })

    w_r = np.array([pesos_r.get(c, 0) for c in retornos_real.columns])
    rp  = (retornos_real @ w_r).dropna()
    filas_m.append({
        "Activo":       "PORTAFOLIO TOTAL",
        "Peso":         "100%",
        "Ret. Anual":   f"{retorno_anualizado(rp):+.1%}",
        "Volatilidad":  f"{volatilidad_anualizada(rp):.1%}",
        "Sharpe":       round(sharpe_ratio(rp), 2),
        "Max Drawdown": f"{(1+rp).cumprod().pipe(lambda p:(p-p.cummax())/p.cummax()).min():.1%}",
        "Fuente":       "—",
    })
    st.dataframe(pd.DataFrame(filas_m), use_container_width=True, hide_index=True)
    st.caption("*Sintético: GBM calibrado con EA declarado + volatilidad estimada del fondo AVC.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Opciones de Inversión
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Opciones de Inversión")

    disponible = portafolio["disponible_cop"]
    comision   = portafolio["comision_por_transaccion_cop"]
    neto       = disponible - comision

    st.info(
        f"💰 Disponible: **${disponible:,.0f} COP**  |  "
        f"Comisión: **${comision:,.0f} COP**  |  "
        f"Neto operativo: **${neto:,.0f} COP**"
    )

    with st.spinner("Descargando historial de opciones..."):
        precios_op, retornos_op, mapa_nombre, ids_opcion = _retornos_opciones(periodo)

    # Tabla rendimiento histórico
    filas_op = []
    for ticker in precios_op.columns:
        s   = retornos_op[ticker].dropna()
        r   = retorno_anualizado(s)
        v   = volatilidad_anualizada(s)
        sh  = sharpe_ratio(s)
        dd  = maximo_drawdown(precios_op[ticker].dropna())
        cat = "Opción TRii" if ticker in ids_opcion else "Alternativa sugerida"
        filas_op.append({
            "Activo":      mapa_nombre.get(ticker, ticker),
            "Ticker":      ticker,
            "Ret. Anual":  f"{r:+.1%}",
            "Volatilidad": f"{v:.1%}",
            "Sharpe":      round(sh, 2),
            "Max DD":      f"{dd:.1%}",
            "Categoría":   cat,
            "_sh":         sh,
        })

    filas_op.sort(key=lambda x: -x["_sh"])
    st.subheader(f"Rendimiento histórico ({periodo}) — ordenado por Sharpe")
    df_op = pd.DataFrame(filas_op).drop(columns=["_sh"])
    st.dataframe(df_op, use_container_width=True, hide_index=True)
    st.caption("Retornos en USD (proxy de ETF en Yahoo Finance).")

    # Gráfica Sharpe
    df_sh = pd.DataFrame([
        {"Activo": r["Activo"], "Sharpe": r["Sharpe"], "Categoría": r["Categoría"]}
        for r in filas_op
    ]).sort_values("Sharpe")

    fig_sh = px.bar(
        df_sh, x="Sharpe", y="Activo", orientation="h",
        color="Categoría", title="Sharpe Ratio por activo",
        color_discrete_map={"Opción TRii": "#ffaa00", "Alternativa sugerida": "#aa88ff"},
    )
    fig_sh.update_layout(**_LAYOUT_BASE, yaxis_title="",
                          xaxis=dict(color="gray", gridcolor="#2e3148"),
                          yaxis=dict(color="gray"))
    fig_sh.add_vline(x=1.0, line_dash="dot", line_color="#00ff88",
                     annotation_text="Sharpe = 1", annotation_font_color="#00ff88")
    st.plotly_chart(fig_sh, use_container_width=True)

    # ¿Qué puedo comprar?
    st.subheader(f"¿Qué comprar con los ${disponible:,.0f} COP disponibles?")
    opciones_lista = portafolio.get("opciones_inversion", [])
    filas_compra = []
    for op in opciones_lista:
        precio = op.get("precio_cop")
        nombre = op["nombre_corto"]
        if precio is None:
            filas_compra.append({"Activo": nombre, "Precio/ud (COP)": "sin precio",
                                  "Unidades": "-", "Total + comisión": "-",
                                  "Sobra (COP)": "-", "Com. efectiva": "-"})
            continue
        unidades = int(neto // precio)
        if unidades > 0 and unidades * precio + comision > disponible:
            unidades -= 1
        if unidades < 1:
            filas_compra.append({"Activo": nombre, "Precio/ud (COP)": f"${precio:,.0f}",
                                  "Unidades": 0, "Total + comisión": "Fuera de presupuesto",
                                  "Sobra (COP)": "-", "Com. efectiva": "-"})
            continue
        total_op = unidades * precio + comision
        filas_compra.append({
            "Activo":           nombre,
            "Precio/ud (COP)":  f"${precio:,.0f}",
            "Unidades":         unidades,
            "Total + comisión": f"${total_op:,.0f}",
            "Sobra (COP)":      f"${disponible - total_op:,.0f}",
            "Com. efectiva":    f"{comision / (unidades * precio):.1%}",
        })
    st.dataframe(pd.DataFrame(filas_compra), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Monte Carlo
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header(f"Simulación Monte Carlo — {anos_mc} año(s), {n_sim:,} simulaciones")

    total_inv = valor_total_cop(portafolio)
    capital_input = st.number_input(
        "Capital inicial (COP)", value=int(total_inv), step=50_000, format="%d",
        help="Por defecto usa el valor total actual de tu portafolio.",
    )

    dias_sim = int(anos_mc * 252)

    with st.spinner("Simulando Monte Carlo (actual vs sugerido)..."):
        tray_act, tray_sug, pw_act, pw_sug, activos_mc, ret_df_mc = _montecarlo(
            periodo, int(capital_input), dias_sim, n_sim
        )

    st_act = estadisticas_montecarlo(tray_act)
    st_sug = estadisticas_montecarlo(tray_sug)

    # Tabla resumen
    df_mc = pd.DataFrame([
        {
            "Portafolio":          "Actual",
            "P5 — Pesimista":      f"${st_act['p05']:,.0f}  ({st_act['p05']/capital_input-1:+.1%})",
            "P50 — Probable":      f"${st_act['p50']:,.0f}  ({st_act['p50']/capital_input-1:+.1%})",
            "P95 — Optimista":     f"${st_act['p95']:,.0f}  ({st_act['p95']/capital_input-1:+.1%})",
            "Prob. ganancia":      f"{st_act['prob_ganancia']:.1%}",
            "Prob. perder > 20%":  f"{st_act['prob_perdida_20']:.1%}",
        },
        {
            "Portafolio":          "Sugerido",
            "P5 — Pesimista":      f"${st_sug['p05']:,.0f}  ({st_sug['p05']/capital_input-1:+.1%})",
            "P50 — Probable":      f"${st_sug['p50']:,.0f}  ({st_sug['p50']/capital_input-1:+.1%})",
            "P95 — Optimista":     f"${st_sug['p95']:,.0f}  ({st_sug['p95']/capital_input-1:+.1%})",
            "Prob. ganancia":      f"{st_sug['prob_ganancia']:.1%}",
            "Prob. perder > 20%":  f"{st_sug['prob_perdida_20']:.1%}",
        },
    ])
    st.dataframe(df_mc, use_container_width=True, hide_index=True)

    # Gráficas MC con plotly
    def _mc_chart(tray: np.ndarray, nombre: str, color: str, capital: float) -> go.Figure:
        eje = np.arange(tray.shape[0])
        p05 = np.percentile(tray, 5,  axis=1)
        p25 = np.percentile(tray, 25, axis=1)
        p50 = np.percentile(tray, 50, axis=1)
        p75 = np.percentile(tray, 75, axis=1)
        p95 = np.percentile(tray, 95, axis=1)

        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

        fig = go.Figure()

        # Subset de trayectorias individuales
        rng_mc = np.random.default_rng(0)
        for i in rng_mc.choice(tray.shape[1], min(80, tray.shape[1]), replace=False):
            clr = f"rgba(255,80,80,0.05)" if tray[-1, i] < capital else f"rgba({r},{g},{b},0.05)"
            fig.add_trace(go.Scatter(x=eje, y=tray[:, i], mode="lines",
                                     line=dict(width=0.5, color=clr), showlegend=False))

        # Bandas de confianza
        fig.add_trace(go.Scatter(
            x=np.concatenate([eje, eje[::-1]]),
            y=np.concatenate([p95, p05[::-1]]),
            fill="toself", fillcolor=f"rgba({r},{g},{b},0.10)",
            line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([eje, eje[::-1]]),
            y=np.concatenate([p75, p25[::-1]]),
            fill="toself", fillcolor=f"rgba({r},{g},{b},0.22)",
            line=dict(width=0), showlegend=False,
        ))

        fig.add_trace(go.Scatter(x=eje, y=p50, mode="lines",
                                 line=dict(color=color, width=2.5),
                                 name=f"Mediana: ${p50[-1]:,.0f}"))
        fig.add_trace(go.Scatter(x=eje, y=p05, mode="lines",
                                 line=dict(color="#ff6666", width=1, dash="dash"),
                                 name=f"P5: ${p05[-1]:,.0f}"))
        fig.add_trace(go.Scatter(x=eje, y=p95, mode="lines",
                                 line=dict(color="#66ff88", width=1, dash="dash"),
                                 name=f"P95: ${p95[-1]:,.0f}"))
        fig.add_hline(y=capital, line_dash="dot", line_color="#888",
                      annotation_text="Capital inicial", annotation_font_color="#888")

        fig.update_layout(
            **_LAYOUT_BASE,
            title=f"Portafolio {nombre}",
            xaxis=dict(title="Días", color="gray", gridcolor="#2e3148"),
            yaxis=dict(title="Valor COP", tickformat="$,.0f", color="gray", gridcolor="#2e3148"),
        )
        return fig

    mc_col1, mc_col2 = st.columns(2)
    with mc_col1:
        st.plotly_chart(
            _mc_chart(tray_act, "Actual",   "#00d4ff", capital_input),
            use_container_width=True,
        )
    with mc_col2:
        st.plotly_chart(
            _mc_chart(tray_sug, "Sugerido", "#00ff88", capital_input),
            use_container_width=True,
        )

    # Composición del portafolio sugerido
    with st.expander("Ver composición del portafolio sugerido"):
        sug = PORTAFOLIO_SUGERIDO
        st.caption(sug.get("descripcion", ""))
        df_sug = pd.DataFrame([
            {"Activo": k, "Peso sugerido": f"{v:.1%}"}
            for k, v in sug["pesos"].items() if v > 0
        ])
        st.dataframe(df_sug, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Optimizador Markowitz
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Optimizador Markowitz — Máximo Sharpe Ratio")

    with st.spinner("Optimizando portafolio..."):
        optimo, pw_act_opt, _, ret_df_opt = _optimizar(periodo)

    if not optimo["exito"]:
        st.error("El optimizador no convergió. Prueba con un período diferente.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpe óptimo",      f"{optimo['sharpe']:.2f}",
                  delta=f"vs {sharpe_ratio((ret_df_opt @ np.array([pw_act_opt.get(a,0) for a in ret_df_opt.columns])).dropna()):.2f} actual")
        c2.metric("Retorno anual",      f"{optimo['retorno_anual']:+.1%}")
        c3.metric("Volatilidad anual",  f"{optimo['volatilidad_anual']:.1%}")

        st.divider()

        # Comparación de pesos
        activos_opt = list(ret_df_opt.columns)
        df_pesos = pd.DataFrame({
            "Activo":       activos_opt,
            "Actual (%)":  [pw_act_opt.get(a, 0) * 100 for a in activos_opt],
            "Óptimo (%)":  [optimo["pesos"].get(a, 0) * 100 for a in activos_opt],
        })

        fig_pesos = go.Figure()
        fig_pesos.add_trace(go.Bar(name="Actual",  x=df_pesos["Activo"],
                                    y=df_pesos["Actual (%)"],  marker_color="#00d4ff"))
        fig_pesos.add_trace(go.Bar(name="Óptimo",  x=df_pesos["Activo"],
                                    y=df_pesos["Óptimo (%)"],  marker_color="#00ff88"))
        fig_pesos.update_layout(
            **_LAYOUT_BASE,
            barmode="group", title="Pesos: Actual vs Óptimo",
            xaxis=dict(color="gray", gridcolor="#2e3148"),
            yaxis=dict(title="%", color="gray", gridcolor="#2e3148"),
        )
        st.plotly_chart(fig_pesos, use_container_width=True)

        # Tabla de pesos óptimos
        df_pesos_tabla = df_pesos[df_pesos["Óptimo (%)"] > 0.1].copy()
        df_pesos_tabla["Actual (%)"]  = df_pesos_tabla["Actual (%)"].map(lambda x: f"{x:.1f}%")
        df_pesos_tabla["Óptimo (%)"]  = df_pesos_tabla["Óptimo (%)"].map(lambda x: f"{x:.1f}%")
        df_pesos_tabla = df_pesos_tabla.sort_values("Óptimo (%)", ascending=False)
        st.dataframe(df_pesos_tabla, use_container_width=True, hide_index=True)

        st.divider()

        # Frontera eficiente
        if st.button("🔬 Generar Frontera Eficiente (5,000 portafolios)"):
            with st.spinner("Simulando 5,000 portafolios aleatorios..."):
                front_df = _frontera(periodo)

            w_act_arr = np.array([pw_act_opt.get(a, 0) for a in activos_opt])
            r_act_series = (ret_df_opt @ w_act_arr).dropna()

            fig_front = px.scatter(
                front_df,
                x="volatilidad", y="retorno",
                color="sharpe",
                color_continuous_scale="Plasma",
                opacity=0.4,
                labels={"volatilidad": "Volatilidad anual", "retorno": "Retorno anual",
                        "sharpe": "Sharpe"},
                title="Frontera Eficiente de Markowitz",
            )
            fig_front.update_traces(marker_size=4)

            # Punto actual
            fig_front.add_trace(go.Scatter(
                x=[volatilidad_anualizada(r_act_series)],
                y=[retorno_anualizado(r_act_series)],
                mode="markers+text",
                marker=dict(size=14, color="#00d4ff", symbol="circle",
                            line=dict(width=2, color="white")),
                text=["  Actual"], textposition="middle right",
                textfont=dict(color="#00d4ff", size=12),
                name="Portafolio actual",
            ))
            # Punto óptimo
            fig_front.add_trace(go.Scatter(
                x=[optimo["volatilidad_anual"]],
                y=[optimo["retorno_anual"]],
                mode="markers+text",
                marker=dict(size=18, color="#00ff88", symbol="star",
                            line=dict(width=2, color="white")),
                text=["  Óptimo"], textposition="middle right",
                textfont=dict(color="#00ff88", size=12),
                name="Portafolio óptimo",
            ))

            fig_front.update_layout(
                **_LAYOUT_BASE,
                xaxis=dict(tickformat=".1%", color="gray", gridcolor="#2e3148",
                           title="Volatilidad anual"),
                yaxis=dict(tickformat=".1%", color="gray", gridcolor="#2e3148",
                           title="Retorno anual"),
            )
            st.plotly_chart(fig_front, use_container_width=True)
            st.caption(
                "Cada punto es un portafolio aleatorio. "
                "El borde superior-izquierdo es la frontera eficiente: "
                "mayor retorno para el mismo riesgo."
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — Explorador de combinaciones
# ═════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("Explorador de Portafolios")
    st.caption("Combina activos libremente, ajusta los pesos y ve Sharpe, riesgo/retorno y Monte Carlo en tiempo real.")

    # ── Session state ─────────────────────────────────────────────────────────
    if "exp_custom" not in st.session_state:
        st.session_state.exp_custom = {}   # {ticker: nombre_display}

    # ── Catálogo desde el JSON ────────────────────────────────────────────────
    catalogo: dict[str, str] = {}
    for _a in portafolio.get("activos", []):
        if _a.get("ticker_yfinance"):
            catalogo[_a["ticker_yfinance"]] = _a["nombre_corto"]
    for _op in portafolio.get("opciones_inversion", []):
        if _op.get("ticker_yfinance"):
            catalogo[_op["ticker_yfinance"]] = _op["nombre_corto"]
    for _alt in portafolio.get("alternativas_sugeridas", []):
        if _alt.get("ticker_yfinance"):
            catalogo[_alt["ticker_yfinance"]] = _alt["nombre_corto"]

    # ── Layout ────────────────────────────────────────────────────────────────
    ctrl_col, res_col = st.columns([1, 2], gap="large")

    with ctrl_col:
        # 1. Selección desde catálogo
        st.subheader("1. Seleccionar activos")
        sel_catalogo = st.multiselect(
            "Tus opciones (TRii / alternativas)",
            options=list(catalogo.keys()),
            format_func=lambda tk: f"{catalogo[tk]}  ({tk})",
            placeholder="Escribe o selecciona...",
            key="exp_sel_catalogo",
        )

        # 2. Buscar ticker libre
        st.caption("O busca cualquier ticker del mundo:")
        inp_col, btn_col = st.columns([3, 1])
        with inp_col:
            nuevo_tk = st.text_input(
                "Ticker", key="exp_nuevo_tk",
                label_visibility="collapsed",
                placeholder="NVDA, MSFT, AMZN...",
            )
        with btn_col:
            if st.button("Añadir", key="exp_btn_add"):
                tk_upper = nuevo_tk.strip().upper()
                if tk_upper:
                    if tk_upper in catalogo or tk_upper in st.session_state.exp_custom:
                        st.warning(f"{tk_upper} ya está en la lista.")
                    else:
                        with st.spinner(f"Buscando {tk_upper}..."):
                            _serie_test = _descargar_ticker(tk_upper, "1y")
                        if _serie_test is not None:
                            st.session_state.exp_custom[tk_upper] = tk_upper
                            st.success(f"{tk_upper} añadido.")
                        else:
                            st.error(f"{tk_upper}: sin datos en Yahoo Finance.")

        # Mostrar personalizados con botón de quitar
        if st.session_state.exp_custom:
            st.caption("Personalizados añadidos:")
            _to_del = []
            for _tk in list(st.session_state.exp_custom):
                _c1, _c2 = st.columns([4, 1])
                _c1.markdown(f"`{_tk}`")
                if _c2.button("✕", key=f"exp_rm_{_tk}"):
                    _to_del.append(_tk)
            for _tk in _to_del:
                del st.session_state.exp_custom[_tk]
                st.rerun()

        # Todos los seleccionados
        todos_sel: dict[str, str] = {tk: catalogo[tk] for tk in sel_catalogo}
        todos_sel.update(st.session_state.exp_custom)

        st.divider()

        # 3. Período
        exp_periodo = st.selectbox(
            "Período histórico", ["6mo", "1y", "2y", "5y"],
            index=2, key="exp_periodo",
        )

        # 4. Pesos
        pesos: dict[str, int] = {}
        total_peso = 0

        if todos_sel:
            st.subheader("2. Pesos (%)")
            b1, b2 = st.columns(2)
            igualar_btn = b1.button("Igualar pesos", key="exp_igualar")
            limpiar_btn = b2.button("Limpiar todo",  key="exp_limpiar")

            if limpiar_btn:
                st.session_state.exp_custom = {}
                st.rerun()

            n_sel = len(todos_sel)
            peso_base = 100 // n_sel

            for i, (tk, nombre) in enumerate(todos_sel.items()):
                default_w = peso_base + (100 - peso_base * n_sel if i == 0 else 0)
                label = f"{nombre[:20]} ({tk})" if nombre != tk else tk
                pesos[tk] = st.slider(
                    label, 0, 100,
                    value=default_w if igualar_btn else default_w,
                    step=5, key=f"exp_w_{tk}",
                )
                total_peso += pesos[tk]

            # Indicador de suma
            if abs(total_peso - 100) < 1:
                st.success(f"Total: {total_peso}%  ✓")
            else:
                st.warning(f"Total: {total_peso}%  — ajusta a 100% para ver el combo")
        else:
            st.info("Selecciona al menos un activo arriba.")

    # ── Panel de resultados ───────────────────────────────────────────────────
    with res_col:
        if not todos_sel:
            st.markdown(
                """
                ### ¿Cómo usar el Explorador?
                1. **Selecciona activos** del catálogo (tus opciones TRii) o añade cualquier ticker.
                2. **Ajusta los pesos** (deben sumar 100%).
                3. Verás automáticamente: Sharpe, retorno, volatilidad, drawdown por activo
                   y métricas del portafolio combinado.
                4. Activa **Monte Carlo** para proyectar escenarios futuros.
                """
            )
        else:
            # Descargar series para cada ticker seleccionado
            series: dict[str, pd.Series] = {}
            sin_datos: list[str] = []
            for tk in todos_sel:
                with st.spinner(f"Cargando {tk}..."):
                    _s = _descargar_ticker(tk, exp_periodo)
                if _s is not None:
                    series[tk] = _s
                else:
                    sin_datos.append(tk)

            if sin_datos:
                st.warning(f"Sin datos en Yahoo Finance para: {', '.join(sin_datos)}")

            if not series:
                st.error("No se pudo cargar ningún activo. Verifica los tickers.")
            else:
                tickers_ok = list(series.keys())
                df_p = pd.DataFrame(series).dropna(how="all").ffill()
                df_r = np.log(df_p / df_p.shift(1)).dropna(how="all")

                # ── Tabla de métricas individuales ────────────────────────────
                filas_ind = []
                for tk in tickers_ok:
                    s   = df_r[tk].dropna()
                    ret = retorno_anualizado(s)
                    vol = volatilidad_anualizada(s)
                    sh  = sharpe_ratio(s)
                    dd  = float(((df_p[tk] / df_p[tk].cummax()) - 1).min())
                    filas_ind.append({
                        "Activo":      todos_sel.get(tk, tk)[:24],
                        "Ticker":      tk,
                        "Peso %":      pesos.get(tk, 0),
                        "Ret.Anual":   ret,
                        "Volatilidad": vol,
                        "Sharpe":      sh,
                        "Max DD":      dd,
                    })

                df_ind = pd.DataFrame(filas_ind)
                df_disp = df_ind.copy()
                df_disp["Ret.Anual"]   = df_disp["Ret.Anual"].map(lambda x: f"{x:+.1%}")
                df_disp["Volatilidad"] = df_disp["Volatilidad"].map(lambda x: f"{x:.1%}")
                df_disp["Sharpe"]      = df_disp["Sharpe"].map(lambda x: f"{x:.2f}")
                df_disp["Max DD"]      = df_disp["Max DD"].map(lambda x: f"{x:.1%}")
                st.dataframe(df_disp, use_container_width=True, hide_index=True)

                # ── Scatter riesgo / retorno ──────────────────────────────────
                PALETA = ["#00d4ff", "#ffaa00", "#aa88ff", "#ff8844",
                          "#00ff88", "#00a8cc", "#ff6666", "#88ffaa"]
                fig_sc = go.Figure()
                for i, row in df_ind.iterrows():
                    color_dot = PALETA[i % len(PALETA)]
                    fig_sc.add_trace(go.Scatter(
                        x=[row["Volatilidad"]], y=[row["Ret.Anual"]],
                        mode="markers+text",
                        marker=dict(size=18, color=color_dot,
                                    line=dict(width=1.5, color="white")),
                        text=[row["Ticker"]],
                        textposition="top center",
                        textfont=dict(color=color_dot, size=11),
                        name=row["Ticker"], showlegend=True,
                    ))

                # ── Portafolio combinado (si pesos ≈ 100%) ───────────────────
                pesos_arr = np.array([pesos.get(tk, 0) for tk in tickers_ok], dtype=float)
                combo_valido = abs(total_peso - 100) < 2 and total_peso > 0

                if combo_valido:
                    pesos_norm = pesos_arr / pesos_arr.sum()
                    r_port = (df_r[tickers_ok] @ pesos_norm).dropna()
                    ret_p  = retorno_anualizado(r_port)
                    vol_p  = volatilidad_anualizada(r_port)
                    sh_p   = sharpe_ratio(r_port)
                    dd_p   = float(((1 + r_port).cumprod()
                                    .pipe(lambda p: (p - p.cummax()) / p.cummax())).min())

                    # Punto estrella del combo en scatter
                    fig_sc.add_trace(go.Scatter(
                        x=[vol_p], y=[ret_p],
                        mode="markers+text",
                        marker=dict(size=26, color="#00ff88", symbol="star",
                                    line=dict(width=2, color="white")),
                        text=["COMBO"], textposition="top right",
                        textfont=dict(color="#00ff88", size=13),
                        name="Portafolio combinado", showlegend=True,
                    ))

                    # Métricas del combo
                    st.divider()
                    st.subheader("Portafolio combinado")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Sharpe", f"{sh_p:.2f}",
                              help="Sharpe > 1 = bueno, > 2 = excelente")
                    m2.metric("Retorno anual", f"{ret_p:+.1%}")
                    m3.metric("Volatilidad", f"{vol_p:.1%}")
                    m4.metric("Max Drawdown", f"{dd_p:.1%}")

                fig_sc.update_layout(
                    **_LAYOUT_BASE,
                    title="Riesgo vs Retorno por activo",
                    xaxis=dict(title="Volatilidad anual", tickformat=".0%",
                               color="gray", gridcolor="#2e3148"),
                    yaxis=dict(title="Retorno anual", tickformat=".0%",
                               color="gray", gridcolor="#2e3148"),
                    height=380,
                )
                st.plotly_chart(fig_sc, use_container_width=True)

                # ── Gráficas secundarias (pie + correlaciones) ────────────────
                if combo_valido and len(tickers_ok) > 1:
                    pie_col2, corr_col2 = st.columns(2)

                    with pie_col2:
                        fig_pie2 = go.Figure(go.Pie(
                            labels=tickers_ok,
                            values=[pesos.get(tk, 0) for tk in tickers_ok],
                            hole=0.45,
                            marker_colors=PALETA[:len(tickers_ok)],
                            textinfo="label+percent",
                        ))
                        fig_pie2.update_layout(
                            **_LAYOUT_BASE,
                            title="Distribución de pesos",
                            showlegend=False,
                            margin=dict(t=45, b=5, l=5, r=5),
                            height=300,
                        )
                        st.plotly_chart(fig_pie2, use_container_width=True)

                    with corr_col2:
                        corr_m = df_r[tickers_ok].corr()
                        fig_corr = px.imshow(
                            corr_m, text_auto=".2f",
                            color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1,
                            title="Correlaciones",
                        )
                        fig_corr.update_layout(
                            **_LAYOUT_BASE, height=300,
                            margin=dict(t=45, b=5, l=5, r=5),
                            coloraxis_showscale=False,
                            xaxis=dict(color="gray"),
                            yaxis=dict(color="gray"),
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                    st.caption(
                        "Correlación cerca de 0 (azul) = activos se mueven independiente = mejor diversificación. "
                        "Cerca de 1 (rojo) = se mueven igual."
                    )

                # ── Monte Carlo del combo ─────────────────────────────────────
                if combo_valido:
                    st.divider()
                    st.subheader("Monte Carlo — proyección de la combinación")
                    mc2_col1, mc2_col2, mc2_col3 = st.columns([2, 1, 1])
                    with mc2_col1:
                        cap_exp = st.number_input(
                            "Capital inicial (COP)", value=207_945,
                            step=50_000, format="%d", key="exp_cap",
                        )
                    with mc2_col2:
                        anos_exp = st.slider("Años", 1, 5, 2, key="exp_anos")
                    with mc2_col3:
                        nsim_exp = st.select_slider(
                            "Simulaciones", [500, 1000, 2000], value=500, key="exp_nsim",
                        )

                    if st.button("Simular", key="exp_mc_btn", type="primary"):
                        dias_exp = int(anos_exp * 252)
                        tray_exp = simular_montecarlo(
                            r_port, int(cap_exp), dias_exp, nsim_exp, semilla=42,
                        )
                        st_exp = estadisticas_montecarlo(tray_exp)

                        e1, e2, e3, e4 = st.columns(4)
                        e1.metric("P5 — pesimista",
                                  f"${st_exp['p05']:,.0f}",
                                  f"{st_exp['p05']/cap_exp - 1:+.1%}")
                        e2.metric("P50 — probable",
                                  f"${st_exp['p50']:,.0f}",
                                  f"{st_exp['p50']/cap_exp - 1:+.1%}")
                        e3.metric("P95 — optimista",
                                  f"${st_exp['p95']:,.0f}",
                                  f"{st_exp['p95']/cap_exp - 1:+.1%}")
                        e4.metric("Prob. ganancia", f"{st_exp['prob_ganancia']:.1%}")

                        st.plotly_chart(
                            _mc_chart(tray_exp, "Combinación", "#aa88ff", cap_exp),
                            use_container_width=True,
                        )
