"""
Optimizador de portafolio basado en la teoría moderna de Markowitz.

Encuentra los pesos que maximizan el Sharpe Ratio usando scipy.optimize.
También genera la Frontera Eficiente mediante simulación de portafolios aleatorios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize

from modulos.metricas import DIAS_HABILES_ANO, TASA_LIBRE_RIESGO_ANUAL


def _sharpe_negativo(pesos: np.ndarray, retornos_diarios: pd.DataFrame) -> float:
    """
    Función objetivo para el optimizador: retorna el Sharpe negativo.
    scipy.optimize siempre minimiza, así que minimizar (-Sharpe) = maximizar Sharpe.
    """
    r = retornos_diarios @ pesos
    retorno_anual = r.mean() * DIAS_HABILES_ANO
    volatilidad_anual = r.std() * np.sqrt(DIAS_HABILES_ANO)
    if volatilidad_anual == 0:
        return 0.0
    sharpe = (retorno_anual - TASA_LIBRE_RIESGO_ANUAL) / volatilidad_anual
    return -sharpe


def optimizar_sharpe(retornos_diarios: pd.DataFrame) -> dict:
    """
    Encuentra los pesos que maximizan el Sharpe Ratio del portafolio.

    Restricciones:
      - Todos los pesos >= 0 (no se permite venta en corto)
      - La suma de pesos = 1 (100% del capital invertido)

    Retorna:
        diccionario con pesos óptimos y métricas del portafolio resultante
    """
    n = len(retornos_diarios.columns)
    tickers = list(retornos_diarios.columns)

    # Punto de partida: pesos iguales
    pesos_iniciales = np.ones(n) / n

    # Restricción: pesos suman 1
    restricciones = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    # Límites: cada peso entre 0% y 100%
    limites = [(0.0, 1.0)] * n

    resultado = minimize(
        fun=_sharpe_negativo,
        x0=pesos_iniciales,
        args=(retornos_diarios,),
        method="SLSQP",
        bounds=limites,
        constraints=restricciones,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    pesos_optimos = resultado.x
    r_port = retornos_diarios @ pesos_optimos
    retorno_anual = r_port.mean() * DIAS_HABILES_ANO
    volatilidad_anual = r_port.std() * np.sqrt(DIAS_HABILES_ANO)
    sharpe = (retorno_anual - TASA_LIBRE_RIESGO_ANUAL) / volatilidad_anual

    return {
        "pesos": {t: round(float(w), 4) for t, w in zip(tickers, pesos_optimos)},
        "retorno_anual": retorno_anual,
        "volatilidad_anual": volatilidad_anual,
        "sharpe": sharpe,
        "exito": resultado.success,
    }


def frontera_eficiente(retornos_diarios: pd.DataFrame, n_portafolios: int = 5000) -> pd.DataFrame:
    """
    Genera la frontera eficiente simulando N portafolios con pesos aleatorios.

    Cada portafolio es un punto en el espacio (riesgo, retorno).
    Los puntos en el borde superior izquierdo forman la frontera eficiente.

    Retorna DataFrame con columnas: retorno, volatilidad, sharpe, y un peso por ticker.
    """
    tickers = list(retornos_diarios.columns)
    n = len(tickers)
    rng = np.random.default_rng(42)

    resultados = []
    for _ in range(n_portafolios):
        # Pesos aleatorios que suman 1 (distribución Dirichlet)
        pesos = rng.dirichlet(np.ones(n))
        r_port = retornos_diarios @ pesos
        ret = r_port.mean() * DIAS_HABILES_ANO
        vol = r_port.std() * np.sqrt(DIAS_HABILES_ANO)
        sharpe = (ret - TASA_LIBRE_RIESGO_ANUAL) / vol if vol > 0 else 0
        fila = {"retorno": ret, "volatilidad": vol, "sharpe": sharpe}
        for t, w in zip(tickers, pesos):
            fila[t] = w
        resultados.append(fila)

    return pd.DataFrame(resultados)


def graficar_frontera(
    frontera: pd.DataFrame,
    portafolio_actual: dict,
    portafolio_optimo: dict,
    retornos_diarios: pd.DataFrame,
):
    """
    Visualiza la frontera eficiente con el portafolio actual y el óptimo marcados.
    """
    tickers = list(retornos_diarios.columns)

    fig = plt.figure(figsize=(14, 7), facecolor="#0f1117")
    fig.suptitle("Optimización de Portafolio — Frontera Eficiente (Markowitz)",
                 color="white", fontsize=13, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for ax in [ax1, ax2]:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="gray")
        ax.spines[:].set_color("#2e3148")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("gray")

    # ── Frontera eficiente coloreada por Sharpe ───────────────────────────────
    sc = ax1.scatter(
        frontera["volatilidad"] * 100,
        frontera["retorno"] * 100,
        c=frontera["sharpe"],
        cmap="plasma",
        alpha=0.4,
        s=8,
        edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label("Sharpe Ratio", color="gray", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="gray")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="gray")

    # Portafolio actual
    r_act = retornos_diarios @ np.array(list(portafolio_actual.values()))
    vol_act = r_act.std() * np.sqrt(DIAS_HABILES_ANO) * 100
    ret_act = r_act.mean() * DIAS_HABILES_ANO * 100
    ax1.scatter(vol_act, ret_act, color="#00d4ff", s=120, zorder=5,
                marker="o", label=f"Portafolio actual  (Sharpe {portafolio_actual.get('_sharpe', 0):.2f})")
    ax1.annotate("  Actual", (vol_act, ret_act), color="#00d4ff", fontsize=9)

    # Portafolio óptimo
    vol_opt = portafolio_optimo["volatilidad_anual"] * 100
    ret_opt = portafolio_optimo["retorno_anual"] * 100
    ax1.scatter(vol_opt, ret_opt, color="#00ff88", s=180, zorder=5,
                marker="*", label=f"Portafolio óptimo  (Sharpe {portafolio_optimo['sharpe']:.2f})")
    ax1.annotate("  Óptimo", (vol_opt, ret_opt), color="#00ff88", fontsize=9)

    ax1.set_xlabel("Volatilidad anual (%)", color="gray", fontsize=10)
    ax1.set_ylabel("Retorno anual (%)", color="gray", fontsize=10)
    ax1.set_title("Cada punto = un portafolio posible", color="white", fontsize=11)
    ax1.legend(facecolor="#1a1d27", labelcolor="white", fontsize=9)

    # ── Comparación de pesos: actual vs óptimo ────────────────────────────────
    pesos_actuales = [portafolio_actual.get(t, 0) for t in tickers]
    pesos_optimos  = [portafolio_optimo["pesos"].get(t, 0) for t in tickers]
    x = np.arange(len(tickers))
    ancho = 0.35

    ax2.bar(x - ancho/2, [p * 100 for p in pesos_actuales], ancho,
            label="Actual", color="#00d4ff", alpha=0.8)
    ax2.bar(x + ancho/2, [p * 100 for p in pesos_optimos], ancho,
            label="Óptimo", color="#00ff88", alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers, color="gray")
    ax2.set_ylabel("Peso (%)", color="gray", fontsize=9)
    ax2.set_title("Comparación de pesos", color="white", fontsize=11)
    ax2.legend(facecolor="#1a1d27", labelcolor="white", fontsize=9)
    ax2.set_ylim(0, 100)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
