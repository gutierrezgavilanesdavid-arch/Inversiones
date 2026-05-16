"""
Simulación Monte Carlo para proyección del portafolio.

Usa movimiento browniano geométrico (GBM) — el mismo modelo base de Black-Scholes.
Genera N caminos aleatorios de precios futuros a partir de los parámetros históricos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter


def simular_montecarlo(
    retornos_diarios: pd.Series,
    capital_inicial: float = 10_000,
    dias: int = 252,
    n_simulaciones: int = 1000,
    semilla: int = 42,
) -> np.ndarray:
    """
    Simula N trayectorias futuras del portafolio usando GBM.

    Fórmula diaria:
        P(t+1) = P(t) × exp((μ - σ²/2)×dt + σ×√dt×ε)
    donde ε ~ N(0,1)

    Parámetros:
        retornos_diarios : retornos log históricos del portafolio
        capital_inicial  : cuánto dinero se invierte hoy
        dias             : cuántos días hacia el futuro simular (252 = 1 año)
        n_simulaciones   : cuántos caminos distintos generar
        semilla          : para reproducibilidad (mismo resultado cada vez)

    Retorna:
        matriz (dias+1) × n_simulaciones con el valor del portafolio cada día
    """
    rng = np.random.default_rng(semilla)

    mu = retornos_diarios.mean()
    sigma = retornos_diarios.std()
    dt = 1  # un día

    # Generamos todos los shocks aleatorios de una vez (más eficiente)
    shocks = rng.standard_normal((dias, n_simulaciones))

    # Retorno diario de cada simulación: deriva + ruido aleatorio
    retornos_sim = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks

    # Acumulamos desde capital_inicial
    factor_acumulado = np.exp(np.cumsum(retornos_sim, axis=0))
    trayectorias = capital_inicial * np.vstack([
        np.ones(n_simulaciones),
        factor_acumulado,
    ])

    return trayectorias  # shape: (dias+1, n_simulaciones)


def estadisticas_montecarlo(trayectorias: np.ndarray) -> dict:
    """
    Calcula percentiles y estadísticas clave del conjunto de simulaciones.

    Percentil 5  → en el 5% peor de los casos terminas aquí o menos
    Percentil 50 → la mediana (mitad de escenarios están encima, mitad abajo)
    Percentil 95 → en el 5% mejor de los casos terminas aquí o más
    """
    valores_finales = trayectorias[-1, :]
    capital_inicial = trayectorias[0, 0]

    return {
        "capital_inicial": capital_inicial,
        "p05": np.percentile(valores_finales, 5),
        "p25": np.percentile(valores_finales, 25),
        "p50": np.percentile(valores_finales, 50),
        "p75": np.percentile(valores_finales, 75),
        "p95": np.percentile(valores_finales, 95),
        "media": valores_finales.mean(),
        "prob_ganancia": (valores_finales > capital_inicial).mean(),
        "prob_perdida_20": (valores_finales < capital_inicial * 0.8).mean(),
        "n_simulaciones": trayectorias.shape[1],
        "dias": trayectorias.shape[0] - 1,
    }


def graficar_montecarlo(
    trayectorias: np.ndarray,
    stats: dict,
    nombre_portafolio: str = "Mi Portafolio",
):
    """
    Visualiza las simulaciones Monte Carlo con percentiles y distribución final.
    """
    dias = stats["dias"]
    eje_tiempo = np.arange(dias + 1)
    capital = stats["capital_inicial"]

    fig = plt.figure(figsize=(14, 8), facecolor="#0f1117")
    fig.suptitle(
        f"Monte Carlo: {nombre_portafolio}  ({stats['n_simulaciones']:,} simulaciones, {dias} días)",
        color="white", fontsize=13, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for ax in [ax1, ax2]:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="gray")
        ax.spines[:].set_color("#2e3148")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("gray")

    # ── Panel izquierdo: trayectorias + percentiles ───────────────────────────
    # Muestra solo 200 trayectorias para no saturar la vista
    n_mostrar = min(200, trayectorias.shape[1])
    indices = np.random.choice(trayectorias.shape[1], n_mostrar, replace=False)
    for i in indices:
        color = "#ff4444" if trayectorias[-1, i] < capital else "#00d4ff"
        ax1.plot(eje_tiempo, trayectorias[:, i], alpha=0.04, linewidth=0.6, color=color)

    # Percentiles como bandas
    p05 = np.percentile(trayectorias, 5, axis=1)
    p25 = np.percentile(trayectorias, 25, axis=1)
    p50 = np.percentile(trayectorias, 50, axis=1)
    p75 = np.percentile(trayectorias, 75, axis=1)
    p95 = np.percentile(trayectorias, 95, axis=1)

    ax1.fill_between(eje_tiempo, p05, p95, alpha=0.15, color="#00d4ff", label="Rango 5%-95%")
    ax1.fill_between(eje_tiempo, p25, p75, alpha=0.25, color="#00d4ff", label="Rango 25%-75%")
    ax1.plot(eje_tiempo, p50, color="#00ff88", linewidth=2, label=f"Mediana: ${p50[-1]:,.0f}")
    ax1.plot(eje_tiempo, p05, color="#ff6666", linewidth=1, linestyle="--",
             label=f"Peor 5%: ${p05[-1]:,.0f}")
    ax1.plot(eje_tiempo, p95, color="#66ff88", linewidth=1, linestyle="--",
             label=f"Mejor 5%: ${p95[-1]:,.0f}")
    ax1.axhline(capital, color="#888888", linewidth=0.8, linestyle=":", label=f"Capital inicial: ${capital:,.0f}")

    ax1.set_title("Trayectorias simuladas", color="white", fontsize=11)
    ax1.set_xlabel("Días desde hoy", color="gray", fontsize=9)
    ax1.set_ylabel("Valor del portafolio ($)", color="gray", fontsize=9)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8, loc="upper left")

    # ── Panel derecho: distribución de valores finales ────────────────────────
    valores_finales = trayectorias[-1, :]
    ax2.hist(valores_finales, bins=60, orientation="horizontal",
             color="#00d4ff", alpha=0.6, edgecolor="none")

    # Líneas de percentiles
    for p, label, color in [
        (stats["p05"],  "P5",     "#ff4444"),
        (stats["p25"],  "P25",    "#ffaa00"),
        (stats["p50"],  "Mediana","#00ff88"),
        (stats["p75"],  "P75",    "#ffaa00"),
        (stats["p95"],  "P95",    "#44ff44"),
    ]:
        ax2.axhline(p, color=color, linewidth=1.2, linestyle="--")
        ax2.text(ax2.get_xlim()[1] if ax2.get_xlim()[1] > 0 else 50,
                 p, f" {label}: ${p:,.0f}", color=color, fontsize=7.5, va="center")

    ax2.axhline(capital, color="#888888", linewidth=0.8, linestyle=":")
    ax2.set_title("Distribución final", color="white", fontsize=11)
    ax2.set_xlabel("Frecuencia", color="gray", fontsize=9)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Texto resumen debajo
    texto = (
        f"Prob. de ganancia:    {stats['prob_ganancia']:.1%}\n"
        f"Prob. de perder >20%: {stats['prob_perdida_20']:.1%}"
    )
    fig.text(0.5, 0.01, texto, ha="center", color="#aaaaaa", fontsize=9)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.show()
