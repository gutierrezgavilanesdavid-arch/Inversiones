"""
Módulo de visualización del portafolio.

Genera 3 gráficas en una sola ventana:
  1. Evolución del valor del portafolio vs benchmark (SPY)
  2. Drawdown histórico
  3. Distribución de retornos diarios (histograma)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter


def _porcentaje(x, _):
    return f"{x:.0%}"


def calcular_valor_portafolio(retornos_diarios: pd.DataFrame, pesos: dict) -> pd.Series:
    """
    Convierte retornos diarios en una curva de valor normalizada a 1.0.
    Si empiezas con $1, esta curva muestra cuánto tienes cada día.
    """
    activos = list(pesos.keys())
    w = np.array([pesos[a] for a in activos])
    retornos_subset = retornos_diarios[activos].dropna()
    retornos_port = retornos_subset @ w
    # Convertimos retornos log a retornos simples y acumulamos
    valor = (1 + retornos_port).cumprod()
    valor.iloc[0] = 1.0
    return valor


def calcular_drawdown(valor: pd.Series) -> pd.Series:
    """
    Calcula la caída porcentual desde el pico más reciente.
    Drawdown = (valor_actual - pico_historico) / pico_historico
    """
    pico = valor.cummax()
    return (valor - pico) / pico


def graficar_portafolio(
    retornos_diarios: pd.DataFrame,
    pesos: dict,
    nombre_portafolio: str = "Mi Portafolio",
):
    """
    Genera el dashboard visual completo en una ventana de matplotlib.
    """
    activos = list(pesos.keys())

    # ── Calcular curvas ──────────────────────────────────────────────────────
    valor_port = calcular_valor_portafolio(retornos_diarios, pesos)
    drawdown_port = calcular_drawdown(valor_port)

    retornos_port = retornos_diarios[activos].dropna() @ np.array(list(pesos.values()))

    # Benchmark: SPY (si está en el portafolio lo usamos, si no lo calculamos aparte)
    benchmark_ticker = "SPY"
    tiene_benchmark = benchmark_ticker in retornos_diarios.columns
    if tiene_benchmark:
        valor_bench = (1 + retornos_diarios[benchmark_ticker].dropna()).cumprod()
        valor_bench = valor_bench.reindex(valor_port.index).ffill()

    # ── Layout de la figura ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10), facecolor="#0f1117")
    fig.suptitle(
        f"Análisis de Portafolio: {nombre_portafolio}",
        color="white", fontsize=14, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])   # Evolución — fila completa
    ax2 = fig.add_subplot(gs[1, :])   # Drawdown  — fila completa
    ax3 = fig.add_subplot(gs[2, 0])   # Histograma de retornos
    ax4 = fig.add_subplot(gs[2, 1])   # Retorno por activo (barras)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="gray")
        ax.spines[:].set_color("#2e3148")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("gray")

    # ── Gráfica 1: Evolución del portafolio ─────────────────────────────────
    ax1.plot(valor_port.index, valor_port.values, color="#00d4ff", linewidth=1.8,
             label=nombre_portafolio)
    if tiene_benchmark:
        ax1.plot(valor_bench.index, valor_bench.values, color="#666688", linewidth=1.2,
                 linestyle="--", label="SPY (benchmark)")

    retorno_total = valor_port.iloc[-1] - 1
    ax1.set_title(f"Evolución del valor  |  Retorno total: {retorno_total:+.1%}",
                  color="white", fontsize=11)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax1.legend(facecolor="#1a1d27", labelcolor="white", fontsize=9)
    ax1.set_ylabel("Valor relativo (base 1.0)", color="gray", fontsize=9)

    # Sombrear zona positiva/negativa respecto al inicio
    ax1.axhline(1.0, color="#444466", linewidth=0.8, linestyle=":")
    ax1.fill_between(valor_port.index, valor_port.values, 1.0,
                     where=(valor_port.values >= 1.0), alpha=0.15, color="#00ff88")
    ax1.fill_between(valor_port.index, valor_port.values, 1.0,
                     where=(valor_port.values < 1.0), alpha=0.15, color="#ff4444")

    # ── Gráfica 2: Drawdown ──────────────────────────────────────────────────
    ax2.fill_between(drawdown_port.index, drawdown_port.values, 0,
                     alpha=0.7, color="#ff4444")
    ax2.plot(drawdown_port.index, drawdown_port.values, color="#ff6666", linewidth=0.8)
    max_dd = drawdown_port.min()
    ax2.set_title(f"Drawdown histórico  |  Máx. caída: {max_dd:.1%}", color="white", fontsize=11)
    ax2.yaxis.set_major_formatter(FuncFormatter(_porcentaje))
    ax2.set_ylabel("Caída desde el pico", color="gray", fontsize=9)
    ax2.axhline(0, color="#444466", linewidth=0.8)

    # ── Gráfica 3: Histograma de retornos diarios ────────────────────────────
    retornos_pct = retornos_port * 100
    ax3.hist(retornos_pct, bins=60, color="#00d4ff", alpha=0.7, edgecolor="none")
    ax3.axvline(0, color="white", linewidth=0.8, linestyle="--")
    ax3.axvline(retornos_pct.mean(), color="#00ff88", linewidth=1.2,
                linestyle="--", label=f"Media: {retornos_pct.mean():.2f}%")
    media = retornos_pct.mean()
    std = retornos_pct.std()
    ax3.axvline(media - 2*std, color="#ffaa00", linewidth=0.8, linestyle=":",
                label=f"±2σ: {media-2*std:.1f}% / {media+2*std:.1f}%")
    ax3.axvline(media + 2*std, color="#ffaa00", linewidth=0.8, linestyle=":")
    ax3.set_title("Distribución de retornos diarios", color="white", fontsize=11)
    ax3.set_xlabel("Retorno diario (%)", color="gray", fontsize=9)
    ax3.set_ylabel("Frecuencia", color="gray", fontsize=9)
    ax3.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)

    # ── Gráfica 4: Retorno anual por activo ──────────────────────────────────
    from modulos.metricas import retorno_anualizado, DIAS_HABILES_ANO
    retornos_activos = {t: retorno_anualizado(retornos_diarios[t].dropna()) for t in activos}
    nombres = list(retornos_activos.keys())
    valores = [retornos_activos[t] * 100 for t in nombres]
    colores = ["#00ff88" if v >= 0 else "#ff4444" for v in valores]

    bars = ax4.bar(nombres, valores, color=colores, alpha=0.8, width=0.5)
    ax4.axhline(0, color="#444466", linewidth=0.8)
    ax4.set_title("Retorno anual por activo", color="white", fontsize=11)
    ax4.set_ylabel("Retorno anual (%)", color="gray", fontsize=9)
    for bar, val in zip(bars, valores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:+.1f}%", ha="center", va="bottom", color="white", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
