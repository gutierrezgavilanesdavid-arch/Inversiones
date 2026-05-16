"""
Módulo de métricas financieras y estadísticas de portafolio.

Conceptos implementados:
  - Retorno anualizado
  - Volatilidad anualizada (riesgo)
  - Sharpe Ratio
  - Correlación entre activos
  - Máximo Drawdown
"""

import numpy as np
import pandas as pd

DIAS_HABILES_ANO = 252
TASA_LIBRE_RIESGO_ANUAL = 0.045  # ~4.5% bonos del Tesoro USA 2025


def retorno_anualizado(retornos_diarios: pd.Series) -> float:
    """
    Convierte retornos diarios logarítmicos a retorno anual.

    Fórmula: R_anual = media_diaria × 252
    Interpretación: Si R_anual = 0.12 → el activo sube ~12% al año en promedio.
    """
    return retornos_diarios.mean() * DIAS_HABILES_ANO


def volatilidad_anualizada(retornos_diarios: pd.Series) -> float:
    """
    Mide el riesgo del activo como desviación estándar de retornos.

    Fórmula: σ_anual = σ_diaria × √252
    Interpretación: Si σ = 0.20 → los retornos varían ±20% al año (1 desviación estándar).
    """
    return retornos_diarios.std() * np.sqrt(DIAS_HABILES_ANO)


def sharpe_ratio(retornos_diarios: pd.Series) -> float:
    """
    Mide retorno ajustado por riesgo.

    Fórmula: Sharpe = (R_portafolio - R_libre_riesgo) / σ_portafolio
    Escala: >1 bueno, >2 excelente, <0 peor que no hacer nada.
    """
    rf_diaria = TASA_LIBRE_RIESGO_ANUAL / DIAS_HABILES_ANO
    exceso = retornos_diarios.mean() - rf_diaria
    if retornos_diarios.std() == 0:
        return 0.0
    return (exceso / retornos_diarios.std()) * np.sqrt(DIAS_HABILES_ANO)


def maximo_drawdown(precios: pd.Series) -> float:
    """
    Máxima caída desde un pico histórico hasta el valle siguiente.

    Ejemplo: Si el portafolio subió a $150 y luego cayó a $90 → drawdown = -40%
    Es la métrica que más duele emocionalmente: cuánto pudiste haber perdido
    si compraste en el peor momento.
    """
    pico_acumulado = precios.cummax()
    drawdown = (precios - pico_acumulado) / pico_acumulado
    return drawdown.min()


def retornos_portafolio(retornos_diarios: pd.DataFrame, pesos: dict) -> pd.Series:
    """
    Calcula el retorno diario del portafolio combinado.

    Parámetro pesos: diccionario {ticker: peso}, ej. {"AAPL": 0.4, "SPY": 0.6}
    Los pesos deben sumar 1.0.

    Fórmula: r_portafolio = Σ (peso_i × retorno_i)
    """
    activos = list(pesos.keys())
    w = np.array([pesos[a] for a in activos])

    if abs(w.sum() - 1.0) > 0.001:
        raise ValueError(f"Los pesos deben sumar 1.0. Suman: {w.sum():.4f}")

    retornos_subset = retornos_diarios[activos].dropna()
    return retornos_subset @ w


def analizar_portafolio(retornos_diarios: pd.DataFrame, pesos: dict, precios: pd.DataFrame) -> dict:
    """
    Genera un análisis completo del portafolio.
    Retorna un diccionario con todas las métricas.
    """
    r_port = retornos_portafolio(retornos_diarios, pesos)
    precios_port = (1 + retornos_diarios[list(pesos.keys())].fillna(0)).cumprod() @ np.array(list(pesos.values()))

    metricas_individuales = {}
    for ticker in pesos:
        s = retornos_diarios[ticker].dropna()
        metricas_individuales[ticker] = {
            "retorno_anual": retorno_anualizado(s),
            "volatilidad_anual": volatilidad_anualizada(s),
            "sharpe": sharpe_ratio(s),
            "max_drawdown": maximo_drawdown(precios[ticker].dropna()),
            "peso": pesos[ticker],
        }

    return {
        "portafolio": {
            "retorno_anual": retorno_anualizado(r_port),
            "volatilidad_anual": volatilidad_anualizada(r_port),
            "sharpe": sharpe_ratio(r_port),
            "max_drawdown": maximo_drawdown(precios_port),
        },
        "individuales": metricas_individuales,
        "correlaciones": retornos_diarios[list(pesos.keys())].corr().to_dict(),
    }
