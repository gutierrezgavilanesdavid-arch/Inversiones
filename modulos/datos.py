"""
Módulo de descarga de datos históricos usando Yahoo Finance.

yfinance descarga datos OHLCV (Open, High, Low, Close, Volume) de Yahoo Finance.
Usamos el precio de cierre ajustado (Adj Close) que ya incluye dividendos y splits.
"""

import yfinance as yf
import pandas as pd


def descargar_precios(tickers: list[str], periodo: str = "2y") -> pd.DataFrame:
    """
    Descarga precios de cierre ajustados para una lista de tickers.

    Parámetros:
        tickers : lista de símbolos, ej. ["AAPL", "SPY", "MSFT"]
        periodo : "1y", "2y", "5y", "10y", "max"

    Retorna:
        DataFrame con fechas como índice y tickers como columnas
    """
    datos = yf.download(tickers, period=periodo, auto_adjust=True, progress=False)

    # yfinance devuelve multi-nivel cuando son varios tickers
    if isinstance(datos.columns, pd.MultiIndex):
        precios = datos["Close"]
    else:
        precios = datos[["Close"]]
        precios.columns = tickers

    # Eliminar días sin datos (fines de semana, feriados)
    precios = precios.dropna(how="all")

    return precios


def calcular_retornos_diarios(precios: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte precios a retornos logarítmicos diarios.

    ¿Por qué logarítmicos?
    - Son aditivos: r_total = r_día1 + r_día2 + ... (útil para sumar períodos)
    - Son más simétricos estadísticamente
    - Fórmula: r_t = ln(P_t / P_{t-1})
    """
    return precios.apply(lambda col: col.dropna().pipe(
        lambda s: pd.Series(
            __import__("numpy").log(s / s.shift(1)),
            index=s.index
        )
    )).dropna()


def resumen_datos(precios: pd.DataFrame) -> dict:
    """Retorna metadatos básicos del dataset descargado."""
    return {
        "activos": list(precios.columns),
        "desde": precios.index[0].strftime("%Y-%m-%d"),
        "hasta": precios.index[-1].strftime("%Y-%m-%d"),
        "dias_de_datos": len(precios),
        "datos_faltantes": precios.isnull().sum().to_dict(),
    }
