# Asistente de Inversión — David Gutiérrez

## ¿Qué es este proyecto?

Herramienta personal de análisis de portafolio de inversiones construida en Python.
Analiza el portafolio real del usuario (TRii / AVC / Binance), descarga datos de Yahoo Finance,
calcula métricas financieras (Sharpe, drawdown, volatilidad), simula Monte Carlo y optimiza
con teoría de Markowitz. Tiene dashboard web interactivo en Streamlit.

## Cómo ejecutar

```bash
# Dashboard web (principal)
python -m streamlit run dashboard.py

# CLI interactivo
python main.py

# Con encoding correcto en Windows terminal
$env:PYTHONIOENCODING = "utf-8"
python main.py
```

## Portafolio real del usuario

Archivo de configuración: `portafolio_real.json`

- **AVC Mayor Riesgo** — $763,582 COP — fondo local, EA 16.41%, vol est. 12%
- **AVC Moderado** — $373,862 COP — fondo local, EA 14.43%, vol est. 7%
- **Bitcoin (BTC)** — $80,000 COP — Binance, ticker BTC-USD
- **EIMICO / IEMG** — $205,000 COP — ETF Emerging Markets, proxy IEMG
- **Disponible** — $207,945 COP — comisión TRii: $14,500 por operación

Plataformas: TRii (CDRs en COP), AVC (fondos colombianos), Binance (crypto)

## Opciones de inversión en TRii (con precios COP)

| ID | Nombre corto | Precio/ud | Ticker Yahoo |
|----|-------------|-----------|-------------|
| RBOTCO | iShares Robotics | $77,020 | BOTZ |
| ECLCO | ENGIE Chile | $7,660 | ECL.SN |
| IUITCO | S&P500 Tech | $185,440 | XLK |
| IUESCO | S&P500 Energy | $47,000 | XLE |
| ICHNCO | MSCI China | $25,500 | MCHI |
| GEB | Grupo Energía Bogotá | $2,950 | null |
| 4BRZCO | MSCI Brazil | $203,100 | EWZ |

Alternativas sugeridas (sin precio TRii): QQQ, VTI, SCHD, GLD, NVDA

## Hallazgos clave ya calculados

- **ECLCO (ENGIE Chile / ECL.SN)**: Sharpe 1.47, retorno +39.4%, vol 23.8% — mejor opción disponible
- **IUITCO (XLK)**: Sharpe 0.83, retorno +25.9% — sólido, cubre hueco de 0% en mercados desarrollados
- **IUESCO (XLE)**: Sharpe 0.47 — la peor opción del grupo actual
- **Combo 1 IUITCO + 1 ECLCO + 1 comisión = $207,600** — cabe en presupuesto por $345
- **ISA.CL (ISA Colombia)**: sin datos en Yahoo Finance — fue adquirida por Ecopetrol en 2021 y delisted
- **Ameris DVA Silicon Fund**: proxy = SMH/SOXX (semiconductores), Sharpe ~1.0, vol ~38%, retorno ~45%
- **EWZ (4BRZCO Brasil)**: delisted o sin datos en Yahoo Finance; además fuera de presupuesto ($217,600)
- USD/COP en tiempo real via ticker `COP=X` en yfinance, fallback 4,100

## Diagnóstico del portafolio actual

- 80% concentrado en Colombia (fondos AVC) — riesgo país/cambiario alto
- 0% exposición a mercados desarrollados (EE.UU./Europa)
- Portafolio total: Sharpe ~0.8, bien diversificado para lo que tiene
- Recomendación inmediata: IUITCO o ECLCO con los $207,945 disponibles

## Estructura de archivos

```
Inversion/
├── main.py                    # CLI interactivo (menú p/i/1-8/q)
├── dashboard.py               # Dashboard Streamlit (5 tabs)
├── portafolio_real.json       # Configuración del portafolio del usuario
├── modulos/
│   ├── datos.py               # Descarga precios Yahoo Finance, retornos log
│   ├── metricas.py            # Sharpe, volatilidad, drawdown, retorno anualizado
│   ├── graficas.py            # Gráficas matplotlib 4-panel (dark theme)
│   ├── montecarlo.py          # Simulación GBM, estadísticas, gráficas
│   ├── optimizador.py         # Markowitz, frontera eficiente
│   └── mi_portafolio.py       # Módulo principal del portafolio real
└── CLAUDE.md                  # Este archivo
```

## Módulos clave — funciones principales

### `modulos/mi_portafolio.py`
- `cargar_portafolio()` — lee portafolio_real.json
- `obtener_usd_cop()` — tipo de cambio de COP=X, fallback 4100
- `verificar_precios(port)` — compara precios TRii vs Yahoo Finance
- `generar_serie_sintetica(ea, vol, dias)` — GBM para fondos AVC sin ticker
- `obtener_retornos_portafolio_real(port, periodo)` — retornos históricos + sintéticos
- `analizar_portafolio_real(port, periodo)` — tabla Rich con métricas
- `analizar_opciones_inversion(port)` — qué comprar con el disponible
- `comparar_opciones_yfinance(port, periodo)` — ranking por Sharpe de opciones + alternativas
- `mostrar_recomendaciones(port)` — 3 paneles Rich: diagnóstico, inmediato, largo plazo
- `construir_retornos_para_analisis(port, periodo)` — retornos para MC + Optimizer
- `comparar_mc_actual_vs_sugerido(ret, pw_act, pw_sug, capital, anos)` — MC lado a lado
- `PORTAFOLIO_SUGERIDO` — pesos objetivo: 32% AVC Mayor, 4% BTC, 18% EIMICO, 28% IUITCO, 10% IUESCO, 8% GEB

### `dashboard.py` — 5 tabs Streamlit
1. **📊 Mi Portafolio** — donut composición, barras, tabla métricas históricas
2. **🔍 Opciones de Inversión** — ranking Sharpe, qué comprar con $207,945
3. **🎲 Monte Carlo** — actual vs sugerido, bandas percentiles, tabla P5/P50/P95
4. **⚙️ Optimizador Markowitz** — pesos óptimos vs actuales, frontera eficiente interactiva
5. **🧪 Explorador** — combina activos libres + búsqueda por ticker, scatter riesgo/retorno, correlaciones, MC rápido

## Conceptos financieros usados

- **Retornos log**: `r = ln(P_t / P_{t-1})` — aditivos en el tiempo
- **Volatilidad anual**: `σ_diaria × √252`
- **Sharpe Ratio**: `(R_port - Rf) / σ` — Rf = 4.5% (US Treasury). >1 = bueno, >2 = excelente
- **Max Drawdown**: caída máxima desde un pico histórico
- **GBM (Monte Carlo)**: `P(t+1) = P(t) × exp((μ - σ²/2)dt + σ√dt × ε)`
- **CDRs**: TRii vende fracciones en COP de ETFs internacionales — los precios difieren de Yahoo Finance pero los retornos son equivalentes
- **Fondos AVC**: sin ticker en Yahoo — se usa serie sintética GBM calibrada a su EA declarado

## Git y entorno

- Repo: https://github.com/gutierrezgavilanesdavid-arch/Inversiones.git
- Python: `C:\Users\DAVID-GTZ\AppData\Local\Python\pythoncore-3.14-64\python.exe`
- Git (GitHub Desktop bundled): `C:\Users\DAVID-GTZ\AppData\Local\GitHubDesktop\app-3.5.8\resources\app\git\cmd\git.exe`
- En PowerShell: usar siempre `python -m pip` (no `pip` directo)
- `$env:PYTHONIOENCODING = "utf-8"` antes de ejecutar scripts con caracteres especiales

## Dependencias instaladas

```
yfinance, pandas, numpy, scipy, matplotlib, rich, streamlit, plotly
```

## Notas importantes

- Los precios de TRii son CDRs (fracciones COP de ETFs) — la diferencia vs Yahoo Finance es normal y esperada
- Los fondos AVC (FONDO_MAYOR, FONDO_MODERADO) no tienen ticker — sus retornos son sintéticos, marcados como "sintético*"
- ISA.CL no tiene datos disponibles en Yahoo Finance (delisted desde adquisición por Ecopetrol 2021)
- EWZ (4BRZCO Brasil) reporta como "possibly delisted" en yfinance — ignorar en análisis
- El usuario prefiere operaciones grandes (menos comisiones) sobre muchas pequeñas
