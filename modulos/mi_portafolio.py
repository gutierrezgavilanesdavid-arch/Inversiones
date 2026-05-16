"""
Módulo del portafolio real del usuario.

Fuente de datos: Yahoo Finance via yfinance.
  - Para ETFs/crypto con ticker: datos históricos y precios actuales reales.
  - Para fondos locales colombianos sin ticker: retorno EA declarado
    + serie sintética de precios (GBM calibrado a sus parámetros).

Tipo de cambio: descargado en tiempo real de Yahoo Finance (COP=X → USD/COP).
Los precios de Yahoo Finance (USD) se convierten a COP para comparar
con los precios que muestra TRii.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
import yfinance as yf

from modulos.datos import descargar_precios, calcular_retornos_diarios
from modulos.metricas import (
    retorno_anualizado, volatilidad_anualizada, sharpe_ratio,
    maximo_drawdown, DIAS_HABILES_ANO,
)

console = Console()
ARCHIVO_PORTAFOLIO = Path("portafolio_real.json")


# ─── Carga ────────────────────────────────────────────────────────────────────

def cargar_portafolio() -> dict:
    with open(ARCHIVO_PORTAFOLIO, "r", encoding="utf-8") as f:
        return json.load(f)


def valor_total_cop(portafolio: dict) -> float:
    return sum(a["valor_cop"] for a in portafolio["activos"])


# ─── Tipo de cambio USD/COP ───────────────────────────────────────────────────

def obtener_usd_cop() -> float:
    """Descarga el tipo de cambio USD/COP actual desde Yahoo Finance."""
    try:
        datos = yf.download("COP=X", period="5d", auto_adjust=True, progress=False)
        tasa = float(datos["Close"].dropna().iloc[-1])
        return tasa
    except Exception:
        console.print("[yellow]No se pudo obtener tipo de cambio en vivo. Usando 4,100 COP/USD.[/yellow]")
        return 4100.0


# ─── Comparación de precios TRii vs Yahoo Finance ────────────────────────────

def verificar_precios(portafolio: dict):
    """
    Para cada activo/opción con ticker en Yahoo Finance:
      1. Descarga el precio actual en USD
      2. Convierte a COP usando el tipo de cambio real
      3. Compara con el precio declarado en TRii
      4. Actualiza el precio en memoria con el de Yahoo si hay diferencia > 5%
         o si el precio era None.
    """
    console.print("\n[dim]Verificando precios con Yahoo Finance...[/dim]")
    usd_cop = obtener_usd_cop()
    console.print(f"  [dim]Tipo de cambio: 1 USD = {usd_cop:,.0f} COP[/dim]\n")

    # Recolectar todos los tickers (opciones + alternativas)
    todos = portafolio.get("opciones_inversion", []) + portafolio.get("alternativas_sugeridas", [])
    tickers = [op["ticker_yfinance"] for op in todos if op.get("ticker_yfinance")]
    tickers = list(dict.fromkeys(tickers))  # eliminar duplicados manteniendo orden

    if not tickers:
        return usd_cop

    try:
        datos = yf.download(tickers, period="5d", auto_adjust=True, progress=False)
        if isinstance(datos.columns, pd.MultiIndex):
            ultimos = datos["Close"].dropna().iloc[-1]
        else:
            ultimos = datos["Close"].dropna().iloc[-1:]
            ultimos = pd.Series(ultimos.values, index=tickers[:1])
    except Exception as e:
        console.print(f"[red]Error descargando precios: {e}[/red]")
        return usd_cop

    tabla = Table(
        title="Precios: TRii (COP) vs Yahoo Finance (USD -> COP)",
        box=box.ROUNDED, border_style="cyan", show_lines=True,
    )
    tabla.add_column("Activo",         min_width=22)
    tabla.add_column("Ticker",         justify="center", min_width=7, style="dim")
    tabla.add_column("TRii (COP)",     justify="right",  min_width=13)
    tabla.add_column("Yahoo USD",      justify="right",  min_width=11)
    tabla.add_column("Yahoo (COP)",     justify="right",  min_width=13)
    tabla.add_column("Diferencia",     justify="center", min_width=11)
    tabla.add_column("Precio a usar",  justify="center", min_width=13)

    # Actualizar precios en el portafolio en memoria
    mapa_ticker_op = {op["ticker_yfinance"]: op for op in todos if op.get("ticker_yfinance")}

    for ticker, op in mapa_ticker_op.items():
        if ticker not in ultimos.index:
            tabla.add_row(op["nombre_corto"], ticker, "-", "-", "-", "-", "[red]sin datos[/red]")
            continue

        precio_usd = float(ultimos[ticker])
        precio_cop_yf = precio_usd * usd_cop
        precio_trii = op.get("precio_cop")

        if precio_trii is None:
            # Sin precio declarado → usar Yahoo Finance
            op["precio_cop"] = round(precio_cop_yf)
            op["precio_fuente"] = "yahoo"
            tabla.add_row(
                op["nombre_corto"], ticker, "[dim]sin dato[/dim]",
                f"${precio_usd:,.2f}",
                Text(f"${precio_cop_yf:,.0f}", style="cyan"),
                "-",
                Text("Yahoo asignado", style="cyan"),
            )
        else:
            dif = (precio_cop_yf - precio_trii) / precio_trii
            if abs(dif) > 0.05:
                color_dif = "red" if abs(dif) > 0.15 else "yellow"
                precio_usar = precio_cop_yf
                fuente_txt = Text("Yahoo OK", style="cyan")
            else:
                color_dif = "green"
                precio_usar = precio_trii
                fuente_txt = Text("TRii OK", style="green")

            op["precio_cop"] = round(precio_usar)
            op["precio_fuente"] = "yahoo" if abs(dif) > 0.05 else "trii"

            tabla.add_row(
                op["nombre_corto"], ticker,
                f"${precio_trii:>10,.0f}",
                f"${precio_usd:,.2f}",
                f"${precio_cop_yf:>10,.0f}",
                Text(f"{dif:+.1%}", style=color_dif),
                fuente_txt,
            )

    console.print(tabla)
    console.print(f"  [dim]Si diferencia > 5%: se usa el precio de Yahoo Finance.[/dim]")
    return usd_cop


# ─── Serie sintética para fondos sin ticker ──────────────────────────────────

def generar_serie_sintetica(rentabilidad_ea: float, volatilidad_anual: float,
                             dias: int = 504, semilla: int = 0) -> pd.Series:
    rng = np.random.default_rng(semilla)
    mu_d = (1 + rentabilidad_ea) ** (1 / DIAS_HABILES_ANO) - 1
    sig_d = volatilidad_anual / np.sqrt(DIAS_HABILES_ANO)
    shocks = rng.normal(mu_d, sig_d, dias)
    precios = 100 * np.cumprod(1 + shocks)
    fechas = pd.bdate_range(end=pd.Timestamp.today(), periods=dias)
    return pd.Series(precios, index=fechas)


# ─── Retornos completos del portafolio ───────────────────────────────────────

def obtener_retornos_portafolio_real(portafolio: dict, periodo: str = "2y") -> pd.DataFrame:
    tickers_reales = [a["ticker_yfinance"] for a in portafolio["activos"] if a.get("ticker_yfinance")]
    ids_sinteticos = [a for a in portafolio["activos"] if not a.get("ticker_yfinance")]
    retornos_dict = {}

    if tickers_reales:
        precios_raw = descargar_precios(tickers_reales, periodo)
        retornos_raw = calcular_retornos_diarios(precios_raw)
        mapa = {a["ticker_yfinance"]: a["id"] for a in portafolio["activos"] if a.get("ticker_yfinance")}
        retornos_raw.columns = [mapa.get(c, c) for c in retornos_raw.columns]
        for col in retornos_raw.columns:
            retornos_dict[col] = retornos_raw[col]

    for activo in ids_sinteticos:
        dias_p = {"1y": 252, "2y": 504, "5y": 1260}.get(periodo, 504)
        serie = generar_serie_sintetica(
            activo["rentabilidad_ea"], activo["volatilidad_estimada_anual"],
            dias=dias_p, semilla=hash(activo["id"]) % 100,
        )
        retornos_dict[activo["id"]] = np.log(serie / serie.shift(1)).dropna()

    return pd.DataFrame(retornos_dict).dropna(how="all")


# ─── Display portafolio ───────────────────────────────────────────────────────

def mostrar_portafolio_actual(portafolio: dict):
    total = valor_total_cop(portafolio)
    disponible = portafolio["disponible_cop"]

    tabla = Table(
        title=f"[bold]{portafolio['nombre']}[/bold]",
        box=box.ROUNDED, border_style="cyan", show_lines=True,
    )
    tabla.add_column("Activo",      style="bold white", min_width=22)
    tabla.add_column("Tipo",        justify="center",   min_width=10, style="dim")
    tabla.add_column("Valor (COP)", justify="right",    min_width=14)
    tabla.add_column("% Total",     justify="center",   min_width=8)
    tabla.add_column("Info",        justify="center",   min_width=14)

    for a in portafolio["activos"]:
        peso = a["valor_cop"] / total
        if a["tipo"] == "fondo_local":
            info = Text(f"EA {a['rentabilidad_ea']:.2%}", style="green")
        elif a.get("ticker_yfinance"):
            info = Text(f"~{a['ticker_yfinance']}", style="dim cyan")
        else:
            info = Text("sin ticker", style="dim")
        tabla.add_row(
            a["nombre_corto"], a["tipo"].replace("_", " "),
            f"${a['valor_cop']:>12,.0f}", f"{peso:.1%}", info,
        )

    tabla.add_section()
    tabla.add_row("[bold]TOTAL INVERTIDO[/bold]", "",
                  f"[bold]${total:>12,.0f}[/bold]", "[bold]100%[/bold]", "")
    tabla.add_row("[yellow]Disponible[/yellow]", "",
                  f"[yellow]${disponible:>12,.0f}[/yellow]", "",
                  f"[dim]comisión ${portafolio['comision_por_transaccion_cop']:,}[/dim]")
    console.print()
    console.print(tabla)
    console.print(f"\n  [dim]Capital total (invertido + disponible): ${total + disponible:,.0f} COP[/dim]")


# ─── Métricas del portafolio real ────────────────────────────────────────────

def analizar_portafolio_real(portafolio: dict, periodo: str = "2y"):
    console.print(f"\n[dim]Obteniendo datos históricos del portafolio ({periodo})...[/dim]")
    retornos = obtener_retornos_portafolio_real(portafolio, periodo)
    total = valor_total_cop(portafolio)
    pesos = {a["id"]: a["valor_cop"] / total for a in portafolio["activos"]}

    tabla = Table(
        title="Metricas historicas - Portafolio Real",
        box=box.ROUNDED, border_style="cyan", show_lines=True,
    )
    tabla.add_column("Activo",      min_width=22, style="bold white")
    tabla.add_column("Peso",        justify="center", min_width=7)
    tabla.add_column("Ret. Anual",  justify="center", min_width=11)
    tabla.add_column("Volatilidad", justify="center", min_width=11)
    tabla.add_column("Sharpe",      justify="center", min_width=8)
    tabla.add_column("Max DD",      justify="center", min_width=9)
    tabla.add_column("Fuente",      justify="center", min_width=10, style="dim")

    def c_sh(v):
        if v >= 1.5: return "bold green"
        if v >= 0.8: return "green"
        if v >= 0:   return "yellow"
        return "red"

    info_map = {a["id"]: a for a in portafolio["activos"]}

    for id_activo in retornos.columns:
        s  = retornos[id_activo].dropna()
        r  = retorno_anualizado(s)
        v  = volatilidad_anualizada(s)
        sh = sharpe_ratio(s)
        dd = (1 + s).cumprod().pipe(lambda p: (p - p.cummax()) / p.cummax()).min()
        peso = pesos.get(id_activo, 0)
        info = info_map.get(id_activo, {})
        fuente = "yfinance" if info.get("ticker_yfinance") else "sintético*"
        tabla.add_row(
            info.get("nombre_corto", id_activo), f"{peso:.1%}",
            Text(f"{r:+.1%}", style="green" if r >= 0 else "red"),
            f"{v:.1%}",
            Text(f"{sh:.2f}", style=c_sh(sh)),
            Text(f"{dd:.1%}", style="red"),
            fuente,
        )

    w = np.array([pesos.get(c, 0) for c in retornos.columns])
    r_port = retornos @ w
    r_clean = r_port.dropna()
    tabla.add_section()
    tabla.add_row(
        "[bold cyan]PORTAFOLIO TOTAL[/bold cyan]", "[bold]100%[/bold]",
        Text(f"{retorno_anualizado(r_clean):+.1%}", style="bold cyan"),
        f"{volatilidad_anualizada(r_clean):.1%}",
        Text(f"{sharpe_ratio(r_clean):.2f}", style=c_sh(sharpe_ratio(r_clean))),
        Text(f"{((1+r_clean).cumprod().pipe(lambda p:(p-p.cummax())/p.cummax()).min()):.1%}", style="red"),
        "",
    )
    console.print()
    console.print(tabla)
    console.print("  [dim]*Sintético: generado con el EA declarado + volatilidad estimada del fondo.[/dim]")
    return retornos, pesos


# ─── Opciones de inversión ────────────────────────────────────────────────────

def analizar_opciones_inversion(portafolio: dict) -> list:
    disponible = portafolio["disponible_cop"]
    comision   = portafolio["comision_por_transaccion_cop"]
    neto       = disponible - comision

    tabla = Table(
        title="¿Qué comprar con los $207,945 disponibles?",
        box=box.ROUNDED, border_style="yellow", show_lines=True,
    )
    tabla.add_column("Activo",           min_width=22)
    tabla.add_column("Precio/ud.",       justify="right",  min_width=12)
    tabla.add_column("Fuente precio",    justify="center", min_width=12, style="dim")
    tabla.add_column("Unidades",         justify="center", min_width=8)
    tabla.add_column("Total c/comisión", justify="right",  min_width=14)
    tabla.add_column("Com. efectiva",    justify="center", min_width=13)
    tabla.add_column("Sobra",            justify="right",  min_width=10)

    viables = []
    for op in portafolio["opciones_inversion"]:
        precio = op.get("precio_cop")
        fuente = op.get("precio_fuente", "trii")

        if precio is None:
            tabla.add_row(op["nombre_corto"], "sin precio", "-", "-", "-", "-", "-")
            continue

        unidades = int(neto // precio)
        total_check = unidades * precio + comision
        if total_check > disponible and unidades > 0:
            unidades -= 1

        if unidades < 1:
            tabla.add_row(
                op["nombre_corto"], f"${precio:,.0f}",
                fuente, "[red]0[/red]", "-", "-", "[red]sin presupuesto[/red]",
            )
            continue

        total = unidades * precio + comision
        com_efec = comision / (unidades * precio)
        sobra = disponible - total
        color = "green" if com_efec < 0.09 else "yellow"

        tabla.add_row(
            op["nombre_corto"],
            f"${precio:>10,.0f}",
            Text(fuente, style="cyan" if fuente == "yahoo" else "dim"),
            str(unidades),
            f"${total:>12,.0f}",
            Text(f"{com_efec:.1%}", style=color),
            f"${sobra:,.0f}",
        )
        viables.append({**op, "unidades": unidades, "total": total,
                        "com_efec": com_efec, "sobra": sobra})

    console.print()
    console.print(tabla)
    console.print(f"\n  [dim]Neto disponible (sin comisión): ${neto:,.0f} COP[/dim]")
    return viables


# ─── Comparación histórica (opciones + alternativas) ─────────────────────────

def comparar_opciones_yfinance(portafolio: dict, periodo: str = "2y"):
    todos = portafolio.get("opciones_inversion", []) + portafolio.get("alternativas_sugeridas", [])
    tickers = list({op["ticker_yfinance"]: op for op in todos if op.get("ticker_yfinance")}.keys())
    mapa_nombre = {op["ticker_yfinance"]: op["nombre_corto"]
                   for op in todos if op.get("ticker_yfinance")}
    ids_viables = {op["ticker_yfinance"]
                   for op in portafolio.get("opciones_inversion", [])
                   if op.get("ticker_yfinance") and op.get("precio_cop")}

    if not tickers:
        return None, None

    console.print(f"\n[dim]Descargando historial ({periodo}) para {len(tickers)} activos...[/dim]")
    try:
        precios = descargar_precios(tickers, periodo)
        retornos = calcular_retornos_diarios(precios)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return None, None

    tabla = Table(
        title="Rendimiento historico - Opciones TRii + Alternativas sugeridas (en USD)",
        box=box.ROUNDED, border_style="green", show_lines=True,
    )
    tabla.add_column("Activo",      min_width=22)
    tabla.add_column("Ticker",      justify="center", min_width=7, style="dim")
    tabla.add_column("Ret. Anual",  justify="center", min_width=11)
    tabla.add_column("Volatilidad", justify="center", min_width=11)
    tabla.add_column("Sharpe",      justify="center", min_width=8)
    tabla.add_column("Max DD",      justify="center", min_width=9)
    tabla.add_column("Categoría",   justify="center", min_width=13)

    def c_sh(v):
        if v >= 1.5: return "bold green"
        if v >= 0.8: return "green"
        if v >= 0:   return "yellow"
        return "red"

    filas = []
    for ticker in precios.columns:
        s  = retornos[ticker].dropna()
        r  = retorno_anualizado(s)
        v  = volatilidad_anualizada(s)
        sh = sharpe_ratio(s)
        dd = maximo_drawdown(precios[ticker].dropna())
        nombre = mapa_nombre.get(ticker, ticker)
        cat = "> opcion TRii" if ticker in ids_viables else "* alternativa"
        filas.append((sh, ticker, nombre, r, v, dd, cat))

    for _, ticker, nombre, r, v, dd, cat in sorted(filas, key=lambda x: -x[0]):
        color_cat = "yellow" if "opción" in cat else "magenta"
        tabla.add_row(
            nombre, ticker,
            Text(f"{r:+.1%}", style="green" if r >= 0 else "red"),
            f"{v:.1%}",
            Text(f"{sh:.2f}", style=c_sh(sh)),
            Text(f"{dd:.1%}", style="red"),
            Text(cat, style=color_cat),
        )

    console.print()
    console.print(tabla)
    console.print("  [dim]Ordenado por Sharpe Ratio (mayor = mejor)  |  Retornos en USD.[/dim]")
    return precios, retornos


# ─── Recomendaciones ─────────────────────────────────────────────────────────

def mostrar_recomendaciones(portafolio: dict):
    total = valor_total_cop(portafolio)
    pesos = {a["id"]: a["valor_cop"] / total for a in portafolio["activos"]}
    conc_colombia = pesos.get("FONDO_MAYOR", 0) + pesos.get("FONDO_MODERADO", 0)

    console.print()
    console.print(Panel(
        f"[bold]Diagnóstico del portafolio actual[/bold]\n\n"
        f"  Colombia (fondos AVC) : [{'red' if conc_colombia > 0.7 else 'yellow'}]{conc_colombia:.1%}[/]"
        f"{'  ! alta concentracion en un solo pais' if conc_colombia > 0.7 else ''}\n"
        f"  Mercados emergentes   : {pesos.get('EIMICO', 0):.1%}  (EIMICO)\n"
        f"  Crypto                : {pesos.get('BTC', 0):.1%}  (Bitcoin)\n"
        f"  EE.UU. / Desarrollados: [red]0.0%[/red]  ! sin exposicion",
        title="[bold yellow]Análisis de diversificación[/bold yellow]",
        border_style="yellow", padding=(1, 3),
    ))

    console.print(Panel(
        "[bold]Para el capital disponible ($207,945 COP):[/bold]\n\n"
        "  [green]1. IUITCO / XLK - S&P500 Tech[/green]  ->  1 unidad ~ $185,440-199,940\n"
        "     Sharpe ~0.84  Retorno anual ~26% USD  Llena el hueco de mercados desarrollados\n\n"
        "  [yellow]2. IUESCO / XLE - S&P500 Energy[/yellow]  ->  4 unidades ~ $202,500\n"
        "     Sharpe ~0.47  Sector diferente al tech  Menor correlacion con EIMICO\n\n"
        "  [dim]3. RBOTCO / BOTZ - Robotics[/dim]  ->  2 unidades ~ $168,540\n"
        "     Sharpe ~0.30  Tematica interesante  Desempeno inferior a XLK en 2 anos\n\n"
        "[bold]Recomendacion:[/bold] [green]IUITCO[/green] - mayor Sharpe, un solo pago de comision,\n"
        "  y cubre el mayor vacio del portafolio (0% en mercados desarrollados).",
        title="[bold green]Recomendacion - Inversion inmediata[/bold green]",
        border_style="green", padding=(1, 3),
    ))

    console.print(Panel(
        "[bold]Redistribucion a largo plazo (6-18 meses):[/bold]\n\n"
        "  El portafolio tiene [red]80% en Colombia[/red] - riesgo cambiario y politico concentrado.\n\n"
        "  [cyan]Objetivo sugerido cuando los fondos AVC venzan o puedas reubicar:[/cyan]\n\n"
        "  | 30-35%  Fondos AVC  (mantener por EA competitivo: 14-16%)\n"
        "  | 25-30%  EE.UU.  (QQQ / XLK / VTI)\n"
        "  | 15-20%  Emergentes  (EIMICO - ya lo tienes)\n"
        "  |  5-10%  LatAm / Colombia sectorial  (GEB, PFGRUPOARG)\n"
        "  |   3-5%  Crypto  (BTC - ya lo tienes, considera mantener)\n\n"
        "  Alternativas a explorar:\n"
        "  [magenta]QQQ[/magenta] - Nasdaq-100, mejor Sharpe historico que XLK puro\n"
        "  [magenta]SCHD[/magenta] - Dividendos crecientes USA, baja volatilidad\n"
        "  [magenta]GLD[/magenta] - Oro, cobertura contra devaluacion del COP",
        title="[bold magenta]Estrategia de largo plazo[/bold magenta]",
        border_style="magenta", padding=(1, 3),
    ))


# ─── Gráfica composición ─────────────────────────────────────────────────────

def graficar_portafolio_cop(portafolio: dict):
    activos = portafolio["activos"]
    total   = valor_total_cop(portafolio)
    nombres = [a["nombre_corto"] for a in activos]
    valores = [a["valor_cop"] for a in activos]
    colores = ["#00d4ff", "#00a8cc", "#ffaa00", "#00ff88", "#aa88ff", "#ff8844"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), facecolor="#0f1117")
    fig.suptitle(f"Composicion - {portafolio['nombre']}",
                 color="white", fontsize=13, fontweight="bold")

    ax1.set_facecolor("#0f1117")
    wedges, texts, autotexts = ax1.pie(
        valores, labels=nombres, autopct="%1.1f%%",
        colors=colores[:len(nombres)], startangle=90,
        wedgeprops={"edgecolor": "#0f1117", "linewidth": 2}, pctdistance=0.82,
    )
    for t in texts:   t.set_color("gray");  t.set_fontsize(8)
    for at in autotexts: at.set_color("white"); at.set_fontsize(9); at.set_fontweight("bold")
    ax1.add_patch(plt.Circle((0, 0), 0.55, fc="#0f1117"))
    ax1.text(0, 0.08, f"${total/1e6:.2f}M", ha="center", va="center",
             color="white", fontsize=14, fontweight="bold")
    ax1.text(0, -0.12, "COP", ha="center", va="center", color="gray", fontsize=10)

    ax2.set_facecolor("#1a1d27")
    ax2.tick_params(colors="gray")
    ax2.spines[:].set_color("#2e3148")
    y_pos = list(range(len(nombres)))
    bars = ax2.barh(y_pos, valores, color=colores[:len(nombres)], alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(nombres, color="gray", fontsize=9)
    ax2.set_xlabel("Valor COP", color="gray")
    ax2.set_title("Valor por activo", color="white", fontsize=11)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    for label in ax2.get_xticklabels(): label.set_color("gray")
    for bar, val in zip(bars, valores):
        ax2.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"${val:,.0f}", va="center", color="white", fontsize=8)

    plt.tight_layout()
    plt.show()


# ─── Portafolio sugerido y retornos para MC/Optimizer ────────────────────────

PORTAFOLIO_SUGERIDO = {
    "nombre": "Portafolio Sugerido",
    "pesos": {
        "FONDO_MAYOR": 0.32,
        "FONDO_MODERADO": 0.00,
        "BTC": 0.04,
        "EIMICO": 0.18,
        "IUITCO": 0.28,
        "IUESCO": 0.10,
        "GEB": 0.08,
    },
    "descripcion": "30% AVC Mayor, 0% AVC Moderado (reubicar), 4% BTC, 18% EM, 28% S&P Tech, 10% Energy, 8% GEB",
}


def construir_retornos_para_analisis(portafolio: dict, periodo: str = "2y") -> tuple[pd.DataFrame, dict, dict]:
    """
    Descarga retornos históricos de Yahoo Finance para todos los activos
    del portafolio actual Y del portafolio sugerido.

    Retorna:
        retornos_df  : DataFrame con todos los activos combinados
        pesos_actual : pesos del portafolio actual
        pesos_sugeri : pesos del portafolio sugerido
    """
    # Activos del portafolio actual con pesos
    total = valor_total_cop(portafolio)
    pesos_actual = {a["id"]: a["valor_cop"] / total for a in portafolio["activos"]}

    # Pesos del sugerido (solo los que existen en retornos)
    pesos_sugeri = PORTAFOLIO_SUGERIDO["pesos"]

    # Mapa id → ticker (activos del portafolio real)
    mapa_ticker = {a["id"]: a.get("ticker_yfinance") for a in portafolio["activos"]}

    # Agregar activos del sugerido que no están en el portafolio actual
    ticker_extra = {
        "IUITCO": "XLK",
        "IUESCO": "XLE",
        "GEB":    None,   # sin ticker Yahoo
    }
    for id_act, ticker in ticker_extra.items():
        if id_act not in mapa_ticker:
            mapa_ticker[id_act] = ticker

    # Descargar los que tienen ticker
    tickers_reales = list({v for v in mapa_ticker.values() if v})
    retornos_dict: dict[str, pd.Series] = {}

    if tickers_reales:
        precios_raw = descargar_precios(tickers_reales, periodo)
        retornos_raw = calcular_retornos_diarios(precios_raw)
        # Invertir mapa para ticker → id
        inv_mapa = {v: k for k, v in mapa_ticker.items() if v}
        for ticker_col in retornos_raw.columns:
            id_act = inv_mapa.get(ticker_col, ticker_col)
            retornos_dict[id_act] = retornos_raw[ticker_col]

    # Fondos sin ticker → serie sintética
    fondos_sinteticos = {
        a["id"]: (a["rentabilidad_ea"], a["volatilidad_estimada_anual"])
        for a in portafolio["activos"]
        if not a.get("ticker_yfinance")
    }
    dias_p = {"1y": 252, "2y": 504, "5y": 1260}.get(periodo, 504)
    for id_act, (ea, vol) in fondos_sinteticos.items():
        serie = generar_serie_sintetica(ea, vol, dias=dias_p, semilla=hash(id_act) % 100)
        retornos_dict[id_act] = np.log(serie / serie.shift(1)).dropna()

    # GEB sin ticker → no incluir en optimización
    retornos_dict.pop("GEB", None)
    pesos_sugeri_filtrado = {k: v for k, v in pesos_sugeri.items() if k in retornos_dict}
    # Renormalizar pesos sugeridos a los activos disponibles
    suma = sum(pesos_sugeri_filtrado.values())
    if suma > 0:
        pesos_sugeri_filtrado = {k: v / suma for k, v in pesos_sugeri_filtrado.items()}
    # Renormalizar pesos actuales
    pesos_actual_filtrado = {k: v for k, v in pesos_actual.items() if k in retornos_dict}
    suma2 = sum(pesos_actual_filtrado.values())
    if suma2 > 0:
        pesos_actual_filtrado = {k: v / suma2 for k, v in pesos_actual_filtrado.items()}

    retornos_df = pd.DataFrame(retornos_dict).dropna(how="all")
    return retornos_df, pesos_actual_filtrado, pesos_sugeri_filtrado


def comparar_mc_actual_vs_sugerido(
    retornos: pd.DataFrame,
    pesos_actual: dict,
    pesos_sugeri: dict,
    capital: float = 1_422_444,
    anos: float = 2.0,
):
    """
    Corre Monte Carlo para portafolio actual Y sugerido, muestra comparación.
    """
    from modulos.montecarlo import simular_montecarlo, estadisticas_montecarlo
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    dias = int(anos * 252)
    activos = list(retornos.columns)

    w_act = np.array([pesos_actual.get(a, 0) for a in activos])
    w_sug = np.array([pesos_sugeri.get(a, 0) for a in activos])

    r_actual  = (retornos @ w_act).dropna()
    r_sugeri  = (retornos @ w_sug).dropna()

    tray_act  = simular_montecarlo(r_actual, capital, dias, 1000, semilla=42)
    tray_sug  = simular_montecarlo(r_sugeri, capital, dias, 1000, semilla=42)
    st_act    = estadisticas_montecarlo(tray_act)
    st_sug    = estadisticas_montecarlo(tray_sug)

    # Tabla comparativa en terminal
    console.print()
    tabla = Table(title=f"Monte Carlo - Actual vs Sugerido ({anos:.0f} ano(s), ${capital:,.0f} COP)",
                  box=box.ROUNDED, border_style="cyan", show_lines=True)
    tabla.add_column("Escenario",          min_width=22, style="bold")
    tabla.add_column("Pesimista (P5)",     justify="right", min_width=16)
    tabla.add_column("Probable (P50)",     justify="right", min_width=16)
    tabla.add_column("Optimista (P95)",    justify="right", min_width=16)
    tabla.add_column("Prob. ganancia",     justify="center", min_width=14)
    tabla.add_column("Prob. perder >20%",  justify="center", min_width=16)

    for nombre, st in [("Portafolio actual", st_act), ("Portafolio sugerido", st_sug)]:
        color = "cyan" if "actual" in nombre else "green"
        tabla.add_row(
            f"[{color}]{nombre}[/{color}]",
            Text(f"${st['p05']:>12,.0f}  ({st['p05']/capital-1:+.1%})", style="red"),
            Text(f"${st['p50']:>12,.0f}  ({st['p50']/capital-1:+.1%})", style=color),
            Text(f"${st['p95']:>12,.0f}  ({st['p95']/capital-1:+.1%})", style="green"),
            Text(f"{st['prob_ganancia']:.1%}", style="green"),
            Text(f"{st['prob_perdida_20']:.1%}", style="red"),
        )
    console.print(tabla)

    # Gráfica comparativa
    fig = plt.figure(figsize=(14, 6), facecolor="#0f1117")
    fig.suptitle(f"Monte Carlo: Portafolio Actual vs Sugerido  ({dias} dias, ${capital/1e6:.2f}M COP)",
                 color="white", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)
    eje = np.arange(dias + 1)

    for idx, (nombre, tray, st, color) in enumerate([
        ("Actual",   tray_act, st_act, "#00d4ff"),
        ("Sugerido", tray_sug, st_sug, "#00ff88"),
    ]):
        ax = fig.add_subplot(gs[idx])
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="gray")
        ax.spines[:].set_color("#2e3148")

        p05 = np.percentile(tray, 5,  axis=1)
        p50 = np.percentile(tray, 50, axis=1)
        p95 = np.percentile(tray, 95, axis=1)

        n_show = min(150, tray.shape[1])
        for i in np.random.choice(tray.shape[1], n_show, replace=False):
            ax.plot(eje, tray[:, i], alpha=0.03, linewidth=0.5,
                    color="#ff4444" if tray[-1, i] < capital else color)

        ax.fill_between(eje, p05, p95, alpha=0.15, color=color)
        ax.fill_between(eje, np.percentile(tray, 25, axis=1),
                        np.percentile(tray, 75, axis=1), alpha=0.25, color=color)
        ax.plot(eje, p50, color=color, linewidth=2,
                label=f"Mediana: ${p50[-1]:,.0f}")
        ax.plot(eje, p05, color="#ff6666", linewidth=1, linestyle="--",
                label=f"P5: ${p05[-1]:,.0f}")
        ax.plot(eje, p95, color="#66ff88", linewidth=1, linestyle="--",
                label=f"P95: ${p95[-1]:,.0f}")
        ax.axhline(capital, color="#888888", linewidth=0.8, linestyle=":")

        w_titulo = w_act if idx == 0 else w_sug
        ax.set_title(f"Portafolio {nombre}  |  Sharpe {sharpe_ratio((retornos @ w_titulo).dropna()):.2f}",
                     color="white", fontsize=10)
        ax.set_xlabel("Dias", color="gray", fontsize=9)
        ax.set_ylabel("Valor COP", color="gray", fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
        ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color("gray")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return st_act, st_sug
