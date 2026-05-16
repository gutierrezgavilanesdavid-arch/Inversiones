"""
Asistente de Inversión Interactivo
Ejecutar: python main.py
"""

import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, FloatPrompt
from rich import box
from rich.text import Text

# Agregamos la carpeta raíz al path para importar módulos
sys.path.insert(0, ".")

from modulos.datos import descargar_precios, calcular_retornos_diarios, resumen_datos
from modulos.metricas import analizar_portafolio
from modulos.graficas import graficar_portafolio
from modulos.montecarlo import simular_montecarlo, estadisticas_montecarlo, graficar_montecarlo
from modulos.optimizador import optimizar_sharpe, frontera_eficiente, graficar_frontera
from modulos.mi_portafolio import (
    cargar_portafolio, mostrar_portafolio_actual,
    analizar_portafolio_real, analizar_opciones_inversion,
    comparar_opciones_yfinance, mostrar_recomendaciones,
    graficar_portafolio_cop, verificar_precios,
    construir_retornos_para_analisis, comparar_mc_actual_vs_sugerido,
)

console = Console()


# ─── Portafolio por defecto para aprender ────────────────────────────────────
PORTAFOLIO_EJEMPLO = {
    "tickers": ["AAPL", "MSFT", "SPY", "QQQ"],
    "pesos":   {"AAPL": 0.25, "MSFT": 0.25, "SPY": 0.30, "QQQ": 0.20},
    "periodo": "2y",
}


# ─── Funciones de visualización ──────────────────────────────────────────────

def mostrar_bienvenida():
    console.print(Panel.fit(
        "[bold cyan]Asistente de Inversión[/bold cyan]\n"
        "[dim]Aprende mientras analizas tu portafolio[/dim]",
        border_style="cyan"
    ))


def mostrar_resumen_datos(resumen: dict):
    console.print(f"\n[bold]Datos descargados:[/bold]")
    console.print(f"  Activos  : {', '.join(resumen['activos'])}")
    console.print(f"  Período  : {resumen['desde']} → {resumen['hasta']}")
    console.print(f"  Días     : {resumen['dias_de_datos']} días hábiles")


def color_sharpe(valor: float) -> str:
    if valor >= 2:   return "bold green"
    if valor >= 1:   return "green"
    if valor >= 0:   return "yellow"
    return "red"


def color_retorno(valor: float) -> str:
    return "green" if valor >= 0 else "red"


def mostrar_tabla_activos(analisis: dict):
    tabla = Table(
        title="Análisis por Activo",
        box=box.ROUNDED,
        border_style="cyan",
        show_lines=True,
    )

    tabla.add_column("Activo",        style="bold white", justify="center")
    tabla.add_column("Peso",          justify="center")
    tabla.add_column("Retorno Anual", justify="center")
    tabla.add_column("Volatilidad",   justify="center")
    tabla.add_column("Sharpe Ratio",  justify="center")
    tabla.add_column("Max Drawdown",  justify="center")

    for ticker, m in analisis["individuales"].items():
        r  = m["retorno_anual"]
        v  = m["volatilidad_anual"]
        sh = m["sharpe"]
        dd = m["max_drawdown"]

        tabla.add_row(
            ticker,
            f"{m['peso']:.0%}",
            Text(f"{r:+.1%}", style=color_retorno(r)),
            f"{v:.1%}",
            Text(f"{sh:.2f}", style=color_sharpe(sh)),
            Text(f"{dd:.1%}", style="red"),
        )

    console.print()
    console.print(tabla)


def mostrar_resumen_portafolio(analisis: dict):
    p = analisis["portafolio"]
    r  = p["retorno_anual"]
    v  = p["volatilidad_anual"]
    sh = p["sharpe"]
    dd = p["max_drawdown"]

    panel_texto = (
        f"[bold]Retorno anual   :[/bold] [{color_retorno(r)}]{r:+.2%}[/]\n"
        f"[bold]Volatilidad     :[/bold] {v:.2%}\n"
        f"[bold]Sharpe Ratio    :[/bold] [{color_sharpe(sh)}]{sh:.2f}[/]\n"
        f"[bold]Máx. Drawdown   :[/bold] [red]{dd:.2%}[/]"
    )

    console.print(Panel(panel_texto, title="[bold cyan]Portafolio Total[/bold cyan]",
                         border_style="cyan", padding=(1, 4)))


def mostrar_correlaciones(analisis: dict):
    tickers = list(analisis["individuales"].keys())
    corr = analisis["correlaciones"]

    tabla = Table(title="Correlaciones entre Activos", box=box.SIMPLE, border_style="dim")
    tabla.add_column("", style="bold")

    for t in tickers:
        tabla.add_column(t, justify="center")

    for t1 in tickers:
        fila = [t1]
        for t2 in tickers:
            val = corr[t1][t2]
            if t1 == t2:
                fila.append("[dim]  1.00[/dim]")
            elif val > 0.8:
                fila.append(f"[yellow]{val:.2f}[/yellow]")
            elif val > 0.5:
                fila.append(f"{val:.2f}")
            else:
                fila.append(f"[green]{val:.2f}[/green]")
        tabla.add_row(*fila)

    console.print()
    console.print(tabla)
    console.print("[dim]  Verde = baja correlación (mejor diversificación)[/dim]")
    console.print("[dim]  Amarillo = alta correlación (activos se mueven igual)[/dim]")


# ─── Menú interactivo ────────────────────────────────────────────────────────

def menu_principal() -> str:
    console.print("\n[bold]¿Qué quieres hacer?[/bold]")
    console.print("  [bold yellow]── Mi portafolio real ──[/bold yellow]")
    console.print("  [yellow]p[/yellow] Ver mi portafolio (TRii/AVC/Binance)")
    console.print("  [yellow]i[/yellow] Analizar opciones de inversión ($207,945 disponibles)")
    console.print("  [bold cyan]── Portafolio de análisis ──[/bold cyan]")
    console.print("  [cyan]1[/cyan] Ver análisis completo del portafolio")
    console.print("  [cyan]2[/cyan] Cambiar los pesos")
    console.print("  [cyan]3[/cyan] Agregar o quitar activos")
    console.print("  [cyan]4[/cyan] Ver correlaciones")
    console.print("  [cyan]5[/cyan] Ver gráficas (evolución, drawdown, retornos)")
    console.print("  [cyan]6[/cyan] Simulación Monte Carlo")
    console.print("  [cyan]7[/cyan] Optimizar portafolio (Markowitz)")
    console.print("  [cyan]8[/cyan] Cambiar período de análisis")
    console.print("  [cyan]q[/cyan] Salir")
    return Prompt.ask("\nOpción", choices=["1","2","3","4","5","6","7","8","p","i","q"], default="p")


def pedir_nuevos_pesos(tickers: list) -> dict:
    console.print(f"\n[bold]Ingresa los nuevos pesos (deben sumar 100%).[/bold]")
    console.print(f"[dim]Activos actuales: {', '.join(tickers)}[/dim]\n")

    pesos = {}
    total = 0.0

    for t in tickers:
        while True:
            val = FloatPrompt.ask(f"  Peso para [cyan]{t}[/cyan] (en %)", default=round(100/len(tickers), 1))
            if 0 <= val <= 100:
                pesos[t] = val / 100
                total += val
                break
            console.print("[red]El peso debe estar entre 0 y 100[/red]")

    if abs(total - 100) > 0.5:
        console.print(f"[red]Los pesos suman {total:.1f}%, deben sumar 100%.[/red]")
        return None

    return pesos


def pedir_nuevos_activos(tickers_actuales: list) -> list:
    console.print(f"\n[bold]Activos actuales:[/bold] {', '.join(tickers_actuales)}")
    console.print("[dim]Ingresa los nuevos tickers separados por coma (ej: AAPL, TSLA, SPY)[/dim]")
    entrada = Prompt.ask("Tickers").upper()
    nuevos = [t.strip() for t in entrada.split(",") if t.strip()]
    return nuevos if nuevos else tickers_actuales


# ─── Loop principal ──────────────────────────────────────────────────────────

def main():
    mostrar_bienvenida()

    portafolio = PORTAFOLIO_EJEMPLO.copy()

    while True:
        # Descargar datos
        console.print(f"\n[dim]Descargando datos: {', '.join(portafolio['tickers'])} ({portafolio['periodo']})...[/dim]")

        try:
            precios = descargar_precios(portafolio["tickers"], portafolio["periodo"])
            retornos = calcular_retornos_diarios(precios)
            resumen = resumen_datos(precios)
        except Exception as e:
            console.print(f"[red]Error al descargar datos: {e}[/red]")
            continue

        mostrar_resumen_datos(resumen)

        opcion = menu_principal()

        if opcion == "q":
            console.print("\n[bold cyan]¡Hasta pronto![/bold cyan]")
            break

        elif opcion == "p":
            mi_port = cargar_portafolio()
            mostrar_portafolio_actual(mi_port)
            submenu = Prompt.ask(
                "\n¿Qué quieres ver?",
                choices=["1","2","3","n"],
                default="1",
                show_choices=False,
                show_default=False,
            )
            console.print("  [dim]1=Métricas históricas  2=Gráfica composición  3=Recomendaciones  n=Volver[/dim]")
            submenu = Prompt.ask("Opción", choices=["1","2","3","n"], default="1")
            if submenu == "1":
                analizar_portafolio_real(mi_port, portafolio["periodo"])
            elif submenu == "2":
                graficar_portafolio_cop(mi_port)
            elif submenu == "3":
                mostrar_recomendaciones(mi_port)
            continue

        elif opcion == "i":
            mi_port = cargar_portafolio()
            verificar_precios(mi_port)
            opciones = analizar_opciones_inversion(mi_port)
            comparar = Prompt.ask(
                "\n¿Comparar rendimiento histórico (opciones + alternativas sugeridas)?",
                choices=["s","n"], default="s"
            )
            if comparar == "s":
                comparar_opciones_yfinance(mi_port, portafolio["periodo"])
            recomendar = Prompt.ask("\n¿Ver recomendaciones?", choices=["s","n"], default="s")
            if recomendar == "s":
                mostrar_recomendaciones(mi_port)
            continue

        elif opcion == "1":
            analisis = analizar_portafolio(retornos, portafolio["pesos"], precios)
            mostrar_tabla_activos(analisis)
            mostrar_resumen_portafolio(analisis)

        elif opcion == "2":
            nuevos_pesos = pedir_nuevos_pesos(portafolio["tickers"])
            if nuevos_pesos:
                portafolio["pesos"] = nuevos_pesos
                console.print("[green]Pesos actualizados.[/green]")

        elif opcion == "3":
            nuevos_tickers = pedir_nuevos_activos(portafolio["tickers"])
            # Ajustar pesos automáticamente con distribución igual
            portafolio["tickers"] = nuevos_tickers
            portafolio["pesos"] = {t: 1/len(nuevos_tickers) for t in nuevos_tickers}
            console.print(f"[green]Portafolio actualizado. Pesos igualados al {100/len(nuevos_tickers):.1f}% cada uno.[/green]")
            console.print("[dim]Usa la opción 2 para ajustar los pesos manualmente.[/dim]")

        elif opcion == "4":
            analisis = analizar_portafolio(retornos, portafolio["pesos"], precios)
            mostrar_correlaciones(analisis)

        elif opcion == "5":
            console.print("\n[dim]Generando gráficas... se abrirá una ventana.[/dim]")
            tickers_con_spy = list(portafolio["pesos"].keys())
            if "SPY" not in tickers_con_spy:
                tickers_con_spy.append("SPY")
                precios_graf = descargar_precios(tickers_con_spy, portafolio["periodo"])
                retornos_graf = calcular_retornos_diarios(precios_graf)
            else:
                retornos_graf = retornos
            graficar_portafolio(retornos_graf, portafolio["pesos"])

        elif opcion == "6":
            modo = Prompt.ask(
                "\n¿Sobre qué portafolio correr Monte Carlo?",
                choices=["r","e"],
                default="r",
                show_choices=False,
            )
            console.print("  [dim]r = mi portafolio real (actual vs sugerido)  |  e = portafolio de ejemplo[/dim]")
            modo = Prompt.ask("Modo", choices=["r","e"], default="r")

            anos = FloatPrompt.ask("Años a simular", default=2.0)

            if modo == "r":
                mi_port = cargar_portafolio()
                capital = float(sum(a["valor_cop"] for a in mi_port["activos"]))
                console.print(f"\n[dim]Construyendo retornos del portafolio real ({portafolio['periodo']})...[/dim]")
                ret_real, pw_act, pw_sug = construir_retornos_para_analisis(mi_port, portafolio["periodo"])
                comparar_mc_actual_vs_sugerido(ret_real, pw_act, pw_sug, capital=capital, anos=anos)
            else:
                capital = FloatPrompt.ask("Capital inicial (USD)", default=10000.0)
                dias_sim = int(anos * 252)
                r_port = retornos[list(portafolio["pesos"].keys())].dropna() @ \
                         list(portafolio["pesos"].values())
                trayectorias = simular_montecarlo(r_port, capital_inicial=capital,
                                                  dias=dias_sim, n_simulaciones=1000)
                stats = estadisticas_montecarlo(trayectorias)
                console.print(f"\n  P95: [green]${stats['p95']:,.0f}[/green]  P50: [cyan]${stats['p50']:,.0f}[/cyan]  P05: [red]${stats['p05']:,.0f}[/red]")
                console.print(f"  Prob. ganancia: [green]{stats['prob_ganancia']:.1%}[/green]")
                ver_grafica = Prompt.ask("Ver grafica?", choices=["s","n"], default="s")
                if ver_grafica == "s":
                    graficar_montecarlo(trayectorias, stats)

        elif opcion == "7":
            modo = Prompt.ask(
                "\n¿Optimizar qué portafolio?",
                choices=["r","e"],
                default="r",
                show_choices=False,
            )
            console.print("  [dim]r = mi portafolio real  |  e = portafolio de ejemplo[/dim]")
            modo = Prompt.ask("Modo", choices=["r","e"], default="r")

            if modo == "r":
                mi_port = cargar_portafolio()
                console.print(f"\n[dim]Descargando datos ({portafolio['periodo']}) para optimizar...[/dim]")
                ret_real, pw_act, _ = construir_retornos_para_analisis(mi_port, portafolio["periodo"])
                optimo = optimizar_sharpe(ret_real)
            else:
                optimo = optimizar_sharpe(retornos)

            if not optimo["exito"]:
                console.print("[red]El optimizador no convergió.[/red]")
            else:
                console.print(f"\n[bold cyan]Portafolio optimo:[/bold cyan]  Sharpe [green]{optimo['sharpe']:.2f}[/green]  "
                              f"Retorno [green]{optimo['retorno_anual']:+.2%}[/green]  "
                              f"Vol {optimo['volatilidad_anual']:.2%}\n")
                for id_act, peso in sorted(optimo["pesos"].items(), key=lambda x: -x[1]):
                    if peso > 0.001:
                        barra = "|" * int(peso * 40)
                        console.print(f"  {id_act:20} {peso:5.1%}  [cyan]{barra}[/cyan]")

                if modo == "r":
                    ver_frontera = Prompt.ask("\nVer frontera eficiente?", choices=["s","n"], default="s")
                    if ver_frontera == "s":
                        console.print("[dim]Generando 5,000 portafolios aleatorios...[/dim]")
                        frontera = frontera_eficiente(ret_real)
                        graficar_frontera(frontera, pw_act, optimo, ret_real)
                else:
                    ver_frontera = Prompt.ask("\nVer frontera eficiente?", choices=["s","n"], default="s")
                    if ver_frontera == "s":
                        frontera = frontera_eficiente(retornos)
                        graficar_frontera(frontera, portafolio["pesos"], optimo, retornos)

        elif opcion == "8":
            periodo = Prompt.ask("Período", choices=["1y","2y","5y","10y","max"], default="2y")
            portafolio["periodo"] = periodo
            console.print(f"[green]Período cambiado a {periodo}.[/green]")


if __name__ == "__main__":
    main()
