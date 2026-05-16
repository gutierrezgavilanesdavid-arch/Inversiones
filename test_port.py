from modulos.mi_portafolio import cargar_portafolio, analizar_opciones_inversion, comparar_opciones_yfinance

port = cargar_portafolio()
opciones = analizar_opciones_inversion(port)
comparar_opciones_yfinance(opciones, "2y")
