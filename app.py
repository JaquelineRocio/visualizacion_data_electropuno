"""
Dashboard · Consumo de energía eléctrica de los clientes de Electro Puno S.A.A.
Proyecto de fin de curso – Visualización de datos (UNAP)

Ejecutar primero los notebooks 01, 02 y 03 (generan data/processed/) y luego:
    python app.py      →  http://127.0.0.1:8050
"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from plotly.subplots import make_subplots
from scipy import stats

import kpis

# ----------------------------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------------------------
DATA = Path(__file__).parent / "data" / "processed"
df_all = pd.read_parquet(DATA / "consumo_final.parquet")
df_all["geo_id"] = df_all["geo_id"].astype(str)
df_all["ubigeo"] = df_all["ubigeo"].astype(str)
ind_static = pd.read_csv(DATA / "indicadores_distrito.csv", dtype={"ubigeo": str, "geo_id": str})
geo = json.load(open(DATA / "puno_distritos.geojson", encoding="utf-8"))
resumen_dist = json.load(open(DATA / "distribuciones_resumen.json", encoding="utf-8"))

# Nombre que se muestra en cada polígono (San Miguel se dibuja junto con Juliaca)
nombre_geo = (df_all.groupby("geo_id")["distrito"].unique()
              .apply(lambda v: " + ".join(sorted(v))).to_dict())
ANIOS = sorted(int(a) for a in df_all["anio"].unique())
PROVINCIAS = sorted(df_all["provincia"].unique())
# El filtro de distrito usa el ubigeo: SANTA ROSA existe en dos provincias
DISTRITOS = (df_all[["ubigeo", "distrito", "provincia"]].drop_duplicates("ubigeo")
             .sort_values(["provincia", "distrito"]))
UBIGEOS = set(DISTRITOS["ubigeo"])
PROVINCIA_DE = dict(zip(DISTRITOS["ubigeo"], DISTRITOS["provincia"]))
# Etiqueta corta del distrito; se agrega la provincia solo si el nombre se repite (Santa Rosa)
_repetido = DISTRITOS["distrito"].duplicated(keep=False)
ETIQUETA_DIST = dict(zip(DISTRITOS["ubigeo"], np.where(
    _repetido, DISTRITOS["distrito"].str.title() + " (" + DISTRITOS["provincia"].str.title() + ")",
    DISTRITOS["distrito"].str.title())))
GRUPOS = ["BT5B residencial", "BT5 otras", "BT potencia", "Media tensión"]
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
MESES_LARGO = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
               "noviembre", "diciembre"]

# ----------------------------------------------------------------------------------------------
# Estilo · paleta corporativa "azul petróleo" (validada con el validador de paletas del curso:
# separación para daltonismo ΔE ≥ 12 entre los 3 primeros colores y contraste ≥ 3:1 sobre blanco)
# ----------------------------------------------------------------------------------------------
# Categóricas en orden fijo: azul petróleo, ámbar energía, turquesa, violeta
CATEG = ["#1d5fa6", "#c77519", "#0f9b8e", "#7b5cc4"]
AZUL = CATEG[0]
# Secuencial de un solo tono (claro → oscuro) para magnitudes
SEQ = ["#dce8f5", "#b3cde9", "#86afdb", "#5b90ca", "#3574b6", "#1d5fa6", "#174a83", "#10355e"]
AZUL_OSCURO = SEQ[-1]
AZUL_SUAVE = SEQ[0]
# Foco + contexto (Gestalt figura-fondo): gris pizarra para el contexto, categóricas 1-3 para el foco
CONTEXTO, TINTA, TINTA_2, TENUE, REJILLA = "#c4cbd5", "#0f1b2d", "#4a5568", "#8a94a6", "#e6eaf0"
FONDO_BANDA = "#f4f7fb"   # banda suave que agrupa los meses de un mismo año
FOCO = CATEG[:3]
# Divergente para "por debajo / por encima": azul ↔ gris neutro ↔ terracota
DIVERGENTE = [[0, CATEG[0]], [0.5, "#eef1f5"], [1, "#c0452f"]]
# Estado de alerta (colores reservados, siempre con símbolo + etiqueta)
ALERTAS = {"Consumo atípico": ("#d03b3b", "diamond"), "Precio atípico": ("#ec835a", "triangle-up"),
           "Consumo sin facturación": ("#fab219", "square")}
UMBRALES = [5, 10, 30, 50, 100, 300, 1000]

_eje = dict(gridcolor=REJILLA, zerolinecolor=CONTEXTO, linecolor=CONTEXTO, automargin=True,
            tickfont=dict(color=TINTA_2), title=dict(font=dict(color=TINTA_2, size=12)))
FUENTE = "Inter, Segoe UI, system-ui, sans-serif"
pio.templates["puno"] = go.layout.Template(
    layout=dict(font=dict(family=FUENTE, size=12, color=TINTA_2),
                paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", colorway=CATEG,
                margin=dict(l=10, r=10, t=16, b=10), xaxis=_eje, yaxis=_eje,
                hoverlabel=dict(bgcolor="white", bordercolor=CONTEXTO, align="left",
                                font=dict(family=FUENTE, size=12, color=TINTA)),
                legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title=dict(text=""),
                            itemclick="toggle", itemdoubleclick="toggleothers"),
                modebar=dict(bgcolor="rgba(255,255,255,0)", color=TENUE, activecolor=AZUL),
                transition=dict(duration=350, easing="cubic-in-out")))   # cambios de filtro animados
pio.templates.default = "puno"

METRICAS = {
    "clientes": ("Clientes", ",.0f"),
    "consumo_kwh": ("Energía (kWh/mes)", ",.0f"),
    "monto_soles": ("Facturación (S//mes)", ",.0f"),
    "kwh_cliente_bt5b": ("kWh por cliente residencial (BT5B)", ",.1f"),
    "precio_kwh_bt5b": ("Precio medio BT5B (S/ por kWh)", ",.3f"),
    "pct_fose": ("% clientes BT5B ≤ 100 kWh (FOSE)", ",.1f"),
    "pct_bajo_consumo": ("% clientes BT5B ≤ 30 kWh", ",.1f"),
    "pct_mt_facturacion": ("% facturación en media tensión", ",.1f"),
    "densidad_clientes": ("Clientes por km²", ",.1f"),
}
UNIDADES = {"consumo_kwh": ("", " kWh"), "monto_soles": ("S/ ", ""), "kwh_cliente_bt5b": ("", " kWh"),
            "precio_kwh_bt5b": ("S/ ", ""), "pct_fose": ("", " %"), "pct_bajo_consumo": ("", " %"),
            "pct_mt_facturacion": ("", " %"), "densidad_clientes": ("", " por km²")}
SERIES = [("consumo_kwh", "Energía"), ("monto_soles", "Facturación"), ("clientes", "Suministros"),
          ("kwh_cliente_bt5b", "kWh por hogar")]
# Pestañas analíticas: unidad de comparación (columna clave) y vista de las series
NIVELES = [("provincia", "Provincias"), ("ubigeo", "Distritos")]
VISTAS = [("indice", "Índice (base 100)"), ("valor", "Valor real")]
MAX_UNIDADES = 15   # tope de distritos comparados a la vez (los de más clientes + los elegidos)
LIMITACIONES = [
    "Los datos están agregados por distrito y tarifa: no identifican suministros individuales ni su fecha de "
    "instalación, por lo que no se comparan clientes nuevos y antiguos.",
    "No incluyen pérdidas de energía, morosidad, cobranza ni calidad de servicio (SAIDI/SAIFI); esos indicadores "
    "requieren información interna de la empresa.",
    "Algunos meses no tienen archivo publicado; las series los muestran como huecos y no se interpolan.",
    "Las alertas son estadísticas (modelo log-normal y z-score robusto): señalan registros a revisar, "
    "no confirman errores ni hurto.",
]


# ----------------------------------------------------------------------------------------------
# Funciones de cálculo
# ----------------------------------------------------------------------------------------------
def filtrar(anios, provincias, distritos, grupos, territorio=True):
    d = df_all
    if anios:
        d = d[d["anio"].isin(anios)]
    if grupos:
        d = d[d["grupo_tarifa"].isin(grupos)]
    if territorio and provincias:
        d = d[d["provincia"].isin(provincias)]
    if territorio and distritos:
        d = d[d["ubigeo"].isin(distritos)]
    return d


def contexto_y_foco(anios, provincias, distritos, grupos):
    """Gráficos comparativos: toda la región como contexto y la selección territorial como foco."""
    d_todo = filtrar(anios, provincias, distritos, grupos, territorio=False)
    if not (provincias or distritos):
        return d_todo, d_todo, []
    d_sel = filtrar(anios, provincias, distritos, grupos)
    focos = list(provincias or []) or sorted(d_sel["provincia"].unique())
    return d_todo, d_sel, focos


def indicadores(d: pd.DataFrame, by: list, meses=None) -> pd.DataFrame:
    """Mismos indicadores del Notebook 2. Totales = promedio mensual (meses=1 si se agrupa por periodo)."""
    n = meses or max(d["periodo_cod"].nunique(), 1)
    b5 = d["tarifa"].eq("BT5B")
    t = d.assign(cli_b5=d["clientes"] * b5, kwh_b5=d["consumo_kwh"] * b5, soles_b5=d["monto_soles"] * b5,
                 soles_mt=d["monto_soles"] * d["nivel_tension"].eq("MT"),
                 cli_fose=d["clientes"] * (b5 & (d["kwh_cliente"] <= 100)),
                 cli_bajo=d["clientes"] * (b5 & (d["kwh_cliente"] <= 30)),
                 n_anom=d["anomalia"].astype(int))
    g = t.groupby(by).agg(clientes=("clientes", "sum"), consumo_kwh=("consumo_kwh", "sum"),
                          monto_soles=("monto_soles", "sum"), cli_b5=("cli_b5", "sum"), kwh_b5=("kwh_b5", "sum"),
                          soles_b5=("soles_b5", "sum"), soles_mt=("soles_mt", "sum"), cli_fose=("cli_fose", "sum"),
                          cli_bajo=("cli_bajo", "sum"), area_km2=("area_km2", "first"), anomalias=("n_anom", "sum"))
    out = pd.DataFrame(index=g.index)
    for c in ["clientes", "consumo_kwh", "monto_soles"]:
        out[c] = g[c] / n
    out["kwh_cliente_bt5b"] = g["kwh_b5"] / g["cli_b5"].replace(0, np.nan)
    out["precio_kwh_bt5b"] = g["soles_b5"] / g["kwh_b5"].replace(0, np.nan)
    out["pct_fose"] = g["cli_fose"] / g["cli_b5"].replace(0, np.nan) * 100
    out["pct_bajo_consumo"] = g["cli_bajo"] / g["cli_b5"].replace(0, np.nan) * 100
    out["pct_mt_facturacion"] = g["soles_mt"] / g["monto_soles"].replace(0, np.nan) * 100
    out["densidad_clientes"] = out["clientes"] / g["area_km2"]
    out["anomalias"] = g["anomalias"]
    return out.reset_index()


def por_mes(d, met, por=None):
    """Indicador mes a mes; los meses sin archivo quedan como hueco (no se interpolan)."""
    rango = pd.date_range(d["periodo"].min(), d["periodo"].max(), freq="MS")
    m = indicadores(d, ["periodo"] + ([por] if por else []), meses=1)
    if not por:
        return m.set_index("periodo")[met].reindex(rango)
    return m.pivot(index=por, columns="periodo", values=met).reindex(columns=rango)


def ajustar_mapa(fig):
    """Mapa SVG (no requiere internet ni WebGL): proyección Mercator ajustada a los distritos."""
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
    fig.update_traces(marker_line_width=0.6, marker_line_color="white")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), dragmode=False)
    return fig


def vacio(msg="Sin datos para los filtros seleccionados"):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=TINTA_2))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def fmt(v, f=",.0f"):
    return "–" if v is None or pd.isna(v) else format(v, f)


def valor(met, v):
    pre, suf = UNIDADES.get(met, ("", ""))
    return "–" if pd.isna(v) else f"{pre}{format(v, METRICAS[met][1])}{suf}"


def hover_valor(met, campo="y"):
    pre, suf = UNIDADES.get(met, ("", ""))
    return f"{pre}%{{{campo}:{METRICAS[met][1]}}}{suf}"


def mes_es(f, salto=" "):
    return f"{MESES[f.month - 1]}{salto}{f.year}"


def ejes_tiempo(fig, fechas, **kw):
    """Marcas trimestrales con nombres de mes en español."""
    marcas = [f for f in fechas if f.month in (1, 4, 7, 10)]
    fig.update_xaxes(tickvals=marcas, ticktext=[mes_es(f, "<br>") if f.month == 1 else MESES[f.month - 1]
                                               for f in marcas], **kw)


def color_provincia(p, focos):
    if not focos:
        return AZUL
    if p not in focos:
        return CONTEXTO
    i = focos.index(p)
    return FOCO[i] if i < len(FOCO) else AZUL_OSCURO


def etiquetas_finales(fig, finales, rango):
    """Etiquetas directas al final de cada línea, separadas para que no se encimen."""
    sep = rango * 0.07
    finales.sort(key=lambda e: e[1])
    for k in range(1, len(finales)):
        if finales[k][0] == finales[k - 1][0] and finales[k][1] - finales[k - 1][1] < sep:
            finales[k][1] = finales[k - 1][1] + sep
    for x, y, txt in finales:
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False, xanchor="left", xshift=8,
                           font=dict(size=11, color=TINTA))


def tile(titulo, val, sub="", chip=None):
    return html.Div([html.Div(titulo, className="kpi-titulo"), html.Div(val, className="kpi-valor"),
                     html.Div(sub, className="kpi-sub"), chip], className="kpi")


def chip_vs(v, ref, sentido=0, puntos=False, frente="vs región"):
    """Píldora de comparación: % relativo (o puntos porcentuales) frente a una referencia."""
    if ref is None or pd.isna(v) or pd.isna(ref) or (not puntos and ref == 0):
        return None
    dif = v - ref if puntos else (v / ref - 1) * 100
    clase = "neutro" if sentido == 0 or abs(dif) < 0.5 else ("bueno" if (dif > 0) == (sentido > 0) else "malo")
    return html.Span(f"{'▲' if dif >= 0 else '▼'} {abs(dif):.1f} {'pp' if puntos else '%'} {frente}",
                     className=f"delta {clase}")


def compacto(v):
    if pd.isna(v):
        return ""
    for lim, suf in [(1e6, "M"), (1e3, "k")]:
        if abs(v) >= lim:
            return f"{v / lim:.1f}{suf}"
    return f"{v:.1f}"


# ---------------------------- Ámbito de las pestañas analíticas -------------------------------
def nombre_unidad(k):
    """Provincia → nombre en título; ubigeo → nombre del distrito."""
    return ETIQUETA_DIST.get(k, str(k).title())


def describir_seleccion(prov, dist):
    if dist:
        nombres = [ETIQUETA_DIST.get(u, u) for u in dist]
        return " y ".join(nombres) if len(nombres) <= 2 else f"{len(nombres)} distritos"
    if prov:
        return " y ".join(p.title() for p in prov) if len(prov) <= 2 else f"{len(prov)} provincias"
    return "Toda la región"


def describir_anios(anio):
    return ", ".join(str(a) for a in sorted(anio)) if anio else f"{ANIOS[0]}–{ANIOS[-1]}"


def ambito(anio, prov, dist, grp, nivel):
    """Contexto (unidades a comparar) y foco (selección) según el nivel elegido.

    nivel='provincia' → se comparan las 13 provincias; foco = provincias filtradas (o las de los distritos).
    nivel='ubigeo'    → se comparan los distritos de las provincias filtradas (máx. MAX_UNIDADES por clientes,
                        más los elegidos); foco = distritos filtrados.
    """
    d_sel = filtrar(anio, prov, dist, grp)
    if nivel == "ubigeo":
        d_ctx = filtrar(anio, prov, None, grp)
        focos = list(dist or [])
        mayores = list(d_ctx.groupby("ubigeo")["clientes"].sum().nlargest(MAX_UNIDADES).index)
        d_ctx = d_ctx[d_ctx["ubigeo"].isin(mayores + focos)]
    else:
        d_ctx = filtrar(anio, None, None, grp)
        focos = list(prov or []) or (sorted(d_sel["provincia"].unique()) if dist else [])
    return d_ctx, d_sel, focos


def alto_filas(n, por_fila=26, minimo=320, extra=90):
    return max(minimo, n * por_fila + extra)


# ----------------------------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------------------------
def cabecera_seccion(etiqueta, titulo, nota, control=None):
    texto = html.Div([html.Div(etiqueta, className="seccion-etiqueta"), html.H2(titulo),
                      html.P(nota, className="nota")])
    return html.Div([texto, html.Div(control, className="seccion-control")] if control is not None else [texto],
                    className="seccion-cabecera")


def tarjeta(hijos, clase="", etiqueta=None, titulo=None, nota="", control=None):
    hijos = hijos if isinstance(hijos, list) else [hijos]
    cab = [cabecera_seccion(etiqueta, titulo, nota, control)] if titulo else []
    return html.Div(cab + hijos, className=f"tarjeta {clase}")


CONFIG_GRAFICO = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                  "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "electropuno"}}


def grafico(id_, alto):
    return dcc.Graph(id=id_, style={"height": f"{alto}px"}, config=CONFIG_GRAFICO)


def grafico_auto(id_):
    """Gráfico cuya altura la define la figura (depende del número de provincias/distritos)."""
    return dcc.Graph(id=id_, style={"minHeight": "300px"}, config=CONFIG_GRAFICO)


def barra_local(controles, id_ambito):
    """Ámbito vigente a la vista; los controles de la pestaña quedan en un panel plegable (menos ruido)."""
    opciones = html.Details([html.Summary("Opciones de vista"), html.Div(controles, className="opciones-panel")],
                            className="opciones")
    return html.Div([html.Div(id=id_ambito, className="ambito"), opciones], className="barra-local")


def control(etiqueta, componente):
    return html.Div([html.Label(etiqueta), componente], className="control")


def filtro(etiqueta, control, clase=""):
    return html.Div([html.Label(etiqueta), control], className=f"filtro {clase}")


def segmentado(id_, opciones, val):
    return dcc.RadioItems(id=id_, value=val, inline=True, className="segmentado",
                          options=[{"label": n, "value": k} for k, n in opciones])


app = Dash(__name__, title="Electro Puno · Tablero gerencial", external_stylesheets=[
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"])
server = app.server

app.layout = html.Div([
    html.Header([
        html.Div([html.Div("ELECTRO PUNO S.A.A. · TABLERO GERENCIAL", className="marca"),
                  html.H1("Consumo de energía eléctrica"),
                  html.P("Datos abiertos 2023-2026 · 13 provincias · 109 distritos")]),
        html.Button("Descargar datos filtrados (CSV)", id="btn-descarga", className="boton"),
        dcc.Download(id="descarga"),
    ], className="cabecera"),

    # Filtros únicos para todo el tablero (una sola fila, fija al desplazarse)
    html.Div([
        filtro("Año", dcc.Dropdown(id="f-anio", options=ANIOS, multi=True, placeholder="Todos los años")),
        filtro("Provincia", dcc.Dropdown(id="f-provincia", options=PROVINCIAS, multi=True,
                                         placeholder="Todas las provincias"), "ancho"),
        filtro("Distrito", dcc.Dropdown(id="f-distrito", multi=True, placeholder="Todos los distritos"), "ancho"),
        filtro("Grupo tarifario", dcc.Dropdown(id="f-grupo", options=GRUPOS, multi=True,
                                               placeholder="Todos los grupos"), "ancho"),
        html.Button("Limpiar filtros", id="btn-limpiar", className="boton-sec"),
    ], className="filtros"),

    dcc.Tabs(className="tabs", children=[
        dcc.Tab(label="Resumen gerencial", children=[
            html.Div(id="resumen-periodo", className="periodo"),
            html.Div(id="kpis-principales", className="kpis-principales"),
            html.Details([
                html.Summary("Ver los otros 8 indicadores"),
                html.Div(id="kpis-complementarios", className="kpis-complementarios"),
            ], className="kpis-detalles"),
            html.Div([tarjeta(grafico("g-resumen-consumo", 400), "col-12", "TENDENCIA",
                              "¿Cómo evoluciona la empresa mes a mes?",
                              "Los huecos son meses sin archivo publicado.",
                              segmentado("r-tendencia", SERIES, "consumo_kwh"))], className="fila"),
            html.Div([
                tarjeta(grafico("g-estacional", 360), "col-6", "ESTACIONALIDAD", "¿Qué meses consumen más?",
                        "Energía de cada mes; una línea por año."),
                tarjeta(grafico("g-concentracion", 360), "col-6", "CONCENTRACIÓN",
                        "¿Dónde se concentra el ingreso?", "Top 10 distritos por % de la facturación del último mes."),
            ], className="fila"),
            html.Div([tarjeta(html.Ul(id="hallazgos", className="hallazgos"), "col-12", "HALLAZGOS",
                              "Lo que la Gerencia debe saber", "Generado automáticamente para el último mes.")],
                     className="fila"),
            html.Div([
                html.Strong("Comparativa de instalaciones nuevas y antiguas pendiente de datos de altas. "),
                "El consumo disponible está agregado y no contiene identificador de suministro ni fecha de instalación.",
            ], className="aviso-datos"),
            html.Details([
                html.Summary("Ficha técnica de los KPIs (definición y fórmula)"),
                dash_table.DataTable(
                    [{"bloque": k["bloque"], "kpi": k["nombre"], "unidad": k["unidad"], "formula": k["formula"],
                      "pregunta": k["pregunta"],
                      "lectura": {1: "▲ más es mejor", -1: "▼ menos es mejor", 0: "informativo"}[k["sentido"]]}
                     for k in kpis.CATALOGO],
                    [{"name": n, "id": c} for c, n in [("bloque", "Bloque"), ("kpi", "KPI"), ("unidad", "Unidad"),
                                                       ("formula", "Fórmula"), ("pregunta", "Pregunta que responde"),
                                                       ("lectura", "Lectura")]],
                    style_cell={"fontFamily": "Inter, Segoe UI, sans-serif", "fontSize": 13, "padding": "6px",
                                "textAlign": "left", "whiteSpace": "normal", "height": "auto"},
                    style_header={"fontWeight": 600, "backgroundColor": FONDO_BANDA}),
                html.P("Con datos abiertos no es posible medir pérdidas de energía, morosidad ni calidad de servicio "
                       "(SAIDI/SAIFI); requieren información interna de la empresa.", className="nota"),
            ], className="kpis-detalles"),
        ]),
        dcc.Tab(label="Territorio", children=[
            html.Div([
                tarjeta([grafico("g-mapa", 520),
                         html.P("San Miguel se muestra junto con Juliaca (no tiene polígono propio en el GeoJSON).",
                                className="nota")], "col-8", "MAPA", "¿Cómo se reparte en el territorio?",
                        "Clic en un distrito para ver su ficha.",
                        dcc.Dropdown(id="f-metrica", value="kwh_cliente_bt5b", clearable=False,
                                     className="selector",
                                     options=[{"label": v[0], "value": k} for k, v in METRICAS.items()])),
                tarjeta([html.Div(id="ficha"), grafico("g-ficha-tarifa", 250)], "col-4", "FICHA",
                        "Distrito seleccionado", "Promedio mensual del periodo filtrado."),
            ], className="fila"),
            html.Div([
                tarjeta(grafico("g-top", 440), "col-6", "RANKING DE DISTRITOS", "Top 15 distritos",
                        "Según el indicador elegido en el mapa."),
                tarjeta(grafico("g-provincia", 440), "col-6", "RANKING DE PROVINCIAS",
                        "Provincias frente a la mediana", "Clic en una provincia para filtrarla."),
            ], className="fila"),
        ]),
        dcc.Tab(label="Tarifas y composición", children=[
            html.Div([
                tarjeta(grafico("g-composicion", 400), "col-6", "COMPOSICIÓN", "¿Quién pesa más?",
                        "Participación de cada grupo tarifario en clientes, energía y facturación."),
                tarjeta(grafico("g-precio", 400), "col-6", "PRECIO", "Precio efectivo en baja tensión",
                        "Distribución del precio (S/ por kWh) en cada tarifa."),
            ], className="fila"),
            html.Div([tarjeta(grafico("g-treemap", 540), "col-12", "JERARQUÍA", "Clientes por provincia y distrito",
                              "Tamaño = clientes · color = kWh por cliente BT5B. Clic para acercar.")],
                     className="fila"),
        ]),
        dcc.Tab(label="Segmentos de distritos", children=[
            html.Div([
                tarjeta(grafico("g-cluster-mapa", 480), "col-6", "SEGMENTOS", "Mapa de segmentos (K-Means)",
                        "Distritos con perfil de consumo parecido comparten color."),
                tarjeta(grafico("g-cluster-scatter", 480), "col-6", "POSICIÓN", "Tamaño frente a consumo por cliente",
                        "Burbuja = facturación mensual."),
            ], className="fila"),
            html.Div([tarjeta(grafico("g-perfil", 300), "col-12", "PERFIL", "¿Qué distingue a cada segmento?",
                              "Mediana de cada indicador; el color compara el segmento con el promedio de los tres.")],
                     className="fila"),
        ]),
        dcc.Tab(label="Distribución y anomalías", children=[
            barra_local([
                control("Comparar por", segmentado("d-nivel", NIVELES, "provincia")),
                control("Tipos de alerta", dcc.Checklist(id="d-tipos", options=list(ALERTAS), value=list(ALERTAS),
                                                         inline=True, className="chips")),
                control("Umbral de consumo (kWh/mes)",
                        html.Div(dcc.Slider(id="d-umbral", min=np.log10(5), max=3, step=None, value=2,
                                            marks={float(np.log10(v)): f"{v}" for v in UMBRALES}),
                                 className="deslizador")),
            ], "d-ambito"),
            html.Div(id="d-tiles", className="kpis"),
            html.Div([
                tarjeta(grafico("g-hist", 360), "col-7", "DISTRIBUCIÓN", "¿Cuánto consume un hogar?",
                        "Gris = región · azul = selección · punteado = umbral."),
                tarjeta(grafico_auto("g-ic"), "col-5", "COMPARACIÓN", "Consumo medio por hogar",
                        "Media con IC 95 %. Clic para filtrar."),
            ], className="fila"),
            html.Div([
                tarjeta(grafico("g-anom", 400), "col-7", "ALERTAS", "¿Dónde aparecen los atípicos?",
                        "Gris = normal. Clic en un punto para ver su distrito."),
                tarjeta(grafico_auto("g-alertas"), "col-5", "PRIORIDAD", "¿Qué revisar primero?",
                        "Clic para filtrar."),
            ], className="fila"),
            html.Div([tarjeta(grafico("g-alertas-mes", 280), "col-12", "CRONOLOGÍA", "¿Cuándo aparecen las alertas?",
                              "Clic en una barra para filtrar ese año.")], className="fila"),
        ]),
        dcc.Tab(label="Evolución temporal", children=[
            barra_local([
                control("Indicador", segmentado("f-serie", SERIES, "consumo_kwh")),
                control("Escala", segmentado("e-vista", VISTAS, "indice")),
                control("Comparar por", segmentado("e-nivel", NIVELES, "provincia")),
            ], "e-ambito"),
            html.Div(id="e-tiles", className="kpis"),
            html.Div([
                tarjeta(grafico("g-tendencia", 360), "col-8", "TENDENCIA", "¿Sube o baja la selección?",
                        "Punteado = año anterior."),
                tarjeta(grafico("g-calendario", 360), "col-4", "CALENDARIO", "Mes × año",
                        "Clic para filtrar el año."),
            ], className="fila"),
            html.Div([
                tarjeta(grafico_auto("g-multiples"), "col-8", "TRAYECTORIAS", "¿Cómo evoluciona cada unidad?",
                        "Gris = contexto · color = selección."),
                tarjeta(grafico_auto("g-pendiente"), "col-4", "CRECIMIENTO", "¿Quién crece más?",
                        "Último año vs primero."),
            ], className="fila"),
        ]),
        dcc.Tab(label="Conclusiones", children=[
            html.Div(id="concl-general", className="concl-general"),
            tarjeta(html.Div(id="conclusiones", className="hz-tabla"), "concl-tabla", "HALLAZGOS Y DECISIONES",
                    "Qué encontramos y qué hacer", "Ordenado por prioridad.",
                    html.Div([html.Span([html.I(className=f"punto {c}"), t]) for c, t in
                              [("alta", "Alta"), ("media", "Media"), ("estrategica", "Estratégica")]],
                             className="leyenda-prio")),
            html.Details([html.Summary("Limitaciones del análisis"),
                          html.Ul([html.Li(t) for t in LIMITACIONES], className="limitaciones")],
                         className="kpis-detalles"),
        ]),
    ]),
    html.Footer("Fuente: Electro Puno S.A.A. – Plataforma Nacional de Datos Abiertos. "
                "Límites distritales: INEI (peru_distrital_simple.geojson)."),
], className="contenedor")

FILTROS = [Input("f-anio", "value"), Input("f-provincia", "value"), Input("f-distrito", "value"),
           Input("f-grupo", "value")]
# Gráficos cuyo clic alterna una provincia, un distrito o un año en los filtros superiores
CLIC_FILTRO = ["g-provincia", "g-ic", "g-anom", "g-alertas", "g-alertas-mes", "g-calendario",
               "g-multiples", "g-pendiente"]


# ----------------------------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------------------------
# ---------------------------- Filtros ---------------------------------------------------------
@app.callback([Output("f-distrito", "options"), Output("f-distrito", "value")],
              Input("f-provincia", "value"), State("f-distrito", "value"))
def opciones_distrito(prov, sel):
    t = DISTRITOS[DISTRITOS["provincia"].isin(prov)] if prov else DISTRITOS
    opciones = [{"label": f"{r.distrito.title()} ({r.provincia.title()})", "value": r.ubigeo}
                for r in t.itertuples()]
    return opciones, [v for v in (sel or []) if v in set(t["ubigeo"])]


def clave_del_clic(punto):
    """Clave del punto: primer valor de customdata o, en los mapas de calor (Dash no envía su customdata),
    el valor del eje categórico (ubigeo, provincia o año como texto)."""
    cd = punto.get("customdata")
    primero = cd[0] if isinstance(cd, list) and cd else cd
    candidatos = (primero, punto.get("y"), punto.get("x"))
    for c in candidatos:
        if isinstance(c, str) and c in UBIGEOS:
            return "distrito", c
        if isinstance(c, str) and c in PROVINCIAS:
            return "provincia", c
    if isinstance(primero, (int, float)) and int(primero) in ANIOS:
        return "anio", int(primero)
    for c in candidatos[1:]:   # ejes categóricos de año ("2025")
        if isinstance(c, str) and c.isdigit() and int(c) in ANIOS:
            return "anio", int(c)
    return None, None


def alternar(lista, v):
    return [x for x in lista if x != v] if v in lista else lista + [v]


@app.callback([Output("f-anio", "value", allow_duplicate=True), Output("f-provincia", "value"),
               Output("f-distrito", "value", allow_duplicate=True)],
              [Input(g, "clickData") for g in CLIC_FILTRO],
              [State("f-anio", "value"), State("f-provincia", "value"), State("f-distrito", "value")],
              prevent_initial_call=True)
def filtro_cruzado(*args):
    """Clic en un gráfico = agregar o quitar esa provincia, distrito o año del filtro superior."""
    click = ctx.triggered[0]["value"] if ctx.triggered else None
    tipo, clave = clave_del_clic(click["points"][0]) if click else (None, None)
    anio, prov, dist = (list(v or []) for v in args[-3:])
    if tipo == "anio":
        return alternar(anio, clave), no_update, no_update
    if tipo == "provincia":
        return no_update, alternar(prov, clave), no_update
    if tipo == "distrito":
        # el distrito debe quedar dentro de las provincias filtradas
        prov_nueva = prov + [PROVINCIA_DE[clave]] if prov and PROVINCIA_DE[clave] not in prov else no_update
        return no_update, prov_nueva, alternar(dist, clave)
    return no_update, no_update, no_update


@app.callback([Output("f-anio", "value"), Output("f-provincia", "value", allow_duplicate=True),
               Output("f-distrito", "value", allow_duplicate=True), Output("f-grupo", "value")],
              Input("btn-limpiar", "n_clicks"), prevent_initial_call=True)
def limpiar(_):
    return [], [], [], []


# ---------------------------- Resumen gerencial -----------------------------------------------
@app.callback(Output("g-resumen-consumo", "figure"), FILTROS + [Input("r-tendencia", "value")])
def resumen_consumo(anio, prov, dist, grp, met):
    d = filtrar(anio, prov, dist, grp)
    if d.empty:
        return vacio()
    s = por_mes(d, met)
    if s.dropna().empty:
        return vacio("El indicador no aplica a los grupos tarifarios elegidos")
    fig = go.Figure(go.Scatter(
        x=s.index, y=s, mode="lines+markers", line=dict(color=AZUL, width=2.5),
        marker=dict(size=7, color=AZUL), text=[mes_es(f) for f in s.index],
        hovertemplate="%{text}<br>" + hover_valor(met) + "<extra></extra>"))
    v = s.dropna()
    for fecha, txt, ay in [(v.idxmax(), "máximo", -32), (v.idxmin(), "mínimo", 32)]:
        fig.add_annotation(x=fecha, y=v[fecha], text=f"{txt} · {mes_es(fecha)}", showarrow=True, arrowhead=0,
                           arrowcolor=TENUE, ax=0, ay=ay, font=dict(size=11, color=TINTA_2))
    fig.add_annotation(x=v.index[-1], y=v.iloc[-1], text=f"<b>{valor(met, v.iloc[-1])}</b>", showarrow=False,
                       xanchor="left", xshift=10, font=dict(size=12, color=TINTA))
    ejes_tiempo(fig, s.index)
    fig.update_yaxes(title=METRICAS[met][0], rangemode="tozero", tickformat=",.3s")
    fig.update_layout(hovermode="x unified", showlegend=False, margin=dict(l=10, r=90, t=16, b=10))
    return fig


@app.callback(Output("g-estacional", "figure"), FILTROS)
def perfil_estacional(anio, prov, dist, grp):
    d = filtrar(anio, prov, dist, grp)
    if d.empty:
        return vacio()
    m = por_mes(d, "consumo_kwh").dropna().to_frame("v")
    m = m.assign(anio=m.index.year, mes=m.index.month)
    fig, finales = go.Figure(), []
    for i, (a, g) in enumerate(m.groupby("anio")):
        fig.add_scatter(x=[MESES[k - 1] for k in g["mes"]], y=g["v"], mode="lines+markers", name=str(a),
                        line=dict(color=CATEG[i % len(CATEG)], width=2), marker=dict(size=7),
                        hovertemplate=f"{a} · %{{x}}<br>%{{y:,.0f}} kWh<extra></extra>")
        finales.append([MESES[g["mes"].iloc[-1] - 1], g["v"].iloc[-1], str(a)])
    etiquetas_finales(fig, finales, m["v"].max() - m["v"].min())
    fig.update_xaxes(categoryorder="array", categoryarray=MESES)
    fig.update_yaxes(title="kWh", tickformat=",.3s")
    fig.update_layout(hovermode="x unified", margin=dict(l=10, r=40, t=16, b=10))
    return fig


def etiqueta_periodo(p):
    return f"{MESES[int(str(p)[4:]) - 1]} {str(p)[:4]}"


def tarjeta_kpi(k, val, var, previo):
    """Tarjeta breve; la pregunta y la fórmula quedan disponibles al pasar el cursor."""
    sufijo = " %" if k["unidad"].startswith("%") else ""
    prefijo = "S/ " if k["unidad"].startswith("S/") and "por kWh" not in k["unidad"] else ""
    if previo is None:
        delta = None  # el encabezado ya indica que hay un solo mes cargado
    elif pd.isna(var):
        delta = html.Span("Sin dato previo", className="delta neutro")
    else:
        bueno = (var >= 0) == (k["sentido"] > 0)
        clase = "neutro" if k["sentido"] == 0 or abs(var) < 0.05 else ("bueno" if bueno else "malo")
        delta = html.Span(f"{'▲' if var >= 0 else '▼'} {abs(var):.1f}% vs {etiqueta_periodo(previo)}",
                          className=f"delta {clase}")
    return html.Div([
        html.Div(k["nombre"], className="kpi-titulo"),
        html.Div(f"{prefijo}{fmt(val, k['fmt'])}{sufijo}", className="kpi-valor"),
        html.Div(k["unidad"], className="kpi-sub"),
        delta,
    ], className="kpi", title=f"{k['pregunta']} Fórmula: {k['formula']}")


@app.callback([Output("resumen-periodo", "children"), Output("kpis-principales", "children"),
               Output("kpis-complementarios", "children"),
               Output("hallazgos", "children"), Output("g-concentracion", "figure")], FILTROS)
def resumen_gerencial(anio, prov, dist, grp):
    d = filtrar(anio, prov, dist, grp)
    if d.empty:
        return "Sin datos para los filtros seleccionados", [], [], [], vacio()
    periodos = sorted(d["periodo_cod"].unique())
    actual, previo = periodos[-1], (periodos[-2] if len(periodos) > 1 else None)
    tabla = kpis.tabla_kpis(d)
    if dist:
        ambito = ", ".join(sorted(d["distrito"].str.title().unique()))
    else:
        ambito = ", ".join(p.title() for p in prov) if prov else "toda la concesión (13 provincias)"
    cab = [html.B(f"Periodo analizado: {etiqueta_periodo(actual)}"),
           f" · comparado con {etiqueta_periodo(previo)}" if previo else " · carga 2+ meses para ver la variación",
           f" · Ámbito: {ambito}"]

    indices_kpi = {k["id"]: i for i, k in enumerate(kpis.CATALOGO)}
    principales = ["energia_mwh", "clientes", "facturacion", "kwh_residencial"]
    tarjetas = [tarjeta_kpi(kpis.POR_ID[i], tabla.loc[indices_kpi[i], "valor"],
                            tabla.loc[indices_kpi[i], "var_vs_mes_anterior_%"], previo) for i in principales]
    complementarios = []
    for bloque in kpis.BLOQUES:
        indicadores_bloque = [tarjeta_kpi(k, tabla.loc[i, "valor"], tabla.loc[i, "var_vs_mes_anterior_%"], previo)
                              for i, k in enumerate(kpis.CATALOGO)
                              if k["bloque"] == bloque and k["id"] not in principales]
        if indicadores_bloque:
            complementarios.append(html.Div([html.Div(bloque, className="bloque-titulo"),
                                             html.Div(indicadores_bloque, className="kpi-grid")], className="bloque"))

    # Hallazgos automáticos (último periodo)
    u = d[d["periodo_cod"] == actual]
    k = kpis.calcular_kpis(u)
    dist_u = indicadores(u, ["provincia", "distrito"])
    dist_u["pct_fact"] = dist_u["monto_soles"] / dist_u["monto_soles"].sum() * 100
    top = dist_u.nlargest(1, "monto_soles").iloc[0]
    items = [f"{top['distrito'].title()} ({top['provincia'].title()}) genera el {top['pct_fact']:.1f}% de la "
             f"facturación; los 10 distritos mayores concentran el {k['pct_top10']:.1f}%.",
             f"El {k['pct_mt']:.1f}% del ingreso viene de media tensión (minería/industria): "
             "perder un gran cliente impacta directamente en la caja."]
    mt = dist_u.assign(mt=dist_u["monto_soles"] * dist_u["pct_mt_facturacion"].fillna(0) / 100).nlargest(3, "mt")
    if mt["mt"].sum() > 0:
        items.append("Mayor facturación en media tensión: " + ", ".join(mt["distrito"].str.title()) + ".")
    if pd.notna(k["monto_alerta"]) and k["n_alertas"] > 0:
        al = u[u["anomalia"]].groupby("distrito")["monto_soles"].sum().nlargest(3)
        items.append(f"S/ {k['monto_alerta']:,.0f} en {k['n_alertas']:,} registros con alerta. "
                     f"Priorizar inspección en: {', '.join(al.index.str.title())}.")
    vul = dist_u[dist_u["clientes"] >= 500].dropna(subset=["pct_bajo_consumo"]).nlargest(3, "pct_bajo_consumo")
    if len(vul):
        items.append(f"El {k['pct_bajo_consumo']:.1f}% de hogares consume ≤ 30 kWh. Mayor proporción en "
                     + ", ".join(f"{r.distrito.title()} ({r.pct_bajo_consumo:.0f}%)" for r in vul.itertuples())
                     + " → focalizar programas sociales o revisar la medición.")
    if pd.notna(k["kwh_residencial"]):
        items.append(f"Un hogar típico consume {k['kwh_residencial']:.1f} kWh/mes y paga en promedio "
                     f"S/ {k['precio_bt']:.3f} por kWh en baja tensión.")

    # Concentración: top 10 distritos (% de la facturación y acumulado)
    t10 = dist_u.nlargest(10, "pct_fact").copy()
    t10["acum"] = t10["pct_fact"].cumsum()
    t10 = t10.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=t10["pct_fact"], y=t10["distrito"].str.title(), orientation="h", marker=dict(color=AZUL, cornerradius=4),
        text=[f"{a:.1f}%  · acum. {b:.0f}%" for a, b in zip(t10["pct_fact"], t10["acum"])],
        textposition="outside", textfont=dict(color=TINTA_2), cliponaxis=False,
        customdata=t10[["provincia", "monto_soles"]],
        hovertemplate="<b>%{y}</b> (%{customdata[0]})<br>%{x:.1f}% · S/ %{customdata[1]:,.0f}<extra></extra>"))
    fig.update_xaxes(range=[0, t10["pct_fact"].max() * 1.5], title="% de la facturación", showgrid=False)
    fig.update_yaxes(showgrid=False)
    return cab, tarjetas, complementarios, [html.Li(i) for i in items], fig


# ---------------------------- Territorio -------------------------------------------------------
@app.callback(Output("g-mapa", "figure"), FILTROS + [Input("f-metrica", "value")])
def mapa(anio, prov, dist, grp, met):
    d = filtrar(anio, prov, dist, grp)
    if d.empty:
        return vacio()
    m = indicadores(d, ["geo_id"])
    m["nombre"] = m["geo_id"].map(nombre_geo)
    etiqueta, f = METRICAS[met]
    hi = m[met].quantile(0.95) if met in ("clientes", "consumo_kwh", "monto_soles", "densidad_clientes") else m[met].max()
    fig = px.choropleth(m, geojson=geo, locations="geo_id", featureidkey="properties.IDDIST", color=met,
                        color_continuous_scale=SEQ, range_color=(m[met].min(), hi), hover_name="nombre",
                        custom_data=["geo_id"],
                        hover_data={"geo_id": False, met: f":{f}", "clientes": ":,.0f",
                                    "kwh_cliente_bt5b": ":,.1f", "pct_fose": ":.1f"},
                        labels={met: etiqueta, "clientes": "Clientes", "kwh_cliente_bt5b": "kWh/cliente BT5B",
                                "pct_fose": "% ≤ 100 kWh"})
    ajustar_mapa(fig)
    fig.update_layout(coloraxis_colorbar=dict(title="", thickness=12, outlinewidth=0))
    return fig


@app.callback([Output("ficha", "children"), Output("g-ficha-tarifa", "figure")],
              FILTROS + [Input("g-mapa", "clickData")])
def ficha(anio, prov, dist, grp, click):
    gid = click["points"][0]["location"] if click else "211101"
    d = filtrar(anio, prov, dist, grp)
    d = d[d["geo_id"] == gid]
    if d.empty:
        return html.P("Selecciona un distrito con datos en el mapa.", className="nota"), vacio("")
    t = indicadores(d.assign(_t=1), ["_t"]).iloc[0]
    seg = ind_static.loc[ind_static["geo_id"] == gid, "cluster"]
    filas = [("Provincia", d["provincia"].iloc[0].title()), ("Clientes", fmt(t.clientes)),
             ("Energía", f"{fmt(t.consumo_kwh)} kWh/mes"), ("Facturación", f"S/ {fmt(t.monto_soles)}/mes"),
             ("kWh por cliente BT5B", fmt(t.kwh_cliente_bt5b, ",.1f")),
             ("% clientes ≤ 100 kWh", f"{fmt(t.pct_fose, ',.1f')} %"),
             ("% facturación MT", f"{fmt(t.pct_mt_facturacion, ',.1f')} %"),
             ("Segmento", seg.iloc[0].split(":")[0] if len(seg) else "–"), ("Registros anómalos", fmt(t.anomalias))]
    cont = [html.H3(nombre_geo.get(gid, gid).title()),
            html.Table([html.Tr([html.Td(a, className="ficha-k"), html.Td(b)]) for a, b in filas], className="ficha")]
    tar = d.groupby("tarifa", as_index=False)["monto_soles"].sum().sort_values("monto_soles")
    fig = go.Figure(go.Bar(x=tar["monto_soles"], y=tar["tarifa"], orientation="h", texttemplate="S/ %{x:.3s}",
                           textposition="outside", textfont=dict(color=TINTA_2), cliponaxis=False,
                           marker=dict(color=AZUL, cornerradius=4),
                           hovertemplate="%{y}<br>S/ %{x:,.0f}<extra></extra>"))
    fig.update_xaxes(range=[0, tar["monto_soles"].max() * 1.3], visible=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(margin=dict(l=10, r=10, t=4, b=4))
    return cont, fig


@app.callback(Output("g-top", "figure"), FILTROS + [Input("f-metrica", "value")])
def top(anio, prov, dist, grp, met):
    d = filtrar(anio, prov, dist, grp)
    if d.empty:
        return vacio()
    m = indicadores(d, ["provincia", "distrito"])
    if met not in ("clientes", "consumo_kwh", "monto_soles"):
        m = m[m["clientes"] >= 100]  # evita ratios inestables en distritos muy pequeños
    m = m.nlargest(15, met).sort_values(met)
    fig = go.Figure(go.Bar(x=m[met], y=m["distrito"].str.title(), orientation="h",
                           marker=dict(color=AZUL, cornerradius=4), customdata=m["provincia"],
                           texttemplate=hover_valor(met, "x"), textposition="outside",
                           textfont=dict(color=TINTA_2), cliponaxis=False,
                           hovertemplate="<b>%{y}</b> (%{customdata})<br>" + hover_valor(met, "x") + "<extra></extra>"))
    fig.update_xaxes(range=[0, m[met].max() * 1.25], title=METRICAS[met][0])
    fig.update_yaxes(showgrid=False)
    return fig


@app.callback(Output("g-provincia", "figure"), FILTROS + [Input("f-metrica", "value")])
def por_provincia(anio, prov, dist, grp, met):
    """Lollipop: todas las provincias como contexto, las filtradas en color (Gestalt figura-fondo)."""
    d_todo, _, focos = contexto_y_foco(anio, prov, dist, grp)
    m = indicadores(d_todo, ["provincia"]).dropna(subset=[met]).sort_values(met)
    if m.empty:
        return vacio()
    ref = m[met].median()
    colores = [color_provincia(p, focos) for p in m["provincia"]]
    fig = go.Figure()
    for p, v, c in zip(m["provincia"], m[met], colores):
        fig.add_shape(type="line", x0=ref, x1=v, y0=p, y1=p, line=dict(color=c, width=2), layer="below")
    fig.add_scatter(x=m[met], y=m["provincia"], mode="markers", showlegend=False,
                    marker=dict(size=12, color=colores, line=dict(color="white", width=2)),
                    hovertemplate="<b>%{y}</b><br>" + hover_valor(met, "x") + "<extra></extra>")
    fig.add_vline(x=ref, line_dash="dot", line_color=TENUE, annotation_text=f"mediana {valor(met, ref)}",
                  annotation_position="top", annotation_font=dict(color=TINTA_2, size=11))
    fig.update_xaxes(title=METRICAS[met][0])
    fig.update_yaxes(showgrid=False)
    return fig


# ---------------------------- Tarifas y composición -------------------------------------------
@app.callback(Output("g-composicion", "figure"), FILTROS)
def composicion(anio, prov, dist, grp):
    d = filtrar(anio, prov, dist, grp)
    if d.empty:
        return vacio()
    g = d.groupby("grupo_tarifa")[["clientes", "consumo_kwh", "monto_soles"]].sum()
    g = (g / g.sum() * 100).rename(columns={"clientes": "Clientes", "consumo_kwh": "Energía",
                                            "monto_soles": "Facturación"})
    fig = go.Figure()
    for grupo, color in zip(GRUPOS, CATEG):
        if grupo in g.index:
            fig.add_bar(y=g.columns, x=g.loc[grupo], name=grupo, orientation="h",
                        marker=dict(color=color, line=dict(color="white", width=2)),
                        texttemplate="%{x:.0f}%", textposition="inside", insidetextanchor="middle",
                        hovertemplate=f"{grupo}<br>%{{y}}: %{{x:.1f}}%<extra></extra>")
    fig.update_layout(barmode="stack", uniformtext=dict(minsize=10, mode="hide"), legend=dict(traceorder="normal"))
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=["Facturación", "Energía", "Clientes"])
    return fig


@app.callback(Output("g-precio", "figure"), FILTROS)
def precio(anio, prov, dist, grp):
    d = filtrar(anio, prov, dist, grp)
    d = d[d["precio_kwh"].between(0.01, 5) & (d["nivel_tension"] == "BT")]
    if d.empty:
        return vacio()
    orden = d.groupby("tarifa")["precio_kwh"].median().sort_values().index
    fig = px.box(d, x="precio_kwh", y="tarifa", points=False, category_orders={"tarifa": list(orden)},
                 labels={"precio_kwh": "S/ por kWh", "tarifa": ""})
    fig.update_traces(marker_color=AZUL, line_color=AZUL, fillcolor="rgba(29,95,166,0.18)")
    fig.update_yaxes(showgrid=False)
    return fig


@app.callback(Output("g-treemap", "figure"), FILTROS)
def treemap(anio, prov, dist, grp):
    d = filtrar(anio, prov, dist, grp)
    if d.empty:
        return vacio()
    m = indicadores(d, ["provincia", "distrito"]).fillna({"kwh_cliente_bt5b": 0})
    fig = px.treemap(m, path=[px.Constant("Puno"), "provincia", "distrito"], values="clientes",
                     color="kwh_cliente_bt5b", color_continuous_scale=SEQ,
                     range_color=(0, m["kwh_cliente_bt5b"].quantile(0.95)),
                     hover_data={"monto_soles": ":,.0f"}, labels={"kwh_cliente_bt5b": "kWh/cliente BT5B"})
    fig.update_traces(root_color=FONDO_BANDA, marker_line_color="white", marker_line_width=2)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), coloraxis_colorbar=dict(thickness=12, outlinewidth=0))
    return fig


# ---------------------------- Segmentos de distritos ------------------------------------------
@app.callback([Output("g-cluster-mapa", "figure"), Output("g-cluster-scatter", "figure"),
               Output("g-perfil", "figure")], [Input("f-provincia", "value"), Input("f-distrito", "value")])
def segmentos(prov, dist):
    s = ind_static
    if prov:
        s = s[s["provincia"].isin(prov)]
    if dist:
        s = s[s["ubigeo"].isin(dist)]
    orden = sorted(ind_static["cluster"].unique())
    cmap = dict(zip(orden, CATEG))
    f1 = px.choropleth(s, geojson=geo, locations="geo_id", featureidkey="properties.IDDIST", color="cluster",
                       color_discrete_map=cmap, category_orders={"cluster": orden}, hover_name="distrito")
    ajustar_mapa(f1)
    f1.update_layout(legend=dict(y=-0.02, yanchor="top", orientation="h"))
    f2 = px.scatter(s, x="clientes", y="kwh_cliente_bt5b", color="cluster", size="monto_soles", log_x=True,
                    size_max=40, hover_name="distrito", color_discrete_map=cmap, category_orders={"cluster": orden},
                    hover_data={"provincia": True, "pct_fose": ":.1f", "pct_mt_facturacion": ":.1f"},
                    labels={"clientes": "Clientes (escala log)", "kwh_cliente_bt5b": "kWh por cliente BT5B",
                            "cluster": ""})
    f2.update_traces(marker_line_color="white", marker_line_width=1.5)
    f2.update_layout(showlegend=False)

    # Perfil de segmentos: mediana real como texto, color = comparación con el promedio de los segmentos
    cols = {"clientes": "Clientes", "kwh_cliente_bt5b": "kWh/cliente BT5B", "pct_fose": "% ≤ 100 kWh",
            "pct_mt_facturacion": "% facturación MT", "densidad_clientes": "Clientes por km²"}
    perfil = ind_static.groupby("cluster")[list(cols)].median()
    z = (perfil - perfil.mean()) / perfil.std(ddof=0).replace(0, 1)
    n = ind_static["cluster"].value_counts()
    f3 = go.Figure(go.Heatmap(
        z=z.to_numpy(), x=list(cols.values()), y=[f"{c}  ({n[c]} distritos)" for c in perfil.index],
        text=perfil.map(lambda v: f"{v:,.1f}").to_numpy(), texttemplate="%{text}",
        textfont=dict(size=13, color=TINTA), colorscale=DIVERGENTE, zmid=0, xgap=3, ygap=3, showscale=False,
        hovertemplate="%{y}<br>%{x}: <b>%{text}</b><extra></extra>"))
    f3.update_xaxes(side="top", showgrid=False)
    f3.update_yaxes(showgrid=False, autorange="reversed")
    return f1, f2, f3


# ---------------------------- Ámbito de las pestañas analíticas -------------------------------
@app.callback([Output("d-ambito", "children"), Output("e-ambito", "children")], FILTROS)
def resumen_ambito(anio, prov, dist, grp):
    texto = " · ".join([describir_seleccion(prov, dist), describir_anios(anio),
                        ", ".join(grp) if grp else "todos los grupos tarifarios"])
    chip = [html.Span("Viendo", className="ambito-k"), html.Span(texto)]
    return chip, chip


# ---------------------------- Distribución y anomalías ----------------------------------------
def residencial(d):
    return d[(d["tarifa"] == "BT5B") & (d["consumo_kwh"] > 0)]


def categoria_alerta(tipo):
    if "Consumo atípico" in tipo:
        return "Consumo atípico"
    if "Precio atípico" in tipo:
        return "Precio atípico"
    return "Consumo sin facturación"


def con_alertas(d, tipos):
    """Columna 'categoria' (Normal o tipo de alerta); se descartan las alertas de tipos no elegidos."""
    d = d.assign(categoria=np.where(d["anomalia"], d["tipo_anomalia"].astype(str).map(categoria_alerta), "Normal"))
    return d[(d["categoria"] == "Normal") | d["categoria"].isin(tipos or [])]


def solo_alertas(d, tipos):
    a = con_alertas(d, tipos)
    return a[a["categoria"] != "Normal"]


def mediana_ponderada(r):
    """Mediana de kWh/cliente ponderada por clientes (cada registro agrupa varios hogares)."""
    if r.empty:
        return np.nan
    r = r.sort_values("kwh_cliente")
    acum = r["clientes"].cumsum().to_numpy()
    return r["kwh_cliente"].iloc[int(np.searchsorted(acum, acum[-1] / 2))]


def pct_hasta(r, umbral):
    return r.loc[r["kwh_cliente"] <= umbral, "clientes"].sum() / r["clientes"].sum() * 100 if len(r) else np.nan


def tasa_alerta(d, tipos):
    a = con_alertas(d, tipos)
    return (a["categoria"] != "Normal").mean() * 100 if len(a) else np.nan


@app.callback([Output("g-hist", "figure"), Output("d-tiles", "children")],
              FILTROS + [Input("d-umbral", "value"), Input("d-tipos", "value")])
def histograma(anio, prov, dist, grp, umbral_log, tipos):
    d_reg, d_sel = filtrar(anio, prov, dist, grp, territorio=False), filtrar(anio, prov, dist, grp)
    r, rs = residencial(d_reg), residencial(d_sel)
    hay_sel = bool(prov or dist)
    if len(r) < 20:
        return vacio("Se necesita la tarifa BT5B en el filtro"), []
    umbral, donde = 10 ** umbral_log, describir_seleccion(prov, dist)
    s, _, esc = stats.lognorm.fit(r["kwh_cliente"], floc=0)

    # Figura-fondo: la región en gris es el fondo, la selección en azul es la figura
    lx = np.log10(r["kwh_cliente"])
    bins = dict(start=lx.min(), end=lx.max(), size=(lx.max() - lx.min()) / 50)
    fig = go.Figure()
    fig.add_vrect(x0=lx.min(), x1=umbral_log, fillcolor=AZUL, opacity=0.06, line_width=0, layer="below")
    fig.add_histogram(x=lx, histnorm="probability density", xbins=bins, name="Región",
                      marker=dict(color=CONTEXTO if hay_sel else AZUL, line=dict(color="white", width=1)),
                      opacity=0.6 if hay_sel else 0.8, hovertemplate="%{y:.2f}<extra>Región</extra>")
    if hay_sel and len(rs):
        fig.add_histogram(x=np.log10(rs["kwh_cliente"]), histnorm="probability density", xbins=bins,
                          name=donde, opacity=0.85, marker=dict(color=FOCO[0], line=dict(color="white", width=1)),
                          hovertemplate="%{y:.2f}<extra>Selección</extra>")
    gx = np.linspace(lx.min(), lx.max(), 300)
    # Si X ~ LogNormal(μ, σ) entonces log10(X) ~ Normal(μ/ln10, σ/ln10)
    fig.add_scatter(x=gx, y=stats.norm.pdf(gx, np.log10(esc), s / np.log(10)), name="Modelo log-normal",
                    line=dict(color=AZUL_OSCURO, width=2), hoverinfo="skip")
    fig.add_vline(x=umbral_log, line_dash="dash", line_color=TINTA, line_width=1.5,
                  annotation_text=f"{umbral:,.0f} kWh" + (" · FOSE" if round(umbral) == 100 else ""),
                  annotation_font_color=TINTA, annotation_position="top right")
    ticks = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
    fig.update_xaxes(tickvals=np.log10(ticks), ticktext=[str(t) for t in ticks],
                     title="kWh por cliente al mes (escala log)")
    fig.update_layout(yaxis_title="densidad", barmode="overlay", bargap=0.02, hovermode="x unified")

    # Indicadores: valor de la selección y comparación con la región
    def frente_region(v, ref, **kw):
        return chip_vs(v, ref, **kw) if hay_sel else None

    base = rs if hay_sel else r
    med_sel, med_reg = mediana_ponderada(base), mediana_ponderada(r)
    pct_sel, pct_reg = pct_hasta(base, umbral), pct_hasta(r, umbral)
    tasa_sel, tasa_reg = tasa_alerta(d_sel, tipos), tasa_alerta(d_reg, tipos)
    alertas = solo_alertas(d_sel, tipos)
    tiles = [
        tile("Consumo típico de un hogar", f"{fmt(med_sel, ',.0f')} kWh", "mediana mensual",
             frente_region(med_sel, med_reg)),
        tile(f"Hogares ≤ {umbral:,.0f} kWh", f"{fmt(pct_sel, '.0f')} %", "según el umbral elegido",
             frente_region(pct_sel, pct_reg, puntos=True)),
        tile("Registros en alerta", f"{len(alertas):,}", f"S/ {compacto(alertas['monto_soles'].sum())} facturados",
             frente_region(tasa_sel, tasa_reg, sentido=-1, puntos=True)),
    ]
    return fig, tiles


@app.callback(Output("g-ic", "figure"), FILTROS + [Input("d-nivel", "value")])
def intervalos(anio, prov, dist, grp, nivel):
    d_ctx, _, focos = ambito(anio, prov, dist, grp, nivel)
    d = residencial(d_ctx)
    if d.empty:
        return vacio()
    rng = np.random.default_rng(0)
    filas = []
    for k, g in d.groupby(nivel):
        v, w = g["kwh_cliente"].to_numpy(), g["clientes"].to_numpy()
        idx = rng.integers(0, len(v), size=(500, len(v)))
        medias = (v[idx] * w[idx]).sum(1) / w[idx].sum(1)
        filas.append({"clave": k, "media": (v * w).sum() / w.sum(),
                      "lo": np.quantile(medias, .025), "hi": np.quantile(medias, .975)})
    m = pd.DataFrame(filas).sort_values("media")
    m["nombre"] = m["clave"].map(nombre_unidad)
    m["color"] = [color_provincia(k, focos) for k in m["clave"]]
    r = residencial(filtrar(anio, prov, dist, grp, territorio=False))
    media_reg = (r["kwh_cliente"] * r["clientes"]).sum() / r["clientes"].sum()
    fig = go.Figure()
    for color, sub in m.groupby("color", sort=False):
        fig.add_scatter(x=sub["media"], y=sub["nombre"], mode="markers", showlegend=False,
                        marker=dict(size=12, color=color, line=dict(color="white", width=2)),
                        error_x=dict(type="data", array=sub["hi"] - sub["media"],
                                     arrayminus=sub["media"] - sub["lo"], color=color, thickness=2, width=0),
                        customdata=sub[["clave", "lo", "hi"]].to_numpy(dtype=object),
                        hovertemplate="<b>%{y}</b><br>%{x:.1f} kWh/cliente<br>"
                                      "IC 95 %: %{customdata[1]:.1f} – %{customdata[2]:.1f}<extra></extra>")
    fig.add_vline(x=media_reg, line_dash="dot", line_color=TENUE, annotation_text=f"región {media_reg:.1f}",
                  annotation_font=dict(color=TINTA_2, size=11), annotation_position="bottom right")
    fig.update_xaxes(title="kWh por cliente BT5B")
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=m["nombre"])
    fig.update_layout(height=alto_filas(len(m)))
    return fig


@app.callback([Output("g-anom", "figure"), Output("g-alertas", "figure")],
              FILTROS + [Input("d-nivel", "value"), Input("d-tipos", "value")])
def anomalias(anio, prov, dist, grp, nivel, tipos):
    d = con_alertas(filtrar(anio, prov, dist, grp, territorio=False), tipos)
    if d.empty:
        return vacio(), vacio()
    d = d.assign(en_foco=d.index.isin(filtrar(anio, prov, dist, grp).index),
                 distrito=d["distrito"].str.title(), provincia_t=d["provincia"].str.title())

    # Dispersión consumo × precio (escala log): normal = fondo gris; alertas = color + símbolo
    p = d[(d["kwh_cliente"] > 0) & (d["precio_kwh"] > 0)]
    fig = go.Figure()
    normal = p[p["categoria"] == "Normal"]
    fig.add_scattergl(x=normal["kwh_cliente"], y=normal["precio_kwh"], mode="markers", name="Normal",
                      marker=dict(size=5, color=CONTEXTO, opacity=0.3), hoverinfo="skip")
    cols = ["ubigeo", "distrito", "provincia_t", "periodo_cod", "tarifa", "clientes", "tipo_anomalia"]
    hover = ("<b>%{customdata[1]}</b> (%{customdata[2]})<br>Periodo %{customdata[3]} · %{customdata[4]}<br>"
             "%{x:,.1f} kWh/cliente · S/ %{y:,.2f} por kWh<br>%{customdata[5]:,} clientes<br>"
             "%{customdata[6]}<extra></extra>")
    for cat, (color, simbolo) in ALERTAS.items():
        a = p[p["categoria"] == cat]
        for activo in (False, True):
            parte = a[a["en_foco"] == activo]
            if parte.empty:
                continue
            fig.add_scattergl(x=parte["kwh_cliente"], y=parte["precio_kwh"], mode="markers", name=cat,
                              legendgroup=cat, showlegend=activo,
                              marker=dict(size=10 if activo else 8, color=color, symbol=simbolo,
                                          opacity=0.95 if activo else 0.18, line=dict(color="white", width=1)),
                              customdata=parte[cols].to_numpy(dtype=object), hovertemplate=hover)
    fig.update_xaxes(type="log", dtick=1, title="kWh por cliente al mes (log)")
    fig.update_yaxes(type="log", dtick=1, title="Precio efectivo S/ por kWh (log)")

    # Prioridad: barras apiladas por provincia o distrito (las de más alertas + la selección)
    d_ctx, _, focos = ambito(anio, prov, dist, grp, nivel)
    a = solo_alertas(d_ctx, tipos)
    if a.empty:
        return fig, vacio("Sin alertas para los filtros seleccionados")
    t = a.groupby([nivel, "categoria"]).size().unstack(fill_value=0)
    total = t.sum(axis=1)
    claves = set(total.nlargest(MAX_UNIDADES).index) | (set(focos) & set(total.index))
    t = t.loc[total[list(claves)].sort_values().index]
    total, nombres = t.sum(axis=1), [nombre_unidad(k) for k in t.index]
    opac = [1 if (not focos or k in focos) else 0.3 for k in t.index]
    barras = go.Figure()
    for cat, (color, _) in ALERTAS.items():
        if cat in t:
            barras.add_bar(y=nombres, x=t[cat], name=cat, orientation="h", customdata=list(t.index),
                           marker=dict(color=color, opacity=opac, line=dict(color="white", width=2)),
                           hovertemplate="<b>%{y}</b><br>" + cat + ": %{x:,}<extra></extra>")
    barras.add_scatter(y=nombres, x=total, mode="text", text=[f"{v:,}" for v in total],
                       textposition="middle right", textfont=dict(color=TINTA_2, size=11),
                       showlegend=False, hoverinfo="skip")
    barras.update_layout(barmode="stack", bargap=0.25, legend=dict(traceorder="normal"), height=alto_filas(len(t)))
    barras.update_xaxes(range=[0, total.max() * 1.18], title="registros en alerta")
    barras.update_yaxes(showgrid=False)
    return fig, barras


@app.callback(Output("g-alertas-mes", "figure"), FILTROS + [Input("d-tipos", "value")])
def cronologia_alertas(anio, prov, dist, grp, tipos):
    d = filtrar(anio, prov, dist, grp)
    a = solo_alertas(d, tipos)
    if a.empty:
        return vacio("Sin alertas para los filtros seleccionados")

    # Cronología: barras apiladas por mes; cada año en una banda (Gestalt: región común)
    t = (a.groupby(["periodo", "categoria"]).size().unstack(fill_value=0)
         .reindex(sorted(d["periodo"].unique()), fill_value=0))
    fig = go.Figure()
    for cat, (color, _) in ALERTAS.items():
        if cat in t:
            fig.add_bar(x=t.index, y=t[cat], name=cat, marker=dict(color=color, line=dict(color="white", width=0.5)),
                        customdata=t.index.year, hovertext=[mes_es(f) for f in t.index],
                        hovertemplate="%{hovertext}<br>" + cat + ": %{y:,}<extra></extra>")
    inicio, fin = t.index.min() - pd.Timedelta(days=16), t.index.max() + pd.Timedelta(days=16)
    for i, a_ in enumerate(sorted(t.index.year.unique())):
        x0 = max(pd.Timestamp(a_, 1, 1) - pd.Timedelta(days=16), inicio)
        x1 = min(pd.Timestamp(a_, 12, 1) + pd.Timedelta(days=16), fin)
        fig.add_vrect(x0=x0, x1=x1, fillcolor=FONDO_BANDA if i % 2 == 0 else "white", line_width=0, layer="below",
                      annotation_text=f"<b>{a_}</b>", annotation_position="top left",
                      annotation_font=dict(color=TINTA_2, size=11))
    ejes_tiempo(fig, t.index)
    fig.update_xaxes(range=[inicio, fin], showgrid=False)
    fig.update_yaxes(title="registros en alerta", rangemode="tozero")
    fig.update_layout(barmode="stack", bargap=0.15, hovermode="x unified", legend=dict(traceorder="normal"))
    return fig


# ---------------------------- Evolución temporal ----------------------------------------------
def a_indice(tabla):
    """Índice por fila: promedio de los 3 primeros meses con dato = 100."""
    base = tabla.apply(lambda f: f.dropna().iloc[:3].mean(), axis=1)
    return tabla.div(base.replace(0, np.nan), axis=0) * 100


def variacion_mismos_meses(d, met, por):
    """Variación % por unidad entre el primer y el último año, usando solo los meses del último año."""
    a0, a1 = d["anio"].min(), d["anio"].max()
    meses = d.loc[d["anio"] == a1, "mes"].unique()
    v0 = indicadores(d[(d["anio"] == a0) & d["mes"].isin(meses)], [por]).set_index(por)[met]
    v1 = indicadores(d[d["anio"] == a1], [por]).set_index(por)[met]
    var = (v1 / v0 - 1) * 100
    return var.replace([np.inf, -np.inf], np.nan).dropna().sort_values(), a0, a1


@app.callback([Output("g-tendencia", "figure"), Output("g-calendario", "figure")],
              FILTROS + [Input("f-serie", "value")])
def tendencia(anio, prov, dist, grp, met):
    d = filtrar(anio, prov, dist, grp)
    if d.empty:
        return vacio(), vacio()
    s = por_mes(d, met)
    v = s.dropna()
    if v.empty:
        msg = "El indicador no aplica a los grupos tarifarios elegidos"
        return vacio(msg), vacio(msg)

    # Referencia: mismo mes del año anterior (se toma de todos los años, aunque se filtre uno)
    todo = por_mes(filtrar(None, prov, dist, grp), met)
    previo = todo.copy()
    previo.index = previo.index + pd.DateOffset(years=1)
    previo = previo[~previo.index.duplicated()].reindex(s.index)
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=previo, name="Año anterior", mode="lines",
                    line=dict(color=CONTEXTO, width=2, dash="dot"),
                    hovertemplate="año anterior: " + hover_valor(met) + "<extra></extra>")
    fig.add_scatter(x=s.index, y=s, name=describir_seleccion(prov, dist), mode="lines+markers",
                    line=dict(color=AZUL, width=2.5), marker=dict(size=6, color=AZUL),
                    hovertext=[mes_es(f) for f in s.index],
                    hovertemplate="<b>%{hovertext}</b>: " + hover_valor(met) + "<extra></extra>")
    for fecha, txt, ay in [(v.idxmax(), "máximo", -30), (v.idxmin(), "mínimo", 30)]:
        fig.add_annotation(x=fecha, y=v[fecha], text=f"{txt} · {mes_es(fecha)}", showarrow=True, arrowhead=0,
                           arrowcolor=TENUE, ax=0, ay=ay, font=dict(size=11, color=TINTA_2))
    fig.add_annotation(x=v.index[-1], y=v.iloc[-1], text=f"<b>{valor(met, v.iloc[-1])}</b>", showarrow=False,
                       xanchor="left", xshift=10, font=dict(size=12, color=TINTA))
    ejes_tiempo(fig, s.index)
    fig.update_xaxes(hoverformat="%m-%Y", rangeselector=dict(
        x=0, y=1.02, yanchor="bottom", bgcolor=FONDO_BANDA, activecolor=AZUL_SUAVE, font=dict(size=11),
        buttons=[dict(count=12, label="12 meses", step="month", stepmode="backward"),
                 dict(count=24, label="24 meses", step="month", stepmode="backward"),
                 dict(step="all", label="Todo")]))
    fig.update_yaxes(title=METRICAS[met][0], tickformat=",.3s")
    fig.update_layout(hovermode="x unified", legend=dict(x=1, xanchor="right"), margin=dict(l=10, r=90, t=40, b=10))

    # Calendario: meses en filas, años en columnas (Gestalt: proximidad; cada columna es un año)
    cal = (pd.DataFrame({"anio": v.index.year, "mes": v.index.month, "v": v.to_numpy()})
           .pivot(index="mes", columns="anio", values="v").reindex(index=range(1, 13)))
    z = cal.to_numpy()
    cal_fig = go.Figure(go.Heatmap(   # x = año como texto: el clic lo devuelve para filtrar
        z=z, x=[str(a) for a in cal.columns], y=MESES, colorscale=SEQ, xgap=3, ygap=3, showscale=False,
        text=[[compacto(x) for x in fila] for fila in z], texttemplate="%{text}", textfont=dict(size=11),
        hoverongaps=False,
        hovertemplate="<b>%{y} %{x}</b><br>" + hover_valor(met, "z") + "<extra></extra>"))
    cal_fig.update_xaxes(showgrid=False, side="top", type="category")
    cal_fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig, cal_fig


def fig_multiples(tabla, focos, vista):
    """Small multiples (Gestalt: proximidad + semejanza): un panel por unidad, ordenados de mayor a menor."""
    indice = vista == "indice"
    serie = a_indice(tabla) if indice else tabla
    claves, cols = list(tabla.index), 4
    filas = -(-len(claves) // cols)
    alto = 135 * filas + 50
    titulos = [f"<b>{nombre_unidad(k)}</b>" if k in focos else nombre_unidad(k) for k in claves]
    f = make_subplots(rows=filas, cols=cols, shared_xaxes="all", shared_yaxes="all" if indice else False,
                      subplot_titles=titulos, vertical_spacing=min(0.15, 60 / alto),
                      horizontal_spacing=0.025 if indice else 0.045)
    fechas = [mes_es(x) for x in serie.columns]
    for i, k in enumerate(claves):
        fila, col = i // cols + 1, i % cols + 1
        color = color_provincia(k, focos)
        custom = np.array([[k, x, t] for x, t in zip(tabla.loc[k], fechas)], dtype=object)
        f.add_scatter(x=serie.columns, y=serie.loc[k], mode="lines", row=fila, col=col, showlegend=False,
                      line=dict(color=color, width=2.6 if k in focos else 1.8), customdata=custom,
                      hovertemplate="<b>%{customdata[2]}</b><br>valor %{customdata[1]:,.1f}"
                                    + ("<br>índice %{y:.0f}" if indice else "")
                                    + f"<extra>{nombre_unidad(k)}</extra>")
        ultimo = serie.loc[k].dropna()
        if len(ultimo):   # punto final: dónde termina cada trayectoria
            f.add_scatter(x=[ultimo.index[-1]], y=[ultimo.iloc[-1]], mode="markers", row=fila, col=col,
                          showlegend=False, hoverinfo="skip", marker=dict(size=6, color=color))
    if indice:
        f.add_hline(y=100, line_dash="dot", line_color=TENUE, line_width=1, row="all", col="all")
    f.update_annotations(font=dict(size=12, color=TINTA))
    f.update_xaxes(dtick="M12", tickformat="%Y", showgrid=False)
    f.update_yaxes(gridcolor=REJILLA, nticks=4, tickformat=".0f" if indice else ".3s")
    for k in range(len(claves), filas * cols):   # paneles vacíos de la última fila
        f.update_xaxes(visible=False, row=k // cols + 1, col=k % cols + 1)
        f.update_yaxes(visible=False, row=k // cols + 1, col=k % cols + 1)
    f.update_layout(height=alto, margin=dict(l=10, r=10, t=30, b=10))
    return f


def fig_crecimiento(var, a0, a1, focos):
    """Lollipop centrado en 0 (Gestalt conexión: el tallo une el cero con el valor)."""
    if a0 == a1 or var.empty:
        return vacio("Selecciona al menos dos años")
    nombres, colores = [nombre_unidad(k) for k in var.index], [color_provincia(k, focos) for k in var.index]
    f = go.Figure()
    for n, v, c in zip(nombres, var, colores):
        f.add_shape(type="line", x0=0, x1=v, y0=n, y1=n, line=dict(color=c, width=2), layer="below")
    f.add_scatter(x=var, y=nombres, mode="markers+text", showlegend=False, customdata=list(var.index),
                  marker=dict(size=12, color=colores, line=dict(color="white", width=2)),
                  text=[f"{v:+.1f}%" for v in var],
                  textposition=["middle left" if v < 0 else "middle right" for v in var],
                  textfont=dict(color=TINTA_2, size=11), cliponaxis=False,
                  hovertemplate="<b>%{y}</b><br>%{x:+.1f}% " + f"({a1} vs {a0})<extra></extra>")
    f.add_vline(x=0, line_color=CONTEXTO)
    lim = max(abs(var.min()), abs(var.max())) * 1.3
    f.update_xaxes(range=[min(-lim * 0.25, var.min() * 1.9), lim], ticksuffix="%", title=f"{a1} vs {a0}")
    f.update_yaxes(showgrid=False, categoryorder="array", categoryarray=nombres)
    f.update_layout(height=alto_filas(len(var)))
    return f


@app.callback([Output("g-multiples", "figure"), Output("g-pendiente", "figure"), Output("e-tiles", "children")],
              FILTROS + [Input("f-serie", "value"), Input("e-vista", "value"), Input("e-nivel", "value")])
def evolucion(anio, prov, dist, grp, met, vista, nivel):
    d_ctx, d_sel, focos = ambito(anio, prov, dist, grp, nivel)
    if d_ctx.empty or d_ctx["periodo"].nunique() < 4:
        msg = "Selecciona un rango con más meses para ver la evolución"
        return vacio(msg), vacio(msg), []
    tabla = por_mes(d_ctx, met, nivel).dropna(how="all")
    if tabla.empty:
        msg = "El indicador no aplica a los grupos tarifarios elegidos"
        return vacio(msg), vacio(msg), []
    tabla = tabla.loc[tabla.mean(axis=1).sort_values(ascending=False).index]
    var, a0, a1 = variacion_mismos_meses(d_ctx, met, nivel)

    # Indicadores de la selección (tres, breves)
    total = por_mes(d_sel, met).dropna()
    tiles = []
    if len(total):
        ult = total.index[-1]
        hace_un_ano = ult - pd.DateOffset(years=1)
        todos = por_mes(filtrar(None, prov, dist, grp), met)   # la referencia puede estar fuera del año filtrado
        previo = todos.get(hace_un_ano, np.nan)
        var_a = (total.iloc[-1] / previo - 1) * 100 if pd.notna(previo) and previo else np.nan
        tiles = [tile("Variación interanual", "–" if pd.isna(var_a) else f"{var_a:+.1f} %",
                      f"{mes_es(ult)} vs {mes_es(hace_un_ano)}"),
                 tile("Mes más alto", mes_es(total.idxmax()), valor(met, total.max()))]
    if not var.empty and a0 != a1:
        unidad = "Provincia" if nivel == "provincia" else "Distrito"
        tiles.append(tile(f"{unidad} que más crece", nombre_unidad(var.index[-1]),
                          f"{var.iloc[-1]:+.1f} % · menor: {nombre_unidad(var.index[0])} {var.iloc[0]:+.1f} %"))
    return fig_multiples(tabla, focos, vista), fig_crecimiento(var, a0, a1, focos), tiles


# ---------------------------- Conclusiones ----------------------------------------------------
# Cada hallazgo = (área, cifra, lectura breve, decisión). AGENDA fija su orden de prioridad y plazo.
AGENDA = {"Control": (1, "alta", "Inmediato"), "Demanda": (2, "alta", "Antes del pico"),
          "Crecimiento": (3, "media", "6–12 meses"), "Ingresos": (4, "media", "Permanente"),
          "Hogares": (5, "media", "6–12 meses"), "Brecha territorial": (6, "estrategica", "Largo plazo")}


def fila_hallazgo(area, cifra, lectura, decision):
    """Una fila alineada: hallazgo (cifra + lectura) → decisión → plazo (Gestalt: alineación y continuidad)."""
    _, clase, plazo = AGENDA[area]
    return html.Div([
        html.Div([html.Div(area, className="hz-area"),
                  html.Div([html.Span(cifra, className="hz-cifra"), html.Span(lectura, className="hz-lectura")])],
                 className="hz-hallazgo"),
        html.Div(decision, className="hz-decision"),
        html.Span(plazo, className=f"hz-plazo {clase}"),
    ], className=f"hz-fila {clase}")


def conclusion_general(ideas, prov, dist, anio):
    """Una sola frase: qué pasa (cifras clave) y qué hacer primero."""
    donde = describir_seleccion(prov, dist)
    por_area = {i[0]: i for i in ideas}
    hechos = []
    if "Demanda" in por_area:
        cifra = por_area["Demanda"][1]
        hechos.append(f"la demanda {'cae' if cifra.startswith('-') else 'crece'} {cifra.lstrip('+-')}")
    if "Ingresos" in por_area:
        hechos.append(f"el {por_area['Ingresos'][1]} del ingreso depende de 10 distritos")
    if "Control" in por_area:
        hechos.append(f"hay {por_area['Control'][1]} registros en alerta")
    prioridades = " y ".join(i[3].rstrip(".")[0].lower() + i[3].rstrip(".")[1:] for i in ideas[:2])
    frase = f"En {donde.lower() if donde == 'Toda la región' else donde}, {lista(hechos)}." if hechos else ""
    return [html.Div([html.Span("CONCLUSIÓN GENERAL", className="cg-k"),
                      html.Span(f"{donde} · {describir_anios(anio)}", className="cg-ambito")], className="cg-cabecera"),
            html.P(frase, className="cg-frase"),
            html.P([html.B("Prioridad: "), prioridades + "."], className="cg-prioridad")]


def lista(nombres):
    nombres = list(nombres)
    return ", ".join(nombres[:-1]) + " y " + nombres[-1] if len(nombres) > 1 else "".join(nombres)


def concl_demanda(d):
    a0, a1 = d["anio"].min(), d["anio"].max()
    if a0 == a1:
        return None
    meses = set(d.loc[d["anio"] == a0, "mes"]) & set(d.loc[d["anio"] == a1, "mes"])
    e0, e1 = (d.loc[(d["anio"] == a) & d["mes"].isin(meses), "consumo_kwh"].sum() for a in (a0, a1))
    s = por_mes(d, "consumo_kwh").dropna()
    pico = MESES_LARGO[s.groupby(s.index.month).mean().idxmax() - 1]
    return ("Demanda", f"{(e1 / e0 - 1) * 100:+.1f} %", f"energía {a1} vs {a0}; pico en {pico}",
            f"Asegurar energía y cuadrillas para {pico}.")


def concl_concentracion(d):
    u = d[d["periodo_cod"] == d["periodo_cod"].max()]
    if u["ubigeo"].nunique() <= 10:
        return None
    k = kpis.calcular_kpis(u)
    return ("Ingresos", f"{k['pct_top10']:.0f} %", "de la facturación en solo 10 distritos",
            "Fidelizar a los grandes clientes.")


def concl_hogares(d, k):
    r = residencial(d)
    if r.empty:
        return None
    lectura = "consumo típico de un hogar al mes"
    if k["pct_bajo_consumo"] >= 0.5:
        lectura += f"; {k['pct_bajo_consumo']:.0f} % usa ≤ 30 kWh"
    return ("Hogares", f"{mediana_ponderada(r):.0f} kWh", lectura, "Focalizar subsidios y revisar medidores.")


def concl_brecha(d, nivel):
    m = indicadores(d, [nivel])
    m = m[m["clientes"] >= (500 if nivel == "ubigeo" else 0)].dropna(subset=["kwh_cliente_bt5b"])
    if len(m) < 2:
        return None
    alto, bajo = m.loc[m["kwh_cliente_bt5b"].idxmax()], m.loc[m["kwh_cliente_bt5b"].idxmin()]
    return ("Brecha territorial", f"{alto['kwh_cliente_bt5b'] / bajo['kwh_cliente_bt5b']:.1f}×",
            f"consumo por hogar de {nombre_unidad(alto[nivel])} frente a {nombre_unidad(bajo[nivel])}",
            "Impulsar usos productivos en zonas rurales.")


def concl_crecimiento(d, nivel):
    var, a0, a1 = variacion_mismos_meses(d, "consumo_kwh", nivel)
    if a0 == a1 or len(var) < 2:
        return None
    lider = nombre_unidad(var.index[-1])
    return ("Crecimiento", f"{var.iloc[-1]:+.1f} %", f"en {lider}, {'la provincia' if nivel == 'provincia' else 'el distrito'} "
            "que más crece", f"Ampliar la capacidad de red en {lider}.")


def concl_control(d, nivel):
    a = solo_alertas(d, list(ALERTAS))
    if a.empty:
        return None
    focos = lista(nombre_unidad(f) for f in a.groupby(nivel).size().nlargest(3).index)
    monto = a["monto_soles"].sum()
    return ("Control", f"{len(a):,}", "registros en alerta" + (f" (S/ {compacto(monto)})" if monto > 0 else ""),
            f"Inspeccionar {focos}.")


@app.callback([Output("concl-general", "children"), Output("conclusiones", "children")], FILTROS)
def conclusiones(anio, prov, dist, grp):
    d = filtrar(anio, prov, dist, grp)
    sin_datos = html.P("No hay hallazgos para los filtros seleccionados; amplía el periodo o el territorio.",
                       className="nota")
    if d.empty:
        return [], sin_datos
    nivel = "ubigeo" if prov or dist else "provincia"
    k = kpis.calcular_kpis(d)
    ideas = [concl_demanda(d), concl_concentracion(d), concl_hogares(d, k), concl_brecha(d, nivel),
             concl_crecimiento(d, nivel), concl_control(d, nivel)]
    ideas = sorted((i for i in ideas if i), key=lambda i: AGENDA[i[0]][0])
    if not ideas:
        return [], sin_datos
    cabecera = html.Div([html.Span("Hallazgo"), html.Span("Decisión"), html.Span("Plazo")], className="hz-cabecera")
    return conclusion_general(ideas, prov, dist, anio), [cabecera] + [fila_hallazgo(*i) for i in ideas]


@app.callback(Output("descarga", "data"), Input("btn-descarga", "n_clicks"),
              [State(c.component_id, c.component_property) for c in FILTROS], prevent_initial_call=True)
def descargar(n, anio, prov, dist, grp):
    buf = io.StringIO()
    filtrar(anio, prov, dist, grp).to_csv(buf, index=False)
    return dict(content=buf.getvalue(), filename="electropuno_filtrado.csv")


if __name__ == "__main__":
    app.run(debug=True)
