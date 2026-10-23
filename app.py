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
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update
from scipy import stats

import kpis

# ----------------------------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------------------------
DATA = Path(__file__).parent / "data" / "processed"
df_all = pd.read_parquet(DATA / "consumo_final.parquet")
df_all["geo_id"] = df_all["geo_id"].astype(str)
ind_static = pd.read_csv(DATA / "indicadores_distrito.csv", dtype={"ubigeo": str, "geo_id": str})
geo = json.load(open(DATA / "puno_distritos.geojson", encoding="utf-8"))
resumen_dist = json.load(open(DATA / "distribuciones_resumen.json", encoding="utf-8"))

# Nombre que se muestra en cada polígono (San Miguel se dibuja junto con Juliaca)
nombre_geo = (df_all.groupby("geo_id")["distrito"].unique()
              .apply(lambda v: " + ".join(sorted(v))).to_dict())
PERIODOS = sorted(df_all["periodo_cod"].unique())
PROVINCIAS = sorted(df_all["provincia"].unique())
GRUPOS = ["BT5B residencial", "BT5 otras", "BT potencia", "Media tensión"]
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# ----------------------------------------------------------------------------------------------
# Estilo (paleta validada: secuencial azul para magnitudes, 3 categóricas para segmentos)
# ----------------------------------------------------------------------------------------------
AZUL = "#2a78d6"
CATEG = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ROJO = "#e34948"
pio.templates["puno"] = go.layout.Template(
    layout=dict(font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#0b0b0b"),
                paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", colorway=CATEG,
                margin=dict(l=10, r=10, t=48, b=10), title=dict(font=dict(size=15), x=0.01),
                xaxis=dict(gridcolor="#ececea", zerolinecolor="#d6d5d0", linecolor="#d6d5d0", automargin=True),
                yaxis=dict(gridcolor="#ececea", zerolinecolor="#d6d5d0", linecolor="#d6d5d0", automargin=True),
                hoverlabel=dict(bgcolor="white", font_size=12), legend=dict(orientation="h", y=-0.15)))
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


# ----------------------------------------------------------------------------------------------
# Funciones de cálculo
# ----------------------------------------------------------------------------------------------
def filtrar(periodos, provincias, grupos):
    d = df_all
    if periodos:
        d = d[d["periodo_cod"].isin(periodos)]
    if provincias:
        d = d[d["provincia"].isin(provincias)]
    if grupos:
        d = d[d["grupo_tarifa"].isin(grupos)]
    return d


def indicadores(d: pd.DataFrame, by: list) -> pd.DataFrame:
    """Mismos indicadores del Notebook 2 (totales = promedio mensual)."""
    n = max(d["periodo_cod"].nunique(), 1)
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


def ajustar_mapa(fig):
    """Mapa SVG (no requiere internet ni WebGL): proyección Mercator ajustada a los distritos."""
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
    fig.update_traces(marker_line_width=0.6, marker_line_color="white")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), dragmode=False)
    return fig


def vacio(msg="Sin datos para los filtros seleccionados"):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="#52514e"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def fmt(v, f=",.0f"):
    return "–" if v is None or pd.isna(v) else format(v, f)


# ----------------------------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------------------------
def tarjeta(hijos, clase=""):
    return html.Div(hijos, className=f"tarjeta {clase}")


app = Dash(__name__, title="Electro Puno · Dashboard de consumo")
server = app.server

app.layout = html.Div([
    html.Header([
        html.Div([html.H1("Consumo de energía eléctrica · Electro Puno S.A.A."),
                  html.P("Prototipo de visualización para la toma de decisiones · Datos Abiertos Perú")]),
        html.Button("⬇ Descargar datos filtrados (CSV)", id="btn-descarga", className="boton"),
        dcc.Download(id="descarga"),
    ], className="cabecera"),

    # Filtros (una sola fila sobre los gráficos)
    html.Div([
        html.Div([html.Label("Periodo comercial"),
                  dcc.Dropdown(id="f-periodo", multi=True, value=PERIODOS,
                               options=[{"label": f"{str(p)[:4]}-{str(p)[4:]}", "value": p} for p in PERIODOS])],
                 className="filtro"),
        html.Div([html.Label("Provincia"),
                  dcc.Dropdown(id="f-provincia", multi=True, placeholder="Todas las provincias",
                               options=PROVINCIAS)], className="filtro ancho"),
        html.Div([html.Label("Grupo tarifario"),
                  dcc.Checklist(id="f-grupo", options=GRUPOS, value=GRUPOS, inline=True,
                                className="checklist")], className="filtro ancho"),
        html.Div([html.Label("Indicador del mapa"),
                  dcc.Dropdown(id="f-metrica", value="kwh_cliente_bt5b", clearable=False,
                               options=[{"label": v[0], "value": k} for k, v in METRICAS.items()])],
                 className="filtro ancho"),
    ], className="filtros"),

    dcc.Tabs(className="tabs", children=[
        dcc.Tab(label="🎯 Resumen gerencial", children=[
            html.Div(id="resumen-periodo", className="periodo"),
            html.Div(id="kpis-principales", className="kpis-principales"),
            html.Details([
                html.Summary("Ver los otros 8 indicadores"),
                html.Div(id="kpis-complementarios", className="kpis-complementarios"),
            ], className="kpis-detalles"),
            html.Div([
                html.Div([
                    html.Div("TENDENCIA DEL SISTEMA", className="seccion-etiqueta"),
                    html.H2("Consumo de energía por periodo"),
                    html.P("¿Está subiendo o bajando el consumo? Cada punto representa la energía total del mes.",
                           className="nota"),
                ], className="seccion-cabecera"),
                dcc.Graph(id="g-resumen-consumo", style={"height": "430px"}),
            ], className="tarjeta grafico-principal"),
            html.Div([
                tarjeta(dcc.Graph(id="g-estacional", style={"height": "350px"}), "col-4"),
                tarjeta(dcc.Graph(id="g-resumen-provincia", style={"height": "350px"}), "col-4"),
                tarjeta(dcc.Graph(id="g-resumen-tarifa", style={"height": "350px"}), "col-4"),
            ], className="fila"),
            html.Div([
                tarjeta(dcc.Graph(id="g-resumen-facturacion", style={"height": "340px"}), "col-6"),
                tarjeta(dcc.Graph(id="g-resumen-clientes", style={"height": "340px"}), "col-6"),
            ], className="fila"),
            html.Div([
                html.Strong("Comparativa de instalaciones nuevas y antiguas pendiente de datos de altas. "),
                "El consumo disponible está agregado y no contiene identificador de suministro ni fecha de instalación.",
            ], className="aviso-datos"),
            html.Div([
                tarjeta([html.H3("Hallazgos clave para la Gerencia"), html.Ul(id="hallazgos", className="hallazgos")],
                        "col-5"),
                tarjeta(dcc.Graph(id="g-concentracion", style={"height": "430px"}), "col-7"),
            ], className="fila"),
            html.Div([tarjeta(html.Details([
                html.Summary("📘 Ficha técnica de los KPIs (definición y fórmula)"),
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
                    style_header={"fontWeight": 600, "backgroundColor": "#f3f3f1"}),
                html.P("Con datos abiertos no es posible medir pérdidas de energía, morosidad ni calidad de servicio "
                       "(SAIDI/SAIFI); requieren información interna de la empresa.", className="nota"),
            ]), "col-12")], className="fila"),
        ]),
        dcc.Tab(label="🗺 Territorio", children=[
            html.Div([
                tarjeta([dcc.Graph(id="g-mapa", style={"height": "560px"}),
                         html.P("Haz clic en un distrito para ver su ficha. San Miguel se muestra junto con "
                                "Juliaca (no tiene polígono propio en el GeoJSON).", className="nota")], "col-8"),
                tarjeta([html.Div(id="ficha"), dcc.Graph(id="g-ficha-tarifa", style={"height": "300px"})], "col-4"),
            ], className="fila"),
            html.Div([tarjeta(dcc.Graph(id="g-top", style={"height": "460px"}), "col-6"),
                      tarjeta(dcc.Graph(id="g-provincia", style={"height": "460px"}), "col-6")], className="fila"),
        ]),
        dcc.Tab(label="⚡ Tarifas y composición", children=[
            html.Div([tarjeta(dcc.Graph(id="g-composicion", style={"height": "430px"}), "col-6"),
                      tarjeta(dcc.Graph(id="g-precio", style={"height": "430px"}), "col-6")], className="fila"),
            html.Div([tarjeta(dcc.Graph(id="g-treemap", style={"height": "560px"}), "col-12")], className="fila"),
        ]),
        dcc.Tab(label="🧩 Segmentos de distritos", children=[
            html.Div([
                tarjeta([dcc.Graph(id="g-cluster-mapa", style={"height": "520px"})], "col-6"),
                tarjeta([dcc.Graph(id="g-cluster-scatter", style={"height": "520px"})], "col-6"),
            ], className="fila"),
            html.Div([tarjeta([html.H3("Perfil promedio de cada segmento (K-Means, Notebook 2)"),
                               html.Div(id="tabla-cluster")], "col-12")], className="fila"),
        ]),
        dcc.Tab(label="📊 Distribución y anomalías", children=[
            html.Div([tarjeta([dcc.Graph(id="g-hist", style={"height": "430px"}), html.Div(id="txt-ajuste",
                                                                                         className="nota")], "col-7"),
                      tarjeta(dcc.Graph(id="g-ic", style={"height": "470px"}), "col-5")], className="fila"),
            html.Div([tarjeta([html.H3("Registros marcados para revisión"),
                               html.P("Consumo atípico: fuera del intervalo P0.5–P99.5 del modelo log-normal. "
                                      "Precio atípico: z-score robusto |z| > 3.5 dentro de cada tarifa de baja tensión.",
                                      className="nota"),
                               html.Div(id="tabla-anom")], "col-12")], className="fila"),
        ]),
        dcc.Tab(label="📈 Evolución temporal", children=[
            html.Div([tarjeta([dcc.RadioItems(id="f-serie", value="consumo_kwh", inline=True, className="radio",
                                              options=[{"label": METRICAS[k][0], "value": k} for k in
                                                       ["clientes", "consumo_kwh", "monto_soles", "kwh_cliente_bt5b"]]),
                               dcc.Graph(id="g-serie", style={"height": "480px"})], "col-12")], className="fila"),
        ]),
    ]),
    html.Footer("Fuente: Electro Puno S.A.A. – Plataforma Nacional de Datos Abiertos. "
                "Límites distritales: INEI (peru_distrital_simple.geojson)."),
], className="contenedor")

FILTROS = [Input("f-periodo", "value"), Input("f-provincia", "value"), Input("f-grupo", "value")]


# ----------------------------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------------------------
COLOR_BUENO, COLOR_MALO, COLOR_NEUTRO = "#1a7f37", "#b42318", "#52514e"


@app.callback(Output("g-resumen-consumo", "figure"), FILTROS)
def resumen_consumo(per, prov, grp):
    d = filtrar(per, prov, grp)
    if d.empty:
        return vacio()
    m = d.groupby("periodo", as_index=False)["consumo_kwh"].sum().sort_values("periodo")
    calendario = pd.date_range(m["periodo"].min(), m["periodo"].max(), freq="MS")
    serie = m.set_index("periodo")["consumo_kwh"].reindex(calendario) / 1000
    fig = go.Figure(go.Scatter(
        x=serie.index, y=serie, mode="lines+markers", name="Energía",
        line=dict(color=AZUL, width=3), marker=dict(color="#ffffff", size=7, line=dict(color=AZUL, width=2)),
        fill="tozeroy", fillcolor="rgba(42,120,214,0.10)",
        hovertemplate="%{x|%b %Y}<br>%{y:,.0f} MWh<extra></extra>",
        connectgaps=False,
    ))
    fig.update_layout(
        title="Energía facturada mensual",
        yaxis_title="MWh", xaxis_title="",
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=16, r=20, t=48, b=12),
    )
    fig.update_xaxes(dtick="M3", tickformat="%b\n%Y")
    return fig


@app.callback(Output("g-estacional", "figure"), FILTROS)
def perfil_estacional(per, prov, grp):
    d = filtrar(per, prov, grp)
    if d.empty:
        return vacio()
    mensual = d.groupby(["anio", "mes"], as_index=False)["consumo_kwh"].sum()
    mensual["consumo_mwh"] = mensual["consumo_kwh"] / 1000
    mensual["mes_nombre"] = mensual["mes"].map(lambda mes: MESES[int(mes) - 1])
    fig = px.line(mensual, x="mes_nombre", y="consumo_mwh", color="anio", markers=True,
                  category_orders={"mes_nombre": MESES},
                  labels={"mes_nombre": "Mes", "consumo_mwh": "Energía (MWh)", "anio": "Año"},
                  title="Estacionalidad · compara el mismo mes entre años")
    fig.update_traces(line=dict(width=2.5), marker=dict(size=7),
                      hovertemplate="%{x}<br>%{y:,.0f} MWh<extra></extra>")
    fig.update_yaxes(tickformat=",.0f", rangemode="tozero")
    return fig


@app.callback(Output("g-resumen-provincia", "figure"), FILTROS)
def resumen_provincia(per, prov, grp):
    d = filtrar(per, prov, grp)
    if d.empty:
        return vacio()
    m = indicadores(d, ["provincia"]).sort_values("consumo_kwh")
    m["energia_mwh"] = m["consumo_kwh"] / 1000
    fig = px.bar(m, x="energia_mwh", y="provincia", orientation="h",
                 labels={"energia_mwh": "MWh/mes promedio", "provincia": ""},
                 title="Consumo promedio mensual por provincia")
    fig.update_traces(marker_color=AZUL, marker_cornerradius=4,
                      hovertemplate="%{y}<br>%{x:,.0f} MWh/mes<extra></extra>")
    fig.update_xaxes(tickformat=",.0f")
    return fig


@app.callback(Output("g-resumen-tarifa", "figure"), FILTROS)
def resumen_tarifa(per, prov, grp):
    d = filtrar(per, prov, grp)
    if d.empty:
        return vacio()
    n = max(d["periodo_cod"].nunique(), 1)
    m = (d.groupby("grupo_tarifa", as_index=False)["consumo_kwh"].sum()
         .sort_values("consumo_kwh"))
    m["energia_mwh"] = m["consumo_kwh"] / n / 1000
    fig = px.bar(m, x="energia_mwh", y="grupo_tarifa", orientation="h",
                 color="grupo_tarifa", color_discrete_map=dict(zip(GRUPOS, CATEG)),
                 category_orders={"grupo_tarifa": GRUPOS},
                 labels={"energia_mwh": "MWh/mes promedio", "grupo_tarifa": ""},
                 title="Consumo promedio por grupo tarifario")
    fig.update_traces(marker_line_color="white", marker_line_width=1,
                      hovertemplate="%{y}<br>%{x:,.0f} MWh/mes<extra></extra>")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickformat=",.0f")
    return fig


@app.callback(Output("g-resumen-facturacion", "figure"), FILTROS)
def resumen_facturacion(per, prov, grp):
    d = filtrar(per, prov, grp)
    if d.empty:
        return vacio()
    m = d.groupby("periodo", as_index=False)["monto_soles"].sum().sort_values("periodo")
    fig = px.bar(m, x="periodo", y="monto_soles",
                 labels={"periodo": "", "monto_soles": "Facturación (S/)"},
                 title="Facturación total por mes")
    fig.update_traces(marker_color="#1baf7a", marker_cornerradius=4,
                      hovertemplate="%{x|%b %Y}<br>S/ %{y:,.0f}<extra></extra>")
    fig.update_xaxes(dtick="M3", tickformat="%b\n%Y")
    fig.update_yaxes(tickformat=",.3s", rangemode="tozero")
    return fig


@app.callback(Output("g-resumen-clientes", "figure"), FILTROS)
def resumen_clientes(per, prov, grp):
    d = filtrar(per, prov, grp)
    if d.empty:
        return vacio()
    m = d.groupby("periodo", as_index=False)["clientes"].sum().sort_values("periodo")
    m["variacion_pct"] = m["clientes"].pct_change() * 100
    m["variacion_pct"] = m["variacion_pct"].fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=m["periodo"], y=m["clientes"], name="Suministros",
        mode="lines+markers", yaxis="y",
        line=dict(color="#eda100", width=2.5),
        marker=dict(color="#eda100", size=6),
        hovertemplate="%{x|%b %Y}<br>%{y:,.0f} suministros<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=m["periodo"], y=m["variacion_pct"], name="Variación mensual",
        yaxis="y2", marker_color="#2a78d6", opacity=0.55,
        hovertemplate="%{x|%b %Y}<br>Variación: %{y:+.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Suministros atendidos · total y variación mensual",
        xaxis=dict(title="", dtick="M3", tickformat="%b\n%Y"),
        yaxis=dict(title="Suministros", tickformat=",.3s", rangemode="tozero"),
        yaxis2=dict(title="Variación (%)", overlaying="y", side="right", ticksuffix="%",
                    showgrid=False, zeroline=True, zerolinecolor="#cbd5e1"),
        hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=16, r=55, t=48, b=40),
    )
    return fig


def etiqueta_periodo(p):
    return f"{str(p)[:4]}-{str(p)[4:]}"


def tarjeta_kpi(k, valor, var, previo):
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
        html.Div(f"{prefijo}{fmt(valor, k['fmt'])}{sufijo}", className="kpi-valor"),
        html.Div(k["unidad"], className="kpi-sub"),
        delta,
    ], className="kpi", title=f"{k['pregunta']} Fórmula: {k['formula']}")


@app.callback([Output("resumen-periodo", "children"), Output("kpis-principales", "children"),
               Output("kpis-complementarios", "children"),
               Output("hallazgos", "children"), Output("g-concentracion", "figure")], FILTROS)
def resumen_gerencial(per, prov, grp):
    d = filtrar(per, prov, grp)
    if d.empty:
        return "Sin datos para los filtros seleccionados", [], [], [], vacio()
    periodos = sorted(d["periodo_cod"].unique())
    actual, previo = periodos[-1], (periodos[-2] if len(periodos) > 1 else None)
    tabla = kpis.tabla_kpis(d)
    ambito = ", ".join(prov) if prov else "toda la concesión (13 provincias)"
    cab = [html.B(f"Periodo analizado: {etiqueta_periodo(actual)}"),
           f" · comparado con {etiqueta_periodo(previo)}" if previo else " · carga 2+ meses para ver la variación",
           f" · Ámbito: {ambito}"]

    indices_kpi = {k["id"]: i for i, k in enumerate(kpis.CATALOGO)}
    principales = ["energia_mwh", "clientes", "facturacion", "kwh_residencial"]
    tarjetas = [
        tarjeta_kpi(
            kpis.POR_ID[kpi_id],
            tabla.loc[indices_kpi[kpi_id], "valor"],
            tabla.loc[indices_kpi[kpi_id], "var_vs_mes_anterior_%"],
            previo,
        )
        for kpi_id in principales
    ]
    complementarios = []
    for bloque in kpis.BLOQUES:
        indicadores_bloque = [
            tarjeta_kpi(
                k,
                tabla.loc[i, "valor"],
                tabla.loc[i, "var_vs_mes_anterior_%"],
                previo,
            )
            for i, k in enumerate(kpis.CATALOGO)
            if k["bloque"] == bloque and k["id"] not in principales
        ]
        if indicadores_bloque:
            complementarios.append(html.Div([
                html.Div(bloque, className="bloque-titulo"),
                html.Div(indicadores_bloque, className="kpi-grid"),
            ], className="bloque"))

    # Hallazgos automáticos (último periodo)
    u = d[d["periodo_cod"] == actual]
    k = kpis.calcular_kpis(u)
    dist = indicadores(u, ["provincia", "distrito"])
    dist["pct_fact"] = dist["monto_soles"] / dist["monto_soles"].sum() * 100
    top = dist.nlargest(1, "monto_soles").iloc[0]
    items = [f"💰 {top['distrito']} ({top['provincia']}) genera el {top['pct_fact']:.1f}% de la facturación; "
             f"los 10 distritos mayores concentran el {k['pct_top10']:.1f}%.",
             f"🏭 El {k['pct_mt']:.1f}% del ingreso viene de media tensión (minería/industria): "
             "perder un gran cliente impacta directamente en la caja."]
    mt = dist.assign(mt=dist["monto_soles"] * dist["pct_mt_facturacion"].fillna(0) / 100).nlargest(3, "mt")
    if mt["mt"].sum() > 0:
        items.append("⛏ Mayor facturación en media tensión: " + ", ".join(mt["distrito"]) + ".")
    if pd.notna(k["monto_alerta"]) and k["n_alertas"] > 0:
        al = u[u["anomalia"]].groupby("distrito")["monto_soles"].sum().nlargest(3)
        items.append(f"🔎 S/ {k['monto_alerta']:,.0f} en {k['n_alertas']:,} registros con alerta. "
                     f"Priorizar inspección en: {', '.join(al.index)}.")
    vul = dist[dist["clientes"] >= 500].dropna(subset=["pct_bajo_consumo"]).nlargest(3, "pct_bajo_consumo")
    if len(vul):
        items.append(f"🏠 El {k['pct_bajo_consumo']:.1f}% de hogares consume ≤ 30 kWh. Mayor proporción en "
                     + ", ".join(f"{r.distrito} ({r.pct_bajo_consumo:.0f}%)" for r in vul.itertuples())
                     + " → focalizar programas sociales o revisar la medición.")
    if pd.notna(k["kwh_residencial"]):
        items.append(f"⚡ Un hogar típico consume {k['kwh_residencial']:.1f} kWh/mes y paga en promedio "
                     f"S/ {k['precio_bt']:.3f} por kWh en baja tensión.")

    # Concentración: top 10 distritos (% de la facturación y acumulado)
    t10 = dist.nlargest(10, "pct_fact").copy()
    t10["acum"] = t10["pct_fact"].cumsum()
    t10 = t10.iloc[::-1]
    fig = px.bar(t10, x="pct_fact", y="distrito", orientation="h",
                 hover_data={"provincia": True, "monto_soles": ":,.0f", "acum": ":.1f"},
                 labels={"pct_fact": "% de la facturación", "distrito": "", "acum": "% acumulado",
                         "monto_soles": "S/"},
                 title="¿Dónde se concentra el ingreso? Top 10 distritos (% de la facturación)")
    fig.update_traces(marker_color=AZUL, marker_cornerradius=4, cliponaxis=False, textposition="outside",
                      text=[f"{a:.1f}%  (acum. {b:.0f}%)" for a, b in zip(t10["pct_fact"], t10["acum"])])
    fig.update_xaxes(range=[0, t10["pct_fact"].max() * 1.45])
    return cab, tarjetas, complementarios, [html.Li(i) for i in items], fig


@app.callback(Output("g-mapa", "figure"), FILTROS + [Input("f-metrica", "value")])
def mapa(per, prov, grp, met):
    d = filtrar(per, prov, grp)
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
                                    "pct_fose": "% ≤ 100 kWh"},
                            )
    ajustar_mapa(fig)
    fig.update_layout(title=f"{etiqueta} por distrito", coloraxis_colorbar=dict(title="", thickness=12))
    return fig


@app.callback([Output("ficha", "children"), Output("g-ficha-tarifa", "figure")],
              FILTROS + [Input("g-mapa", "clickData")])
def ficha(per, prov, grp, click):
    gid = click["points"][0]["location"] if click else "211101"
    d = filtrar(per, prov, grp)
    d = d[d["geo_id"] == gid]
    if d.empty:
        return html.P("Selecciona un distrito con datos en el mapa."), vacio("")
    t = indicadores(d.assign(_t=1), ["_t"]).iloc[0]
    seg = ind_static.loc[ind_static["geo_id"] == gid, "cluster"]
    filas = [("Provincia", d["provincia"].iloc[0]), ("Clientes", fmt(t.clientes)),
             ("Energía", f"{fmt(t.consumo_kwh)} kWh/mes"), ("Facturación", f"S/ {fmt(t.monto_soles)}/mes"),
             ("kWh por cliente BT5B", fmt(t.kwh_cliente_bt5b, ",.1f")),
             ("% clientes ≤ 100 kWh", f"{fmt(t.pct_fose, ',.1f')} %"),
             ("% facturación MT", f"{fmt(t.pct_mt_facturacion, ',.1f')} %"),
             ("Segmento", seg.iloc[0] if len(seg) else "–"), ("Registros anómalos", fmt(t.anomalias))]
    cont = [html.H3(f"📍 {nombre_geo.get(gid, gid)}"),
            html.Table([html.Tr([html.Td(a, className="ficha-k"), html.Td(b)]) for a, b in filas], className="ficha")]
    tar = d.groupby("tarifa", as_index=False)["monto_soles"].sum().sort_values("monto_soles")
    fig = px.bar(tar, x="monto_soles", y="tarifa", orientation="h", text_auto=".3s",
                 labels={"monto_soles": "S/", "tarifa": ""}, title="Facturación por tarifa")
    fig.update_traces(marker_color=AZUL, marker_cornerradius=4, textposition="outside", cliponaxis=False)
    fig.update_xaxes(range=[0, tar["monto_soles"].max() * 1.2])
    return cont, fig


@app.callback(Output("g-top", "figure"), FILTROS + [Input("f-metrica", "value")])
def top(per, prov, grp, met):
    d = filtrar(per, prov, grp)
    if d.empty:
        return vacio()
    m = indicadores(d, ["provincia", "distrito"])
    if met not in ("clientes", "consumo_kwh", "monto_soles"):
        m = m[m["clientes"] >= 100]  # evita ratios inestables en distritos muy pequeños
    m = m.nlargest(15, met).sort_values(met)
    etiqueta, f = METRICAS[met]
    fig = px.bar(m, x=met, y="distrito", orientation="h", hover_data=["provincia"], text_auto=f,
                 labels={met: etiqueta, "distrito": ""}, title=f"Top 15 distritos · {etiqueta}")
    fig.update_traces(marker_color=AZUL, marker_cornerradius=4, textposition="outside", cliponaxis=False)
    return fig


@app.callback(Output("g-provincia", "figure"), FILTROS)
def por_provincia(per, prov, grp):
    d = filtrar(per, prov, grp)
    if d.empty:
        return vacio()
    m = indicadores(d, ["provincia"]).sort_values("monto_soles")
    fig = px.bar(m, x="monto_soles", y="provincia", orientation="h", text_auto=".3s",
                 hover_data={"clientes": ":,.0f", "kwh_cliente_bt5b": ":.1f"},
                 labels={"monto_soles": "Facturación (S//mes)", "provincia": ""}, title="Facturación por provincia")
    fig.update_traces(marker_color=AZUL, marker_cornerradius=4, textposition="outside", cliponaxis=False)
    return fig


@app.callback(Output("g-composicion", "figure"), FILTROS)
def composicion(per, prov, grp):
    d = filtrar(per, prov, grp)
    if d.empty:
        return vacio()
    g = d.groupby("grupo_tarifa")[["clientes", "consumo_kwh", "monto_soles"]].sum()
    g = (g / g.sum() * 100).rename(columns={"clientes": "Clientes", "consumo_kwh": "Energía",
                                            "monto_soles": "Facturación"})
    largo = g.reset_index().melt(id_vars="grupo_tarifa", var_name="medida", value_name="pct")
    fig = px.bar(largo, x="pct", y="medida", color="grupo_tarifa", orientation="h", text_auto=".1f",
                 category_orders={"grupo_tarifa": GRUPOS},
                 color_discrete_map=dict(zip(GRUPOS, CATEG)),
                 labels={"pct": "% del total", "medida": "", "grupo_tarifa": ""},
                 title="¿Quién pesa más? Participación por grupo tarifario (%)")
    fig.update_traces(marker_line_color="white", marker_line_width=2)
    fig.update_layout(legend=dict(y=1.02, yanchor="bottom", x=0), margin=dict(t=80))
    return fig


@app.callback(Output("g-precio", "figure"), FILTROS)
def precio(per, prov, grp):
    d = filtrar(per, prov, grp)
    d = d[d["precio_kwh"].between(0.01, 5) & (d["nivel_tension"] == "BT")]
    if d.empty:
        return vacio()
    orden = d.groupby("tarifa")["precio_kwh"].median().sort_values().index
    fig = px.box(d, x="precio_kwh", y="tarifa", points=False, category_orders={"tarifa": list(orden)},
                 labels={"precio_kwh": "S/ por kWh", "tarifa": ""},
                 title="Precio medio efectivo en baja tensión (S/ por kWh)")
    fig.update_traces(marker_color=AZUL, line_color=AZUL, fillcolor="rgba(42,120,214,0.25)")
    return fig


@app.callback(Output("g-treemap", "figure"), FILTROS)
def treemap(per, prov, grp):
    d = filtrar(per, prov, grp)
    if d.empty:
        return vacio()
    m = indicadores(d, ["provincia", "distrito"]).fillna({"kwh_cliente_bt5b": 0})
    fig = px.treemap(m, path=[px.Constant("Puno"), "provincia", "distrito"], values="clientes",
                     color="kwh_cliente_bt5b", color_continuous_scale=SEQ,
                     range_color=(0, m["kwh_cliente_bt5b"].quantile(0.95)),
                     hover_data={"monto_soles": ":,.0f"}, labels={"kwh_cliente_bt5b": "kWh/cliente BT5B"},
                     title="Clientes por provincia y distrito (tamaño = clientes · color = kWh por cliente BT5B)")
    fig.update_traces(root_color="#f3f3f1", marker_line_color="white", marker_line_width=2)
    return fig


@app.callback([Output("g-cluster-mapa", "figure"), Output("g-cluster-scatter", "figure"),
               Output("tabla-cluster", "children")], [Input("f-provincia", "value")])
def segmentos(prov):
    s = ind_static if not prov else ind_static[ind_static["provincia"].isin(prov)]
    orden = sorted(ind_static["cluster"].unique())
    cmap = dict(zip(orden, CATEG))
    f1 = px.choropleth(s, geojson=geo, locations="geo_id", featureidkey="properties.IDDIST", color="cluster",
                           color_discrete_map=cmap, category_orders={"cluster": orden}, hover_name="distrito",
                           title="Mapa de segmentos")
    ajustar_mapa(f1)
    f1.update_layout(legend=dict(title="", y=-0.02))
    f2 = px.scatter(s, x="clientes", y="kwh_cliente_bt5b", color="cluster", size="monto_soles", log_x=True,
                    size_max=40, hover_name="distrito", color_discrete_map=cmap, category_orders={"cluster": orden},
                    hover_data={"provincia": True, "pct_fose": ":.1f", "pct_mt_facturacion": ":.1f"},
                    labels={"clientes": "Clientes (escala log)", "kwh_cliente_bt5b": "kWh por cliente BT5B",
                            "cluster": ""}, title="Tamaño vs consumo por cliente (burbuja = facturación)")
    f2.update_traces(marker_line_color="white", marker_line_width=1.5)
    f2.update_layout(legend=dict(y=-0.25))
    perfil = (ind_static.groupby("cluster")
              .agg(distritos=("distrito", "size"), clientes=("clientes", "median"),
                   kwh_cliente_bt5b=("kwh_cliente_bt5b", "median"), pct_fose=("pct_fose", "median"),
                   pct_mt_facturacion=("pct_mt_facturacion", "median"),
                   densidad_clientes=("densidad_clientes", "median"),
                   ejemplos=("distrito", lambda v: ", ".join(ind_static.loc[v.index].nlargest(4, "clientes")["distrito"])))
              .round(1).reset_index())
    tabla = dash_table.DataTable(
        perfil.to_dict("records"),
        [{"name": n, "id": c} for c, n in [("cluster", "Segmento"), ("distritos", "N° distritos"),
                                           ("clientes", "Clientes (mediana)"), ("kwh_cliente_bt5b", "kWh/cliente BT5B"),
                                           ("pct_fose", "% ≤ 100 kWh"), ("pct_mt_facturacion", "% fact. MT"),
                                           ("densidad_clientes", "Clientes/km²"), ("ejemplos", "Distritos principales")]],
        style_cell={"fontFamily": "Inter, Segoe UI, sans-serif", "fontSize": 13, "padding": "6px", "textAlign": "left"},
        style_header={"fontWeight": 600, "backgroundColor": "#f3f3f1"})
    return f1, f2, tabla


@app.callback([Output("g-hist", "figure"), Output("txt-ajuste", "children")], FILTROS)
def histograma(per, prov, grp):
    d = filtrar(per, prov, grp)
    x = d.loc[(d["tarifa"] == "BT5B") & (d["consumo_kwh"] > 0), "kwh_cliente"].to_numpy()
    if len(x) < 20:
        return vacio("Se necesita la tarifa BT5B en el filtro"), ""
    s, _, esc = stats.lognorm.fit(x, floc=0)
    lx = np.log10(x)
    fig = go.Figure()
    fig.add_histogram(x=lx, histnorm="probability density", nbinsx=60, name="Registros BT5B",
                      marker=dict(color="rgba(42,120,214,0.45)", line=dict(color="white", width=1)))
    gx = np.linspace(lx.min(), lx.max(), 300)
    # Si X ~ LogNormal(μ, σ) entonces log10(X) ~ Normal(μ/ln10, σ/ln10)
    fig.add_scatter(x=gx, y=stats.norm.pdf(gx, np.log10(esc), s / np.log(10)), name="Log-normal ajustada",
                    line=dict(color="#0d366b", width=2))
    fig.add_vline(x=2, line_dash="dash", line_color=ROJO, annotation_text="100 kWh (FOSE)")
    fig.add_vline(x=np.log10(30), line_dash="dot", line_color="#52514e", annotation_text="30 kWh",
                  annotation_position="bottom right")
    ticks = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
    fig.update_xaxes(tickvals=np.log10(ticks), ticktext=[str(t) for t in ticks],
                     title="kWh por cliente al mes (escala log)")
    fig.update_layout(title="Distribución del consumo por cliente residencial (BT5B)", yaxis_title="densidad",
                      bargap=0.02, legend=dict(y=1.02, yanchor="bottom", x=0), margin=dict(t=80))
    txt = (f"Modelo log-normal: mediana = {esc:,.1f} kWh, σ(log) = {s:.2f} · "
           f"P(X ≤ 30) = {stats.lognorm.cdf(30, s, 0, esc):.1%} · P(X ≤ 100) = {stats.lognorm.cdf(100, s, 0, esc):.1%} · "
           f"P(X > 300) = {stats.lognorm.sf(300, s, 0, esc):.2%}. "
           f"(Mejor ajuste global en el Notebook 3: {resumen_dist['mejor_distribucion']}).")
    return fig, txt


@app.callback(Output("g-ic", "figure"), FILTROS)
def intervalos(per, prov, grp):
    d = filtrar(per, prov, grp)
    d = d[(d["tarifa"] == "BT5B") & (d["consumo_kwh"] > 0)]
    if d.empty:
        return vacio()
    rng = np.random.default_rng(0)
    filas = []
    for p, g in d.groupby("provincia"):
        v, w = g["kwh_cliente"].to_numpy(), g["clientes"].to_numpy()
        idx = rng.integers(0, len(v), size=(500, len(v)))
        medias = (v[idx] * w[idx]).sum(1) / w[idx].sum(1)
        filas.append({"provincia": p, "media": (v * w).sum() / w.sum(),
                      "lo": np.quantile(medias, .025), "hi": np.quantile(medias, .975)})
    m = pd.DataFrame(filas).sort_values("media")
    fig = go.Figure(go.Scatter(x=m["media"], y=m["provincia"], mode="markers",
                               marker=dict(size=10, color=AZUL, line=dict(color="white", width=2)),
                               error_x=dict(type="data", array=m["hi"] - m["media"], arrayminus=m["media"] - m["lo"],
                                            color=AZUL, thickness=2),
                               hovertemplate="%{y}<br>%{x:.1f} kWh/cliente<extra></extra>"))
    fig.update_layout(title="kWh por cliente BT5B por provincia (IC 95% bootstrap)", xaxis_title="kWh por cliente")
    return fig


@app.callback(Output("tabla-anom", "children"), FILTROS)
def tabla_anomalias(per, prov, grp):
    d = filtrar(per, prov, grp)
    a = d[d["anomalia"]].copy()
    cols = [("periodo_cod", "Periodo"), ("provincia", "Provincia"), ("distrito", "Distrito"), ("tarifa", "Tarifa"),
            ("clientes", "Clientes"), ("consumo_kwh", "kWh"), ("monto_soles", "S/"), ("kwh_cliente", "kWh/cliente"),
            ("precio_kwh", "S//kWh"), ("tipo_anomalia", "Tipo de alerta")]
    a = a[[c for c, _ in cols]].round(2).sort_values("monto_soles", ascending=False)
    return dash_table.DataTable(
        a.to_dict("records"), [{"name": n, "id": c} for c, n in cols], page_size=12,
        sort_action="native", filter_action="native",
        style_cell={"fontFamily": "Inter, Segoe UI, sans-serif", "fontSize": 13, "padding": "6px"},
        style_header={"fontWeight": 600, "backgroundColor": "#f3f3f1"},
        style_data_conditional=[{"if": {"filter_query": '{tipo_anomalia} contains "Consumo atípico"',
                                        "column_id": "tipo_anomalia"}, "color": "#b42318", "fontWeight": 600}])


@app.callback(Output("g-serie", "figure"), FILTROS + [Input("f-serie", "value")])
def serie(per, prov, grp, met):
    d = filtrar(per, prov, grp)
    if d.empty:
        return vacio()
    m = indicadores(d, ["periodo", "provincia"])
    etiqueta = METRICAS[met][0]
    if m["periodo"].nunique() == 1:
        m = m.sort_values(met)
        fig = px.bar(m, x=met, y="provincia", orientation="h", labels={met: etiqueta, "provincia": ""},
                     title=f"{etiqueta} · único periodo cargado ({m['periodo'].iloc[0]:%Y-%m}). "
                           "Agrega más CSV mensuales en data/raw/ para ver la evolución.")
        fig.update_traces(marker_color=AZUL, marker_cornerradius=4)
        return fig
    tot = indicadores(d, ["periodo"]).assign(provincia="TOTAL PUNO")
    fig = px.line(m, x="periodo", y=met, line_group="provincia", hover_name="provincia", markers=True,
                  labels={met: etiqueta, "periodo": ""}, title=f"Evolución mensual · {etiqueta} (gris = provincias)")
    fig.update_traces(line=dict(color="#c3c2b7", width=1.5), marker=dict(size=6, color="#c3c2b7"))
    if met in ("kwh_cliente_bt5b",):
        fig.add_scatter(x=tot["periodo"], y=tot[met], name="Total Puno", mode="lines+markers",
                        line=dict(color=AZUL, width=3), marker=dict(size=8))
    return fig


@app.callback(Output("descarga", "data"), Input("btn-descarga", "n_clicks"),
              State("f-periodo", "value"), State("f-provincia", "value"), State("f-grupo", "value"),
              prevent_initial_call=True)
def descargar(n, per, prov, grp):
    d = filtrar(per, prov, grp)
    buf = io.StringIO()
    d.to_csv(buf, index=False)
    return dict(content=buf.getvalue(), filename="electropuno_filtrado.csv")


if __name__ == "__main__":
    app.run(debug=True)
