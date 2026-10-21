"""
KPIs para la Gerencia de Electro Puno S.A.A.

Un solo lugar donde se definen y calculan los indicadores clave, para que los notebooks
y el dashboard (app.py) muestren exactamente los mismos números.

Los datos abiertos solo traen clientes, energía (CEA) y monto facturado por zona y tarifa;
por eso no se incluyen KPIs que requieren otras fuentes (pérdidas técnicas, morosidad,
SAIDI/SAIFI, cobranza).
"""
import numpy as np
import pandas as pd

# sentido: +1 = más es mejor, -1 = menos es mejor, 0 = informativo (sin juicio)
CATALOGO = [
    # --- Bloque 1: Comercial ---------------------------------------------------------------
    dict(id="facturacion", bloque="Comercial", nombre="Facturación mensual", unidad="S/ por mes", fmt=",.0f", sentido=+1,
         formula="Σ M_Total / n° de meses",
         pregunta="¿Cuánto factura la empresa al mes?"),
    dict(id="energia_mwh", bloque="Comercial", nombre="Energía facturada", unidad="MWh/mes", fmt=",.1f", sentido=+1,
         formula="Σ CEA / 1000 / n° de meses",
         pregunta="¿Cuánta energía vendemos?"),
    dict(id="precio_bt", bloque="Comercial", nombre="Precio medio en BT", unidad="S/ por kWh", fmt=",.3f", sentido=0,
         formula="Σ M_Total BT / Σ CEA BT (solo registros con consumo)",
         pregunta="¿Cuánto se cobra en promedio por cada kWh en baja tensión?"),
    # --- Bloque 2: Clientes y consumo -----------------------------------------------------
    dict(id="clientes", bloque="Clientes y consumo", nombre="Clientes atendidos", unidad="suministros", fmt=",.0f",
         sentido=+1, formula="Σ Clientes / n° de meses",
         pregunta="¿A cuántos suministros atendemos?"),
    dict(id="ingreso_cliente", bloque="Clientes y consumo", nombre="Ingreso medio por cliente", unidad="S/ por mes",
         fmt=",.2f", sentido=+1, formula="Facturación / Clientes",
         pregunta="¿Cuánto aporta en promedio cada cliente?"),
    dict(id="kwh_residencial", bloque="Clientes y consumo", nombre="Consumo medio residencial", unidad="kWh/cliente BT5B",
         fmt=",.1f", sentido=+1, formula="Σ CEA BT5B / Σ Clientes BT5B",
         pregunta="¿Cuánto consume un hogar típico?"),
    # --- Bloque 3: Riesgo y concentración ---------------------------------------------------
    dict(id="pct_mt", bloque="Riesgo y concentración", nombre="Dependencia de grandes clientes",
         unidad="% de la facturación en MT", fmt=".1f", sentido=-1,
         formula="Σ M_Total media tensión / Σ M_Total × 100",
         pregunta="¿Qué parte del ingreso depende de minería/industria (media tensión)?"),
    dict(id="pct_top10", bloque="Riesgo y concentración", nombre="Concentración territorial",
         unidad="% de la facturación en 10 distritos", fmt=".1f", sentido=-1,
         formula="Facturación de los 10 distritos mayores / Facturación total × 100",
         pregunta="¿Qué tan concentrado está el negocio en pocos distritos?"),
    dict(id="pct_sin_consumo", bloque="Riesgo y concentración", nombre="Suministros sin consumo",
         unidad="% de clientes", fmt=".1f", sentido=-1,
         formula="Clientes en registros con CEA = 0 / Clientes × 100",
         pregunta="¿Cuántos suministros no registran consumo (inactivos, medidor, posible hurto)?"),
    # --- Bloque 4: Control y social ---------------------------------------------------------
    dict(id="monto_alerta", bloque="Control y social", nombre="Facturación bajo revisión", unidad="S/ en registros con alerta",
         fmt=",.0f", sentido=-1,
         formula="Σ M_Total de registros marcados como anómalos (Notebook 3)",
         pregunta="¿Cuánto dinero está en registros con consumo o precio atípico?"),
    dict(id="pct_fose", bloque="Control y social", nombre="Hogares en rango FOSE", unidad="% clientes BT5B ≤ 100 kWh",
         fmt=".1f", sentido=0, formula="Clientes BT5B con consumo ≤ 100 kWh / Clientes BT5B × 100",
         pregunta="¿Qué parte de los hogares recibe el subsidio FOSE (impacto regulatorio)?"),
    dict(id="pct_bajo_consumo", bloque="Control y social", nombre="Hogares de muy bajo consumo",
         unidad="% clientes BT5B ≤ 30 kWh", fmt=".1f", sentido=0,
         formula="Clientes BT5B con consumo ≤ 30 kWh / Clientes BT5B × 100",
         pregunta="¿Dónde hay posible pobreza energética o subregistro?"),
]
BLOQUES = ["Comercial", "Clientes y consumo", "Riesgo y concentración", "Control y social"]
POR_ID = {k["id"]: k for k in CATALOGO}


def _div(a, b):
    return np.nan if b == 0 or pd.isna(b) else a / b


def calcular_kpis(d: pd.DataFrame) -> dict:
    """Calcula los 12 KPIs sobre un DataFrame limpio (uno o varios periodos)."""
    if d.empty:
        return {k["id"]: np.nan for k in CATALOGO} | {"n_alertas": np.nan}
    n = max(d["periodo_cod"].nunique(), 1)
    bt = d[(d["nivel_tension"] == "BT") & (d["consumo_kwh"] > 0)]
    b5 = d[d["tarifa"] == "BT5B"]
    fact = d["monto_soles"].sum()
    top10 = d.groupby("ubigeo")["monto_soles"].sum().nlargest(10).sum()
    k = {
        "facturacion": fact / n,
        "energia_mwh": d["consumo_kwh"].sum() / 1000 / n,
        "precio_bt": _div(bt["monto_soles"].sum(), bt["consumo_kwh"].sum()),
        "clientes": d["clientes"].sum() / n,
        "ingreso_cliente": _div(fact, d["clientes"].sum()),
        "kwh_residencial": _div(b5["consumo_kwh"].sum(), b5["clientes"].sum()),
        "pct_mt": _div(d.loc[d["nivel_tension"] == "MT", "monto_soles"].sum(), fact) * 100,
        "pct_top10": _div(top10, fact) * 100,
        "pct_sin_consumo": _div(d.loc[d["consumo_kwh"] == 0, "clientes"].sum(), d["clientes"].sum()) * 100,
        "pct_fose": _div(b5.loc[b5["kwh_cliente"] <= 100, "clientes"].sum(), b5["clientes"].sum()) * 100,
        "pct_bajo_consumo": _div(b5.loc[b5["kwh_cliente"] <= 30, "clientes"].sum(), b5["clientes"].sum()) * 100,
    }
    if "anomalia" in d.columns:
        k["monto_alerta"] = d.loc[d["anomalia"], "monto_soles"].sum() / n
        k["n_alertas"] = int(d["anomalia"].sum())
    else:  # antes del Notebook 3 aún no existen las alertas
        k["monto_alerta"], k["n_alertas"] = np.nan, np.nan
    return k


def tabla_kpis(d: pd.DataFrame) -> pd.DataFrame:
    """KPIs del último periodo y variación % frente al periodo anterior (si existe)."""
    periodos = sorted(d["periodo_cod"].unique())
    actual = calcular_kpis(d[d["periodo_cod"] == periodos[-1]])
    previo = calcular_kpis(d[d["periodo_cod"] == periodos[-2]]) if len(periodos) > 1 else None
    filas = []
    for k in CATALOGO:
        v = actual[k["id"]]
        var = np.nan if previo is None else _div(v - previo[k["id"]], abs(previo[k["id"]])) * 100
        filas.append({"bloque": k["bloque"], "kpi": k["nombre"], "valor": v, "unidad": k["unidad"],
                      "var_vs_mes_anterior_%": var, "formula": k["formula"], "pregunta_gerencial": k["pregunta"]})
    return pd.DataFrame(filas)
