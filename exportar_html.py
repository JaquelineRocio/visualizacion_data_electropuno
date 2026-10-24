"""
Exporta el tablero a un solo archivo HTML que se abre sin Python (para presentar o enviar).

Reutiliza el layout y los callbacks de app.py: cada callback se ejecuta para "Toda la región" y para
cada provincia, y el HTML resultante cambia de ámbito con un menú. Los controles locales de cada
pestaña (indicador, escala, umbral, etc.) quedan en su valor predeterminado.

    python exportar_html.py      →  tablero_electropuno.html
"""
import hashlib
import json
from html import escape
from pathlib import Path

import plotly.io as pio
from plotly.offline import get_plotlyjs

import app as tablero

RAIZ = Path(__file__).parent
SALIDA = RAIZ / "tablero_electropuno.html"
FILTROS = {"f-anio": "anio", "f-provincia": "prov", "f-distrito": "dist", "f-grupo": "grp"}
OMITIR = ("Dropdown", "RadioItems", "Checklist", "Slider", "Download", "Button")
CLASES_OMITIDAS = {"control", "deslizador", "seccion-control", "opciones"}


# ---------------------------- Callbacks → datos por ámbito ------------------------------------
def componentes_por_id(c, acc=None):
    acc = {} if acc is None else acc
    if isinstance(c, (list, tuple)):
        for h in c:
            componentes_por_id(h, acc)
    elif hasattr(c, "to_plotly_json"):
        if getattr(c, "id", None):
            acc[c.id] = c
        componentes_por_id(getattr(c, "children", None), acc)
        for tab in getattr(c, "tabs", None) or []:
            componentes_por_id(tab, acc)
    return acc


def salidas(clave):
    """'..a.figure...b.children..' → [('a', 'figure'), ('b', 'children')] (sin el sufijo @hash)."""
    partes = clave.strip(".").split("...") if clave.startswith("..") else [clave]
    return [tuple(p.split("@")[0].rsplit(".", 1)) for p in partes]


def mayor_geo(prov):
    d = tablero.filtrar(None, prov, None, None)
    return d.groupby("geo_id")["clientes"].sum().idxmax()


def ejecutar(ambito, prov, por_id):
    valores = {"anio": [], "prov": prov, "dist": [], "grp": []}
    figs, bloques = {}, {}
    for clave, cb in tablero.app.callback_map.items():
        outs = salidas(clave)
        if any(i.startswith("f-") or i == "descarga" for i, _ in outs):
            continue   # filtros, limpieza y descarga no aplican al HTML estático
        args = []
        for dep in cb["inputs"] + cb.get("state", []):
            if dep["id"] in FILTROS:
                args.append(valores[FILTROS[dep["id"]]])
            elif dep["property"] == "clickData":   # ficha: distrito con más clientes del ámbito
                args.append({"points": [{"location": mayor_geo(prov)}]} if dep["id"] == "g-mapa" else None)
            else:
                args.append(getattr(por_id[dep["id"]], dep["property"], None))
        res = cb["callback"].__wrapped__(*args)
        res = res if len(outs) > 1 else [res]
        for (id_, prop), r in zip(outs, res):
            if prop == "figure":
                figs[id_] = json.loads(pio.to_json(r, validate=False))
            elif prop == "children":
                bloques[id_] = a_html(r)
    print(f"  · {ambito}: {len(figs)} gráficos")
    return {"figs": figs, "html": bloques}


# ---------------------------- Deduplicación (geojson, plantilla y trazas repetidas) -----------
def compactar(ambitos):
    geo, plantilla, pool, indice = None, None, [], {}
    for amb in ambitos.values():
        for fig in amb["figs"].values():
            plantilla = fig["layout"].pop("template", plantilla)
            refs = []
            for traza in fig["data"]:
                if "geojson" in traza:
                    geo, traza["geojson"] = traza["geojson"], "__GEO__"
                txt = json.dumps(traza, sort_keys=True, separators=(",", ":"))
                h = hashlib.md5(txt.encode()).hexdigest()
                if h not in indice:
                    indice[h] = len(pool)
                    pool.append(traza)
                refs.append(indice[h])
            fig["data"] = refs
    return {"geo": geo, "plantilla": plantilla, "trazas": pool, "ambitos": ambitos}


# ---------------------------- Layout de Dash → HTML -------------------------------------------
def estilo(d):
    return ";".join(f"{''.join('-' + c.lower() if c.isupper() else c for c in k)}:{v}" for k, v in (d or {}).items())


def atributos(c, extra_clase=""):
    clase = " ".join(x for x in [getattr(c, "className", None), extra_clase] if x)
    pares = [("id", getattr(c, "id", None)), ("class", clase or None), ("style", estilo(getattr(c, "style", None))),
             ("title", getattr(c, "title", None))]
    return "".join(f' {k}="{escape(str(v))}"' for k, v in pares if v)


def tabla(c):
    cols = c.columns or []
    cab = "".join(f"<th>{escape(col['name'])}</th>" for col in cols)
    filas = "".join("<tr>" + "".join(f"<td>{escape(str(f.get(col['id'], '')))}</td>" for col in cols) + "</tr>"
                    for f in c.data or [])
    return f'<table class="tabla-ficha"><thead><tr>{cab}</tr></thead><tbody>{filas}</tbody></table>'


def a_html(c, dinamicos=frozenset()):
    if c is None:
        return ""
    if isinstance(c, (list, tuple)):
        return "".join(a_html(h, dinamicos) for h in c)
    if not hasattr(c, "to_plotly_json"):
        return escape(str(c))
    tipo = c._type
    if tipo in OMITIR:
        return ""
    if set((getattr(c, "className", "") or "").split()) & CLASES_OMITIDAS:
        return ""
    if tipo == "Graph":
        return f'<div class="grafico" data-id="{c.id}" style="{estilo(c.style)}"></div>'
    if tipo == "DataTable":
        return tabla(c)
    if tipo == "Tabs":
        nav = "".join(f'<button class="pestana{" activa" if i == 0 else ""}" data-tab="{i}">{escape(t.label)}</button>'
                      for i, t in enumerate(c.children))
        paneles = "".join(f'<section class="panel{" activo" if i == 0 else ""}" data-tab="{i}">'
                          f'{a_html(t.children, dinamicos)}</section>' for i, t in enumerate(c.children))
        return f'<nav class="pestanas">{nav}</nav>{paneles}'
    if "filtros" in (getattr(c, "className", "") or "").split():
        return SELECTOR
    hijos = "" if getattr(c, "id", None) in dinamicos else a_html(getattr(c, "children", None), dinamicos)
    tag = tipo.lower()
    return f"<{tag}{atributos(c)}>{hijos}</{tag}>"


SELECTOR = """
<div class="filtros">
  <div class="filtro ancho"><label for="sel-ambito">Provincia</label>
    <select id="sel-ambito" class="selector-html"></select></div>
  <p class="nota nota-estatica">Versión HTML del tablero: el menú cambia todo el tablero de provincia.
    Los selectores internos de cada pestaña usan su vista predeterminada; para filtrar por distrito o año
    y usar los clics entre gráficos, abre el tablero en Dash (<code>python app.py</code>).</p>
</div>"""

CSS_EXTRA = """
.pestanas { display: flex; flex-wrap: wrap; border-bottom: 1px solid var(--linea); }
.pestana { border: 0; background: transparent; color: var(--texto-2); padding: 12px 18px; font: inherit;
           font-weight: 600; font-size: 14px; cursor: pointer; border-bottom: 3px solid transparent; }
.pestana.activa { color: var(--marca); border-bottom-color: var(--acento); }
.pestana:focus-visible { outline: 2px solid var(--foco); }
.panel { display: none; } .panel.activo { display: block; }
.selector-html { width: 100%; height: 38px; border: 1px solid var(--linea); border-radius: 8px; padding: 0 10px;
                 font: inherit; font-size: 15px; background: #fff; color: var(--texto); }
.nota-estatica { flex: 3 1 420px; margin: 0; align-self: center; }
.tabla-ficha { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
.tabla-ficha th { text-align: left; background: var(--fondo); padding: 6px; }
.tabla-ficha td { padding: 6px; border-top: 1px solid var(--linea); vertical-align: top; }
"""

JS = """
const D = JSON.parse(document.getElementById('datos').textContent);
const CONFIG = {displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d']};
let ambito = '';
function figura(id) {
  const f = D.ambitos[ambito].figs[id];
  if (!f) return null;
  const data = f.data.map(i => {
    const t = JSON.parse(JSON.stringify(D.trazas[i]));
    if (t.geojson === '__GEO__') t.geojson = D.geo;
    return t;
  });
  const layout = JSON.parse(JSON.stringify(f.layout));
  layout.template = D.plantilla;
  return {data, layout};
}
function pintar(panel) {
  panel.querySelectorAll('.grafico').forEach(div => {
    if (div.dataset.ambito === ambito) return;
    const f = figura(div.dataset.id);
    if (f) Plotly.react(div, f.data, f.layout, CONFIG);
    div.dataset.ambito = ambito;
  });
}
function aplicar() {
  for (const [id, h] of Object.entries(D.ambitos[ambito].html)) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = h;
  }
  pintar(document.querySelector('.panel.activo'));
}
const sel = document.getElementById('sel-ambito');
Object.keys(D.ambitos).forEach(k => sel.add(new Option(k ? k.charAt(0) + k.slice(1).toLowerCase() : 'Toda la región', k)));
sel.addEventListener('change', () => { ambito = sel.value; aplicar(); });
document.querySelectorAll('.pestana').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('.pestana, .panel').forEach(e => e.classList.toggle('activo', e.dataset.tab === b.dataset.tab));
  const panel = document.querySelector('.panel.activo');
  pintar(panel);
  panel.querySelectorAll('.grafico').forEach(div => Plotly.Plots.resize(div));
}));
aplicar();
"""


def main():
    layout = tablero.app.layout() if callable(tablero.app.layout) else tablero.app.layout
    por_id = componentes_por_id(layout)
    print("Calculando ámbitos…")
    ambitos = {"": ejecutar("Toda la región", [], por_id)}
    for p in tablero.PROVINCIAS:
        ambitos[p] = ejecutar(p.title(), [p], por_id)
    datos = compactar(ambitos)
    dinamicos = frozenset(datos["ambitos"][""]["html"])
    cuerpo = a_html(layout, dinamicos)
    css = (RAIZ / "assets" / "style.css").read_text(encoding="utf-8")
    datos_json = json.dumps(datos, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    pagina = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Electro Puno · Tablero gerencial</title>
<style>{css}{CSS_EXTRA}</style>
<script>{get_plotlyjs()}</script></head>
<body>{cuerpo}
<script id="datos" type="application/json">{datos_json}</script>
<script>{JS}</script></body></html>"""
    SALIDA.write_text(pagina, encoding="utf-8")
    print(f"Listo: {SALIDA.name} ({SALIDA.stat().st_size / 1e6:.1f} MB, {len(datos['trazas'])} trazas únicas)")


if __name__ == "__main__":
    main()
