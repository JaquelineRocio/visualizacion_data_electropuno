# Electro Puno · Proyecto de fin de curso (Visualización de datos – UNAP)

Prototipo de visualización del **consumo de energía eléctrica de los clientes de Electro Puno S.A.A.**
([Datos Abiertos Perú](https://www.datosabiertos.gob.pe/dataset/consumo-de-energ%C3%ADa-el%C3%A9ctrica-de-los-clientes-de-electro-puno-saa)).

## Estructura

```
TRABAJO FINAL/
├── data/
│   ├── raw/                  ← aquí van los CSV del portal (uno por mes)
│   ├── geo/                  ← peru_distrital_simple.geojson (límites INEI, sesión 11)
│   └── processed/            ← lo generan los notebooks
├── 01_pretratamiento.ipynb   ← limpieza, formato, ubigeos, geolocalización, variables derivadas
├── 02_eda.ipynb              ← descriptiva, tarifas, territorio, Lorenz/Gini, indicadores, K-Means
├── 03_distribuciones.ipynb   ← ajuste de distribuciones, indicadores probabilísticos, anomalías
├── app.py                    ← dashboard en Dash
├── assets/style.css          ← estilos del dashboard
├── exportar_html.py          ← genera tablero_electropuno.html (versión que se abre sin Python)
└── tablero_electropuno.html  ← tablero exportado, listo para presentar o enviar
```

## Cómo cargar más datos
1. Descarga cada mes del portal y guarda el CSV en `data/raw/` con un nombre distinto,
   p. ej. `ReporteDatosAbiertosLibro_202607.csv`, `ReporteDatosAbiertosLibro_202608.csv`.
2. Ejecuta los notebooks **en orden** (01 → 02 → 03). Cada uno lee lo que produjo el anterior.
3. Con varios meses se activan las series de tiempo (pestaña *Evolución temporal* y sección 9 del EDA).

## Ejecutar
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace 01_pretratamiento.ipynb 02_eda.ipynb 03_distribuciones.ipynb
python app.py        # abrir http://127.0.0.1:8050
```

## KPIs para la Gerencia (`kpis.py`)
12 indicadores en 4 bloques, definidos una sola vez y usados por los notebooks y el dashboard:

| Bloque | KPIs | Pregunta |
|---|---|---|
| Comercial | Facturación mensual · Energía facturada · Precio medio en BT | ¿Cuánto vendemos y a qué precio? |
| Clientes y consumo | Clientes atendidos · Ingreso medio por cliente · Consumo medio residencial | ¿A quién atendemos y cuánto consume? |
| Riesgo y concentración | % facturación en media tensión · % facturación en 10 distritos · % suministros sin consumo | ¿Qué tan vulnerable es el ingreso? |
| Control y social | Facturación bajo revisión (alertas) · % hogares ≤ 100 kWh (FOSE) · % hogares ≤ 30 kWh | ¿Dónde revisar y a quién proteger? |

- Notebook 02, sección 11: definición, fórmulas y KPIs por provincia.
- Notebook 03, sección 8: tablero final de KPIs y lectura para la Gerencia.
- Dashboard, pestaña **Resumen gerencial**: los 12 KPIs filtrables, variación vs mes anterior, hallazgos automáticos y ficha técnica.

## Dashboard
- **Filtros:** año, provincia, distrito y grupo tarifario. Un clic en los gráficos comparativos agrega o quita
  esa provincia, distrito o año del filtro.
- **Paleta corporativa "azul petróleo"** validada para daltonismo y contraste; contexto en gris y selección en color (Gestalt).
- **Resumen gerencial:** gráfico principal de energía mensual; estacionalidad comparable entre años; consumo por provincia y grupo tarifario; facturación y suministros por mes; cuatro KPIs principales y ocho indicadores complementarios desplegables, más hallazgos y concentración del ingreso.
- **Territorio:** mapa coroplético por distrito (clic → ficha del distrito), top 15 y facturación por provincia.
- **Tarifas:** participación por grupo tarifario, precio efectivo por tarifa, treemap provincia → distrito.
- **Segmentos:** clusters K-Means de distritos (mapa, burbujas y perfil).
- **Distribución y anomalías:** histograma con log-normal ajustada, IC 95 % bootstrap, dispersión consumo × precio
  de las alertas, prioridad por provincia o distrito y cronología mensual. Controles en *Opciones de vista*.
- **Evolución temporal:** tendencia frente al año anterior, calendario mes × año, trayectorias por provincia o
  distrito (índice o valor real) y crecimiento entre años. Controles en *Opciones de vista*.
- **Conclusiones:** conclusión general y tabla de hallazgos → decisión → plazo, ordenada por prioridad y
  recalculada con los filtros.
- Botón para **descargar** los datos filtrados en CSV.

### Versión HTML (sin Python)
```bash
python exportar_html.py      # genera tablero_electropuno.html (~9 MB)
```
Incluye las 7 pestañas con un menú para cambiar de provincia; los controles internos usan su vista predeterminada.

La comparación entre instalaciones nuevas y antiguas requiere un archivo de altas con identificador de suministro y fecha de instalación. Los datos abiertos de consumo actuales están agregados y no permiten separar instalaciones por antigüedad; el dashboard no los presenta como si fueran cohortes de instalaciones.

Los mapas del dashboard son SVG (`px.choropleth`): no necesitan internet ni WebGL.
