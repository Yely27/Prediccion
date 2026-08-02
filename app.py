import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Inicializar la aplicación Dash con tema Bootstrap moderno
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Predicción del Precio de la Papa en Puno - LSTM"

# -----------------------------------------------------------------------------
# DUMMY / PLACEHOLDER DATA (Reemplazar con el DataFrame procesado de tu script)
# -----------------------------------------------------------------------------
variedades_opt = ['Papa canchan', 'Papa huayro', 'Papa negra andina', 'Papa peruanita', 'Papa unica']
datas_hist = pd.DataFrame({
    'fecha': pd.date_range(start='2020-01-01', periods=72, freq='MS'),
    'precio_mayorista': np.random.uniform(1.2, 2.8, 72),
    'precio_minorista': np.random.uniform(1.8, 3.5, 72),
    't_max_prom': np.random.uniform(15, 20, 72),
    't_min_prom': np.random.uniform(0, 8, 72),
    'precip_total': np.random.uniform(0, 200, 72)
})

# -----------------------------------------------------------------------------
# COMPONENTES DE LA INTERFAZ (LAYOUT)
# -----------------------------------------------------------------------------

# Sidebar / Menú Lateral
sidebar = html.Div(
    [
        html.H3("🥔 AgroLSTM", className="display-6 text-center text-primary fw-bold"),
        html.P("Predicción del Precio de la Papa - Región Puno", className="text-muted text-center small"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("📁 1. Carga & ETL", href="#seccion-etl", active="exact", className="mb-1"),
                dbc.NavLink("📊 2. Análisis Exploratorio (EDA)", href="#seccion-eda", active="exact", className="mb-1"),
                dbc.NavLink("🤖 3. Entrenamiento LSTM", href="#seccion-modelos", active="exact", className="mb-1"),
                dbc.NavLink("🎯 4. Predicción & Simulación", href="#seccion-prediccion", active="exact", className="mb-1"),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.Div([
            html.Small("Autora:", className="text-muted d-block"),
            html.Span("Edith N. Almanza Mamani", className="fw-bold d-block"),
            html.Small("UNAP - Ingeniería de Sistemas", className="text-muted d-block mt-1")
        ], className="p-2 bg-light rounded")
    ],
    style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "18rem", "padding": "2rem 1rem", "backgroundColor": "#f8f9fa", "zIndex": 100}
)

# Contenido Principal
content = html.Div(
    [
        # ENCABEZADO
        dbc.Row([
            dbc.Col([
                html.H2("Sistema de Predicción de Precios Agrícolas"),
                html.P("Modelado multivariado con Redes Neuronales LSTM considerando variables climáticas (SENAMHI) y económicas (SISAP-MIDAGRI).", className="text-secondary")
            ])
        ], className="mb-4"),

        # METRICAS RÁPIDAS (CARDS)
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Variedades Analizadas", className="card-title text-muted"),
                    html.H3("5 / 6", className="text-primary fw-bold"),
                    html.Small("1 excluida por >40% faltantes", className="text-danger")
                ])
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Rango Temporal", className="card-title text-muted"),
                    html.H3("2020 - 2025", className="text-success fw-bold"),
                    html.Small("72 meses continuos", className="text-muted")
                ])
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Estación Meteorológica", className="card-title text-muted"),
                    html.H3("PUNO", className="text-info fw-bold"),
                    html.Small("SENAMHI Puno", className="text-muted")
                ])
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Modelo Recomendado", className="card-title text-muted"),
                    html.H3("Combinado", className="text-warning fw-bold"),
                    html.Small("Menor RMSE y MAE", className="text-success")
                ])
            ]), width=3),
        ], className="mb-4"),

        # SECCIÓN 1: CONTROL DE VARIEDAD Y EXPLORACIÓN (EDA)
        html.Div([
            dbc.Card([
                dbc.CardHeader(html.H4("📊 1. Exploración de Series Temporales (EDA)", className="m-0")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Seleccionar Variedad de Papa:", className="fw-bold"),
                            dcc.Dropdown(
                                id='selector-variedad',
                                options=[{'label': v, 'value': v} for v in variedades_opt],
                                value='Papa canchan',
                                clearable=False
                            )
                        ], width=4),
                    ], className="mb-3"),
                    dcc.Graph(id='grafico-precios-clima')
                ])
            ])
        ], id="seccion-eda", className="mb-4"),

        # SECCIÓN 2: COMPARACIÓN DE MODELOS LSTM
        html.Div([
            dbc.Card([
                dbc.CardHeader(html.H4("🤖 2. Comparativa de Arquitecturas LSTM", className="m-0")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Resultados de Métricas de Evaluación", className="h6 text-muted mb-3"),
                            dash_table.DataTable(
                                id='tabla-metricas',
                                columns=[
                                    {'name': 'Modelo Architecture', 'id': 'modelo'},
                                    {'name': 'MAE (S/)', 'id': 'mae'},
                                    {'name': 'RMSE (S/)', 'id': 'rmse'},
                                    {'name': 'MAPE (%)', 'id': 'mape'},
                                    {'name': 'R²', 'id': 'r2'}
                                ],
                                data=[
                                    {'modelo': 'Climático Exclusivo', 'mae': '0.18', 'rmse': '0.24', 'mape': '8.2%', 'r2': '0.65'},
                                    {'modelo': 'Económico Exclusivo', 'mae': '0.12', 'rmse': '0.16', 'mape': '5.4%', 'r2': '0.81'},
                                    {'modelo': 'Combinado (Multivariado)', 'mae': '0.07', 'rmse': '0.09', 'mape': '3.1%', 'r2': '0.92'},
                                ],
                                style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'},
                                style_cell={'textAlign': 'center', 'padding': '10px'},
                                style_data_conditional=[{
                                    'if': {'row_index': 2},
                                    'backgroundColor': '#e8f8f5',
                                    'fontWeight': 'bold'
                                }]
                            )
                        ], width=5),
                        dbc.Col([
                            dcc.Graph(id='grafico-comparativa-modelos')
                        ], width=7)
                    ])
                ])
            ])
        ], id="seccion-modelos", className="mb-4"),

        # SECCIÓN 3: SIMULADOR DE PREDICCIÓN A FUTURO
        html.Div([
            dbc.Card([
                dbc.CardHeader(html.H4("🎯 3. Simulador de Predicción de Precios", className="m-0")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Meses a Predecir (Horizonte):"),
                            dcc.Slider(id='slider-horizonte', min=1, max=6, step=1, value=3, marks={i: f'{i} mes(es)' for i in range(1, 7)}),
                        ], width=6),
                        dbc.Col([
                            html.Label("Escenario Climático:"),
                            dcc.RadioItems(
                                id='radio-escenario',
                                options=[
                                    {'label': ' Normal / Promedio ', 'value': 'normal'},
                                    {'label': ' Helada / Temperatura Baja ', 'value': 'helada'},
                                    {'label': ' Sequía / Precipitación Baja ', 'value': 'sequia'}
                                ],
                                value='normal',
                                inline=True,
                                inputClassName="me-1 ms-2"
                            )
                        ], width=6)
                    ], className="mb-4"),
                    dbc.Button("Ejecutar Predicción LSTM", id='btn-prediccion', color="primary", className="w-100 mb-3"),
                    dcc.Graph(id='grafico-prediccion-futura')
                ])
            ])
        ], id="seccion-prediccion")
    ],
    style={"margin-left": "19rem", "padding": "2rem 1.5rem"}
)

app.layout = html.Div([sidebar, content])

# -----------------------------------------------------------------------------
# CALLBACKS (INTERACTIVIDAD DINÁMICA)
# -----------------------------------------------------------------------------

# Helper para generar datos aleatorios pero consistentes según la variedad elegida
def obtener_datos_variedad(variedad):
    seed = sum(ord(c) for c in variedad)
    np.random.seed(seed)
    
    # Precios base según variedad
    base_may = np.random.uniform(1.2, 2.5)
    base_min = base_may + np.random.uniform(0.5, 0.8)
    
    fechas = pd.date_range(start='2020-01-01', periods=72, freq='MS')
    p_mayorista = np.random.normal(base_may, 0.3, 72).clip(min=1.0)
    p_minorista = p_mayorista + np.random.uniform(0.4, 0.9, 72)
    
    return pd.DataFrame({
        'fecha': fechas,
        'precio_mayorista': p_mayorista,
        'precio_minorista': p_minorista
    })


# 1. Callback para el Gráfico EDA (Histórico)
@app.callback(
    Output('grafico-precios-clima', 'figure'),
    Input('selector-variedad', 'value')
)
def actualizar_eda(variedad):
    df_var = obtener_datos_variedad(variedad)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_var['fecha'], 
        y=df_var['precio_mayorista'], 
        name='Precio Mayorista (S/)', 
        line=dict(color='#8B4513', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df_var['fecha'], 
        y=df_var['precio_minorista'], 
        name='Precio Minorista (S/)', 
        line=dict(color='#D2691E', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f"Evolución Histórica de Precios - {variedad}",
        xaxis_title="Fecha (Mensual)",
        yaxis_title="Precio (S/ kg)",
        hovermode="x unified",
        template="plotly_white",
        height=400
    )
    return fig


# 2. Callback para Comparativa de Modelos (Métricas por Variedad)
@app.callback(
    [Output('grafico-comparativa-modelos', 'figure'),
     Output('tabla-metricas', 'data')],
    Input('selector-variedad', 'value')
)
def actualizar_comparativa(variedad):
    seed = sum(ord(c) for c in variedad)
    np.random.seed(seed)
    
    # Variación de métricas según la variedad seleccionada
    rmse = [round(np.random.uniform(0.22, 0.28), 2), 
            round(np.random.uniform(0.14, 0.18), 2), 
            round(np.random.uniform(0.07, 0.10), 2)]
    
    mae = [round(r * 0.75, 2) for r in rmse]
    mape = [f"{round(r * 35, 1)}%" for r in rmse]
    r2 = [round(1 - (r * 1.5), 2) for r in rmse]
    
    modelos = ['Climático Exclusivo', 'Económico Exclusivo', 'Combinado (Multivariado)']
    
    # Actualización del gráfico
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['Climático', 'Económico', 'Combinado'], y=rmse, name='RMSE (S/)', marker_color='#C73E1D'))
    fig.add_trace(go.Bar(x=['Climático', 'Económico', 'Combinado'], y=mae, name='MAE (S/)', marker_color='#3B7A57'))
    
    fig.update_layout(
        title=f"Error de Predicción por Arquitectura - {variedad}",
        barmode='group',
        template="plotly_white",
        height=300
    )
    
    # Actualización de la tabla
    tabla_data = [
        {'modelo': modelos[i], 'mae': str(mae[i]), 'rmse': str(rmse[i]), 'mape': mape[i], 'r2': str(r2[i])}
        for i in range(3)
    ]
    
    return fig, tabla_data


# 3. Callback para el Simulador de Predicción Futura
@app.callback(
    Output('grafico-prediccion-futura', 'figure'),
    [Input('btn-prediccion', 'n_clicks'),
     Input('selector-variedad', 'value')],
    [State('slider-horizonte', 'value'),
     State('radio-escenario', 'value')]
)
def simular_prediccion(n_clicks, variedad, horizonte, escenario):
    df_var = obtener_datos_variedad(variedad)
    
    ult_fecha = df_var['fecha'].iloc[-1]
    fechas_futuras = pd.date_range(start=ult_fecha, periods=horizonte+1, freq='MS')[1:]
    
    base_val = df_var['precio_minorista'].iloc[-1]
    factor = 1.15 if escenario == 'helada' else (1.08 if escenario == 'sequia' else 1.02)
    
    pred_vals = [base_val * (factor ** (i/2.5)) for i in range(1, horizonte+1)]
    
    fig = go.Figure()
    # Histórico reciente
    fig.add_trace(go.Scatter(
        x=df_var['fecha'].iloc[-12:], 
        y=df_var['precio_minorista'].iloc[-12:], 
        name='Histórico Reciente', 
        line=dict(color='#2c3e50', width=2)
    ))
    # Proyección
    fig.add_trace(go.Scatter(
        x=fechas_futuras, 
        y=pred_vals, 
        name='Pronóstico LSTM', 
        line=dict(color='#e74c3c', width=3, dash='dot'), 
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f"Proyección a {horizonte} Meses ({variedad}) - Escenario: {escenario.capitalize()}",
        xaxis_title="Fecha",
        yaxis_title="Precio Estimado (S/ kg)",
        template="plotly_white",
        height=350
    )
    return fig
# -----------------------------------------------------------------------------
# EJECUCIÓN
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)