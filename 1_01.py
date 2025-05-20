from dash import Dash, dcc, html
import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sheet_names = ['202301', '202302', '202303']

xls = 'clients.xlsx'

total_clients = []
lost_clients = []
new_clients = []

all_seen_clients = set()

for i, sheet in enumerate(tqdm(sheet_names, desc="Обработка месяцев")):
    df = pd.read_excel(xls, sheet_name=sheet)
    clients = set(df['CLIENTBASENUMBER'].unique())
    total_clients.append(len(clients))
    lost_clients.append(df.loc[df['pr'] == True, 'CLIENTBASENUMBER'].nunique())
    if i == 0:
        new_clients.append(np.nan)
    else:
        new = clients - all_seen_clients
        new_clients.append(len(new))
    all_seen_clients.update(clients)

fig = make_subplots(rows=1, cols=3,
                    subplot_titles=["Общее число клиентов", "Ушедшие клиенты", "Пришедшие клиенты"],
                    horizontal_spacing=0.08)

fig.add_trace(go.Bar(
    x=sheet_names, y=total_clients,
    marker_color='rgba(70,130,180,0.8)',
    text=total_clients, textposition='outside',
    hovertemplate='Месяц: %{x}<br>Клиенты: %{y}<extra></extra>'),
    row=1, col=1)

fig.add_trace(go.Bar(
    x=sheet_names, y=lost_clients,
    marker_color='rgba(205,92,92,0.8)',
    text=lost_clients, textposition='outside',
    hovertemplate='Месяц: %{x}<br>Ушли: %{y}<extra></extra>'),
    row=1, col=2)

fig.add_trace(go.Bar(
    x=sheet_names, y=new_clients,
    marker_color='rgba(46,139,87,0.8)',
    text=[None if np.isnan(val) else val for val in new_clients],
    textposition='outside',
    hovertemplate='Месяц: %{x}<br>Новые: %{y}<extra></extra>'),
    row=1, col=3)

fig.update_layout(
    title={'text': "Анализ клиентской базы по месяцам – 2023", 'y': 0.98, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
           'font': dict(size=18, color='black', family='Helvetica Neue')},
    plot_bgcolor='white', paper_bgcolor='white', height=400, width=1200,
    margin=dict(t=90, l=40, r=40, b=40),
    font=dict(family='Arial', size=10, color='black'),
    showlegend=False)

fig.update_xaxes(tickangle=45, showgrid=True, gridcolor='lightgrey', zeroline=False, tickfont=dict(size=9))
fig.update_yaxes(showgrid=True, gridcolor='lightgrey', zeroline=False, tickfont=dict(size=9))

#fig.update_yaxes(range=[30000, 35000], row=1, col=1)
#fig.update_yaxes(range=[30000, 35000], row=1, col=2)
#fig.update_yaxes(range=[30000, 35000], row=1, col=3)

app = Dash(__name__)

app.layout = html.Div(style={'maxWidth': '1200px', 'margin': '30px auto', 'padding': '25px', 'backgroundColor': '#fafafa',
                             'boxShadow': '0 6px 25px rgba(0,0,0,0.12)', 'borderRadius': '14px', 'textAlign': 'center'}, children=[
    html.H1("Дэшборд клиентской базы 2023", style={'fontFamily': 'Helvetica Neue', 'marginBottom': '20px'}),
    dcc.Graph(figure=fig, config={'displayModeBar': False}),
])

app.run_server(mode='inline')
