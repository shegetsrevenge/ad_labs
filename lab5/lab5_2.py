import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go

t = np.linspace(0, 2 * np.pi, 1000)

def harmonic_with_noise(amplitude, frequency, phase, noise_mean, noise_std, show_noise, keep_noise=None):
    clean = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    if keep_noise is None:
        noise = np.random.normal(noise_mean, noise_std, size=t.shape)
    else:
        noise = keep_noise
    result = clean + noise if show_noise else clean
    return clean, result, noise

def moving_average_filter(signal, window_size=10):
    if window_size < 1:
        return signal
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Гармоніка з шумом та фільтрацією (Plotly Dash)"),

    html.Div([
        html.Label("Drop-down: тип фільтру (поки тільки Moving Average)"),
        dcc.Dropdown(
            id='filter-type',
            options=[{'label': 'Moving Average', 'value': 'ma'}],
            value='ma'
        ),
        html.Label("Розмір вікна фільтру:"),
        dcc.Slider(id='filter-window', min=1, max=100, step=1, value=20),
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        html.Label("Амплітуда:"),
        dcc.Slider(id='amplitude', min=0.1, max=5, step=0.1, value=1.0),
        html.Label("Частота:"),
        dcc.Slider(id='frequency', min=0.1, max=5, step=0.1, value=1.0),
        html.Label("Фаза:"),
        dcc.Slider(id='phase', min=0, max=2*np.pi, step=0.1, value=0.0),
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        html.Label("Середнє шуму:"),
        dcc.Slider(id='noise-mean', min=-1.0, max=1.0, step=0.05, value=0.0),
        html.Label("Дисперсія шуму:"),
        dcc.Slider(id='noise-std', min=0.0, max=1.0, step=0.05, value=0.2),
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        dcc.Checklist(
            id='options',
            options=[
                {'label': 'Показати шум', 'value': 'show_noise'},
                {'label': 'Показати фільтровану', 'value': 'show_filtered'}
            ],
            value=['show_noise', 'show_filtered'],
            labelStyle={'display': 'inline-block'}
        ),
    ]),

    html.Button("Оновити шум", id='generate-noise', n_clicks=0),

    dcc.Graph(id='graph1', style={'height': '400px'}),
    dcc.Graph(id='graph2', style={'height': '400px'}),
    dcc.Store(id='stored-noise')
])

@app.callback(
    Output('graph1', 'figure'),
    Output('graph2', 'figure'),
    Output('stored-noise', 'data'),
    Input('amplitude', 'value'),
    Input('frequency', 'value'),
    Input('phase', 'value'),
    Input('noise-mean', 'value'),
    Input('noise-std', 'value'),
    Input('options', 'value'),
    Input('filter-window', 'value'),
    Input('generate-noise', 'n_clicks'),
    State('stored-noise', 'data'),
)
def update_graph(amplitude, frequency, phase, noise_mean, noise_std,
                 options, window_size, n_clicks, stored_noise):

    show_noise = 'show_noise' in options
    show_filtered = 'show_filtered' in options

    keep_noise = np.array(stored_noise) if stored_noise and n_clicks == 0 else None

    clean, noisy, new_noise = harmonic_with_noise(
        amplitude, frequency, phase, noise_mean, noise_std, show_noise, keep_noise
    )

    filtered = moving_average_filter(noisy, window_size)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=t, y=clean, mode='lines', name='Чиста гармоніка'))
    if show_noise:
        fig1.add_trace(go.Scatter(x=t, y=noisy, mode='lines', name='З шумом'))

    fig1.update_layout(title='Гармоніка з шумом')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=clean, mode='lines', name='Чиста гармоніка'))
    if show_filtered:
        fig2.add_trace(go.Scatter(x=t, y=filtered, mode='lines', name='Фільтрована'))

    fig2.update_layout(title='Фільтрована гармоніка')

    return fig1, fig2, new_noise.tolist()

if __name__ == '__main__':
    app.run(debug=True)

