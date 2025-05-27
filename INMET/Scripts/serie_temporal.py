import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

variaveis_automatico = [
    'temp_max','temp_min','temp_media','umid_max','umid_min','umid_mediana',
    'rad_min','rad_max','rad_mediana','vento_raj_max','vento_vel_media','precipitacao_total'
]

def grafico_interativo_com_dropdown(caminho_csv, janela_movel=7):
    df = pd.read_csv(caminho_csv, parse_dates=['data'])
    fig = go.Figure()
    
    visibilidade = []

    for idx, var in enumerate(variaveis_automatico):
        if var in df.columns:
            df[f'{var}_movel'] = df[var].rolling(window=janela_movel, min_periods=1).mean()

            fig.add_trace(go.Scattergl(
                x=df['data'],
                y=df[f'{var}_movel'],
                mode='lines',
                name=f'{var} - média móvel',
                line=dict(width=2),
                visible=(idx == 0)
            ))

            fig.add_trace(go.Scattergl(
                x=df['data'],
                y=df[var],
                mode='lines+markers',
                name=f'{var} - original',
                line=dict(width=1),
                marker=dict(size=4),
                visible=(idx == 0)
            ))

            visibilidade.append(2)

    botoes = []
    for i, var in enumerate(variaveis_automatico):
        vis = [False] * len(visibilidade) * 2
        vis[i*2] = True
        vis[i*2 + 1] = True

        botoes.append(dict(
            label=var,
            method='update',
            args=[{'visible': vis},
                  {'title': f'Série Temporal: {var} (com média móvel de {janela_movel} dias)'}]
        ))

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=botoes,
                direction="down",
                x=1.3,
                y=1.15
            )
        ],
        title=f'Série Temporal: {variaveis_automatico[0]} (com média móvel de {janela_movel} dias)',
        xaxis_title='Data',
        yaxis_title='Valor',
        hovermode='x unified',
        template='plotly_white'
    )

    # ✅ Exportar como HTML e abrir
    fig.write_html('grafico_interativo_dropdown.html', auto_open=True)

# Exemplo de uso:
grafico_interativo_com_dropdown('resumo_diario_climatico.csv', janela_movel=7)
