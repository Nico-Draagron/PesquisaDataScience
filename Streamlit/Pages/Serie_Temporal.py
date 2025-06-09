import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Série Temporal", layout="wide")
st.title("📈 Série Temporal Climática")

# Configurações
variaveis_automatico = [
    'temp_max','temp_min','temp_media','umid_max','umid_min','umid_mediana',
    'rad_min','rad_max','rad_mediana','vento_raj_max','vento_vel_media','precipitacao_total'
]

# Input do usuário
caminho_csv = "resumo_diario_climatico.csv"
janela_movel = st.slider("Selecione a janela móvel (dias):", 1, 30, 7)

# Carrega CSV
df = pd.read_csv(caminho_csv, parse_dates=['data'])

# Dropdown de variáveis
var_selecionada = st.selectbox("Escolha a variável:", variaveis_automatico)

# Verifica se a variável existe
if var_selecionada in df.columns:
    df[f'{var_selecionada}_movel'] = df[var_selecionada].rolling(window=janela_movel, min_periods=1).mean()

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=df['data'],
        y=df[f'{var_selecionada}_movel'],
        mode='lines',
        name=f'{var_selecionada} - média móvel',
        line=dict(width=2)
    ))

    fig.add_trace(go.Scattergl(
        x=df['data'],
        y=df[var_selecionada],
        mode='lines+markers',
        name=f'{var_selecionada} - original',
        line=dict(width=1),
        marker=dict(size=4)
    ))

    fig.update_layout(
        title=f"Série Temporal: {var_selecionada} (com média móvel de {janela_movel} dias)",
        xaxis_title="Data",
        yaxis_title="Valor",
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(f"A variável '{var_selecionada}' não está no CSV.")
