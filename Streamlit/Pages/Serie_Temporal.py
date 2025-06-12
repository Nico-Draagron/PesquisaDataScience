import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Configuração da página
st.set_page_config(
    page_title="Série Temporal Climática", 
    layout="wide",
    page_icon="📈"
)

# Título principal
st.title("📈 Análise de Série Temporal Climática")
st.markdown("---")

# Configurações
VARIAVEIS_AUTOMATICO = [
    'temp_max', 'temp_min', 'temp_media', 'umid_max', 'umid_min', 'umid_mediana',
    'rad_min', 'rad_max', 'rad_mediana', 'vento_raj_max', 'vento_vel_media', 'precipitacao_total'
]

LABELS_VARIAVEIS = {
    'temp_max': 'Temperatura Máxima (°C)',
    'temp_min': 'Temperatura Mínima (°C)',
    'temp_media': 'Temperatura Média (°C)',
    'umid_max': 'Umidade Máxima (%)',
    'umid_min': 'Umidade Mínima (%)',
    'umid_mediana': 'Umidade Mediana (%)',
    'rad_min': 'Radiação Mínima',
    'rad_max': 'Radiação Máxima',
    'rad_mediana': 'Radiação Mediana',
    'vento_raj_max': 'Rajada Máxima (m/s)',
    'vento_vel_media': 'Velocidade Média do Vento (m/s)',
    'precipitacao_total': 'Precipitação Total (mm)'
}

@st.cache_data
def carregar_dados(caminho):
    """Carrega e processa os dados do CSV"""
    try:
        df = pd.read_csv(caminho, parse_dates=['data'])
        return df, None
    except FileNotFoundError:
        return None, "Arquivo CSV não encontrado!"
    except Exception as e:
        return None, f"Erro ao carregar o arquivo: {str(e)}"

def validar_dados(df):
    """Valida se os dados estão corretos"""
    if df is None:
        return False, "DataFrame não carregado"
    
    if df.empty:
        return False, "O arquivo CSV está vazio!"
    
    if 'data' not in df.columns:
        return False, "Coluna 'data' não encontrada no CSV!"
    
    return True, "Dados válidos"

def calcular_media_movel(df, variavel, janela):
    """Calcula a média móvel para uma variável"""
    return df[variavel].rolling(window=janela, min_periods=1).mean()

def calcular_estatisticas(df, variavel):
    """Calcula estatísticas básicas da variável"""
    serie = df[variavel].dropna()
    return {
        'média': serie.mean(),
        'mediana': serie.median(),
        'desvio_padrão': serie.std(),
        'mínimo': serie.min(),
        'máximo': serie.max(),
        'valores_nulos': df[variavel].isnull().sum()
    }

def criar_grafico(df, variavel, janela_movel, mostrar_original=True):
    """Cria o gráfico da série temporal"""
    # Calcula média móvel
    df_plot = df.copy()
    df_plot[f'{variavel}_movel'] = calcular_media_movel(df_plot, variavel, janela_movel)
    
    fig = go.Figure()
    
    # Adiciona a série original se solicitado
    if mostrar_original:
        fig.add_trace(go.Scattergl(
            x=df_plot['data'],
            y=df_plot[variavel],
            mode='lines',
            name=f'{LABELS_VARIAVEIS.get(variavel, variavel)} - Original',
            line=dict(width=1, color='lightblue'),
            opacity=0.6
        ))
    
    # Adiciona a média móvel
    fig.add_trace(go.Scattergl(
        x=df_plot['data'],
        y=df_plot[f'{variavel}_movel'],
        mode='lines',
        name=f'Média Móvel ({janela_movel} dias)',
        line=dict(width=3, color='darkblue')
    ))
    
    # Configurações do layout
    fig.update_layout(
        title=f"Série Temporal: {LABELS_VARIAVEIS.get(variavel, variavel)}",
        xaxis_title="Data",
        yaxis_title=LABELS_VARIAVEIS.get(variavel, variavel),
        hovermode="x unified",
        template="plotly_white",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Sidebar para controles
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Input do arquivo
    caminho_csv = st.text_input(
        "Caminho do arquivo CSV:", 
        value="resumo_diario_climatico.csv",
        help="Digite o caminho completo para o arquivo CSV"
    )
    
    st.markdown("---")
    
    # Controles de análise
    janela_movel = st.slider(
        "Janela da média móvel (dias):", 
        min_value=1, 
        max_value=30, 
        value=7,
        help="Número de dias para calcular a média móvel"
    )
    
    var_selecionada = st.selectbox(
        "Variável para análise:", 
        VARIAVEIS_AUTOMATICO,
        format_func=lambda x: LABELS_VARIAVEIS.get(x, x)
    )
    
    mostrar_original = st.checkbox(
        "Mostrar série original", 
        value=True,
        help="Exibir a série original junto com a média móvel"
    )
    
    st.markdown("---")
    
    # Filtro de datas
    st.subheader("📅 Filtro de Período")
    usar_filtro_data = st.checkbox("Filtrar por período")

# Área principal
try:
    # Carrega os dados
    with st.spinner("Carregando dados..."):
        df, erro = carregar_dados(caminho_csv)
    
    if erro:
        st.error(erro)
        st.stop()
    
    # Valida os dados
    dados_validos, mensagem_validacao = validar_dados(df)
    if not dados_validos:
        st.error(mensagem_validacao)
        st.stop()
    
    # Filtro de datas (se habilitado)
    if usar_filtro_data:
        with st.sidebar:
            data_min = df['data'].min().date()
            data_max = df['data'].max().date()
            
            data_inicio = st.date_input(
                "Data de início:",
                value=data_min,
                min_value=data_min,
                max_value=data_max
            )
            
            data_fim = st.date_input(
                "Data de fim:",
                value=data_max,
                min_value=data_min,
                max_value=data_max
            )
            
            # Aplica o filtro
            df = df[(df['data'].dt.date >= data_inicio) & (df['data'].dt.date <= data_fim)]
    
    # Verifica se a variável existe no DataFrame
    if var_selecionada not in df.columns:
        st.error(f"A variável '{var_selecionada}' não foi encontrada no CSV.")
        st.info("Variáveis disponíveis no arquivo:")
        st.write(list(df.columns))
        st.stop()
    
    # Informações gerais dos dados
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📅 Período", 
            f"{df['data'].min().strftime('%d/%m/%Y')} - {df['data'].max().strftime('%d/%m/%Y')}"
        )
    
    with col2:
        st.metric("📊 Total de Registros", f"{len(df):,}")
    
    with col3:
        dias_periodo = (df['data'].max() - df['data'].min()).days
        st.metric("🗓️ Dias de Dados", f"{dias_periodo:,}")
    
    with col4:
        dados_faltantes = df[var_selecionada].isnull().sum()
        st.metric("❓ Dados Faltantes", f"{dados_faltantes:,}")
    
    st.markdown("---")
    
    # Gráfico principal
    st.subheader(f"📈 Análise: {LABELS_VARIAVEIS.get(var_selecionada, var_selecionada)}")
    
    fig = criar_grafico(df, var_selecionada, janela_movel, mostrar_original)
    st.plotly_chart(fig, use_container_width=True)
    
    # Estatísticas descritivas
    st.subheader("📊 Estatísticas Descritivas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stats = calcular_estatisticas(df, var_selecionada)
        
        st.markdown("**Estatísticas Gerais:**")
        st.write(f"• **Média:** {stats['média']:.2f}")
        st.write(f"• **Mediana:** {stats['mediana']:.2f}")
        st.write(f"• **Desvio Padrão:** {stats['desvio_padrão']:.2f}")
        st.write(f"• **Valor Mínimo:** {stats['mínimo']:.2f}")
        st.write(f"• **Valor Máximo:** {stats['máximo']:.2f}")
        st.write(f"• **Valores Nulos:** {stats['valores_nulos']}")
    
    with col2:
        # Gráfico de distribuição
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df[var_selecionada].dropna(),
            nbinsx=30,
            name="Distribuição",
            marker_color="lightblue"
        ))
        
        fig_hist.update_layout(
            title="Distribuição dos Valores",
            xaxis_title=LABELS_VARIAVEIS.get(var_selecionada, var_selecionada),
            yaxis_title="Frequência",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Tabela de dados recentes
    st.subheader("📋 Dados Recentes")
    
    # Mostra os últimos 10 registros
    colunas_exibir = ['data', var_selecionada]
    if f'{var_selecionada}_movel' in df.columns:
        colunas_exibir.append(f'{var_selecionada}_movel')
    
    df_recente = df[colunas_exibir].tail(10).copy()
    df_recente['data'] = df_recente['data'].dt.strftime('%d/%m/%Y')
    
    st.dataframe(
        df_recente,
        use_container_width=True,
        hide_index=True
    )
    
    # Opção de download
    st.subheader("💾 Download dos Dados")
    
    # Prepara dados para download
    df_download = df.copy()
    df_download[f'{var_selecionada}_media_movel_{janela_movel}d'] = calcular_media_movel(
        df_download, var_selecionada, janela_movel
    )
    
    csv_download = df_download.to_csv(index=False)
    
    st.download_button(
        label="📥 Baixar dados processados (CSV)",
        data=csv_download,
        file_name=f"dados_processados_{var_selecionada}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"Erro inesperado: {str(e)}")
    st.info("Verifique se o arquivo CSV existe e está no formato correto.")
    
    # Exemplo de formato esperado
    with st.expander("💡 Formato esperado do CSV"):
        st.code("""
data,temp_max,temp_min,temp_media,umid_max,umid_min,umid_mediana,rad_min,rad_max,rad_mediana,vento_raj_max,vento_vel_media,precipitacao_total
2024-01-01,25.5,18.2,21.8,85.0,45.0,65.0,100.0,800.0,450.0,15.2,8.5,0.0
2024-01-02,27.1,19.5,23.3,82.0,48.0,68.0,120.0,820.0,470.0,12.8,7.2,2.5
...
        """)
        
        st.markdown("""
        **Colunas obrigatórias:**
        - `data`: Data no formato YYYY-MM-DD
        - Pelo menos uma das variáveis climáticas listadas
        """)

# Rodapé
st.markdown("---")
st.markdown("*💡 Dica: Use a barra lateral para ajustar as configurações da análise.*")
