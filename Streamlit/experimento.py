import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import requests
import json
from datetime import datetime, timedelta
import warnings
import pytz

# Configurações
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="📊 Dashboard Clima x Vendas", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CONSTANTES ===
CAMPOS_NUMERICOS = [
    "valor_total", "valor_medio", "temp_max", "temp_min", "temp_media",
    "umid_max", "umid_min", "umid_mediana", "rad_min", "rad_max",
    "rad_mediana", "vento_raj_max", "vento_vel_media", "precipitacao_total"
]

ORDEM_DIAS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DIAS_PT = {
    "Monday": "Segunda", "Tuesday": "Terça", "Wednesday": "Quarta",
    "Thursday": "Quinta", "Friday": "Sexta", "Saturday": "Sábado", "Sunday": "Domingo"
}

VARIAVEIS_CLIMA = {
    "temp_max": "Temperatura Máxima (°C)",
    "temp_min": "Temperatura Mínima (°C)", 
    "temp_media": "Temperatura Média (°C)",
    "umid_max": "Umidade Máxima (%)",
    "umid_min": "Umidade Mínima (%)",
    "umid_mediana": "Umidade Mediana (%)",
    "rad_mediana": "Radiação Mediana",
    "vento_vel_media": "Velocidade do Vento (m/s)",
    "precipitacao_total": "Precipitação Total (mm)"
}

# Coordenadas do Recanto Maestro (aproximadas - ajuste conforme necessário)
RECANTO_MAESTRO_LAT = -29.4167  # Latitude aproximada para Restinga Seca/RS
RECANTO_MAESTRO_LON = -53.3833  # Longitude aproximada para Restinga Seca/RS

# === FUNÇÕES PARA DADOS METEOROLÓGICOS ===
@st.cache_data(ttl=3600)  # Cache por 1 hora
def obter_condicoes_atuais():
    """Obtém condições meteorológicas atuais usando Open-Meteo API."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": RECANTO_MAESTRO_LAT,
            "longitude": RECANTO_MAESTRO_LON,
            "current": [
                "temperature_2m", "relative_humidity_2m", "precipitation",
                "weather_code", "wind_speed_10m", "wind_direction_10m"
            ],
            "timezone": "America/Sao_Paulo"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "current" in data:
            current = data["current"]
            return {
                "temperatura": current.get("temperature_2m", "N/A"),
                "umidade": current.get("relative_humidity_2m", "N/A"),
                "precipitacao": current.get("precipitation", "N/A"),
                "vento_velocidade": current.get("wind_speed_10m", "N/A"),
                "vento_direcao": current.get("wind_direction_10m", "N/A"),
                "codigo_tempo": current.get("weather_code", 0),
                "timestamp": current.get("time", "")
            }
    except Exception as e:
        st.error(f"Erro ao obter condições atuais: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache por 1 hora
def obter_previsao_tempo(dias=7):
    """Obtém previsão do tempo usando Open-Meteo API (dados GFS)."""
    try:
        url = "https://api.open-meteo.com/v1/gfs"  # Usando modelo GFS especificamente
        params = {
            "latitude": RECANTO_MAESTRO_LAT,
            "longitude": RECANTO_MAESTRO_LON,
            "daily": [
                "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                "wind_speed_10m_max", "wind_direction_10m_dominant", "weather_code"
            ],
            "timezone": "America/Sao_Paulo",
            "forecast_days": dias
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "daily" in data:
            daily = data["daily"]
            previsao = []
            
            for i in range(len(daily["time"])):
                previsao.append({
                    "data": daily["time"][i],
                    "temp_max": daily["temperature_2m_max"][i],
                    "temp_min": daily["temperature_2m_min"][i],
                    "precipitacao": daily["precipitation_sum"][i],
                    "vento_max": daily["wind_speed_10m_max"][i],
                    "vento_direcao": daily["wind_direction_10m_dominant"][i],
                    "codigo_tempo": daily["weather_code"][i]
                })
            
            return pd.DataFrame(previsao)
            
    except Exception as e:
        st.error(f"Erro ao obter previsão: {str(e)}")
        return pd.DataFrame()

def interpretar_codigo_tempo(codigo):
    """Interpreta códigos de tempo WMO em descrições legíveis."""
    codigos = {
        0: "☀️ Céu limpo",
        1: "🌤️ Principalmente limpo",
        2: "⛅ Parcialmente nublado",
        3: "☁️ Nublado",
        45: "🌫️ Neblina",
        48: "🌫️ Neblina com geada",
        51: "🌦️ Garoa leve",
        53: "🌦️ Garoa moderada",
        55: "🌦️ Garoa intensa",
        61: "🌧️ Chuva leve",
        63: "🌧️ Chuva moderada",
        65: "🌧️ Chuva forte",
        71: "🌨️ Neve leve",
        73: "🌨️ Neve moderada",
        75: "🌨️ Neve forte",
        95: "⛈️ Tempestade",
        96: "⛈️ Tempestade com granizo"
    }
    return codigos.get(codigo, f"🌤️ Código {codigo}")

def criar_widget_clima_atual():
    """Cria widget com informações climáticas atuais."""
    condicoes = obter_condicoes_atuais()
    
    if condicoes:
        # Horário atual no fuso horário do Brasil
        tz_brasil = pytz.timezone('America/Sao_Paulo')
        agora = datetime.now(tz_brasil)
        
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            st.markdown("### 🕐 Horário Atual")
            st.markdown(f"**{agora.strftime('%H:%M:%S')}**")
            st.markdown(f"*{agora.strftime('%d/%m/%Y')}*")
        
        with col2:
            st.markdown("### 🌡️ Condições Atuais")
            condicao = interpretar_codigo_tempo(condicoes["codigo_tempo"])
            st.markdown(f"**{condicao}**")
            st.markdown(f"🌡️ {condicoes['temperatura']}°C")
        
        with col3:
            st.markdown("### 📊 Detalhes")
            st.markdown(f"💧 Umidade: {condicoes['umidade']}%")
            st.markdown(f"🌧️ Precipitação: {condicoes['precipitacao']} mm")
            st.markdown(f"💨 Vento: {condicoes['vento_velocidade']} km/h")
        
        st.markdown("---")
    else:
        st.warning("⚠️ Não foi possível obter dados meteorológicos atuais.")

# === FUNÇÕES AUXILIARES (mantidas do código original) ===
@st.cache_data
def carregar_dados():
    """Carrega e prepara os dados com tratamento robusto de erros."""
    try:
        # Tenta diferentes caminhos para o arquivo
        caminhos_possiveis = [
            os.path.join(os.path.dirname(__file__), "..", "dados_unificados_completos.csv"),
            "dados_unificados_completos.csv",
            os.path.join("data", "dados_unificados_completos.csv")
        ]
        
        df = None
        for caminho in caminhos_possiveis:
            if os.path.exists(caminho):
                df = pd.read_csv(caminho)
                break
        
        if df is None:
            raise FileNotFoundError("Arquivo de dados não encontrado em nenhum dos caminhos esperados")
        
        # Validação básica da estrutura
        colunas_obrigatorias = ["data", "valor_total"]
        for col in colunas_obrigatorias:
            if col not in df.columns:
                raise ValueError(f"Coluna obrigatória '{col}' não encontrada no arquivo")
        
        # Processamento de datas
        df["data"] = pd.to_datetime(df["data"], errors="coerce")
        df = df.dropna(subset=["data"])  # Remove linhas com datas inválidas
        df["data"] = df["data"].astype("datetime64[ns]")
        
        # Conversão de tipos numéricos com tratamento de erros
        for col in CAMPOS_NUMERICOS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Criação de variáveis derivadas
        df["dia_semana"] = df["data"].dt.day_name()
        df["dia_semana_pt"] = df["dia_semana"].map(DIAS_PT)
        df["dia_semana"] = pd.Categorical(df["dia_semana"], categories=ORDEM_DIAS, ordered=True)
        df["final_de_semana"] = df["dia_semana"].isin(["Friday", "Saturday", "Sunday"])
        df["mes"] = df["data"].dt.month
        df["ano"] = df["data"].dt.year
        df["trimestre"] = df["data"].dt.quarter
        
        # Variáveis climáticas derivadas
        if "precipitacao_total" in df.columns:
            df["choveu"] = df["precipitacao_total"] > 0
            df["chuva_categoria"] = pd.cut(
                df["precipitacao_total"], 
                bins=[-0.1, 0, 10, 50, float('inf')],
                labels=["Sem chuva", "Chuva leve", "Chuva moderada", "Chuva forte"]
            )
        
        # Remove linhas completamente vazias
        df = df.dropna(how='all')
        
        return df
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame()

def calcular_estatisticas_vendas(df):
    """Calcula estatísticas básicas de vendas."""
    if df.empty or "valor_total" not in df.columns:
        return {}
    
    return {
        "total_vendas": df["valor_total"].sum(),
        "media_diaria": df["valor_total"].mean(),
        "mediana_diaria": df["valor_total"].median(),
        "maior_venda": df["valor_total"].max(),
        "menor_venda": df["valor_total"].min(),
        "dias_com_vendas": len(df[df["valor_total"] > 0]),
        "total_dias": len(df)
    }

def criar_metricas_cards(stats):
    """Cria cards com métricas principais."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Total de Vendas", 
            f"R$ {stats['total_vendas']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
    
    with col2:
        st.metric(
            "📊 Média Diária", 
            f"R$ {stats['media_diaria']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
    
    with col3:
        st.metric(
            "🎯 Maior Venda", 
            f"R$ {stats['maior_venda']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
    
    with col4:
        st.metric(
            "📈 Taxa de Vendas", 
            f"{(stats['dias_com_vendas']/stats['total_dias']*100):.1f}%"
        )

def plotar_correlacao_matriz(df):
    """Plota matriz de correlação das variáveis numéricas."""
    colunas_numericas = [col for col in CAMPOS_NUMERICOS if col in df.columns]
    if len(colunas_numericas) < 2:
        st.warning("Dados insuficientes para matriz de correlação")
        return
    
    corr_matrix = df[colunas_numericas].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        cmap='RdYlBu_r',
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={"shrink": .8},
        ax=ax
    )
    
    ax.set_title('Matriz de Correlação - Variáveis Climáticas e Vendas', fontsize=14, pad=20)
    st.pyplot(fig)

# === CARREGAMENTO E VALIDAÇÃO ===
df = carregar_dados()

if df.empty:
    st.error("❌ Não foi possível carregar os dados. Verifique se o arquivo existe.")
    st.stop()

# === SIDEBAR COM FILTROS ===
with st.sidebar:
    st.header("🎛️ Filtros")
    
    # Filtro de data com validação melhorada
    data_min = df["data"].min().date()
    data_max = df["data"].max().date()
    
    data_inicio = st.date_input(
        "📅 Data inicial", 
        value=data_min,
        min_value=data_min,
        max_value=data_max
    )
    
    data_fim = st.date_input(
        "📅 Data final", 
        value=data_max,
        min_value=data_min,
        max_value=data_max
    )
    
    # Validação de intervalo
    if data_inicio > data_fim:
        st.error("⚠️ Data inicial deve ser anterior à data final")
        st.stop()
    
    # Filtro adicional por final de semana
    incluir_fds = st.checkbox("📅 Incluir finais de semana", value=True)
    
    # Filtro por faixa de valores
    if "valor_total" in df.columns and not df["valor_total"].isna().all():
        valor_min = float(df["valor_total"].min())
        valor_max = float(df["valor_total"].max())
        
        faixa_valores = st.slider(
            "💰 Faixa de valores (R$)",
            min_value=valor_min,
            max_value=valor_max,
            value=(valor_min, valor_max),
            format="R$ %.2f"
        )

# Aplicação dos filtros
df_filtrado = df[
    (df["data"] >= pd.to_datetime(data_inicio)) & 
    (df["data"] <= pd.to_datetime(data_fim))
].copy()

if not incluir_fds:
    df_filtrado = df_filtrado[~df_filtrado["final_de_semana"]]

if "valor_total" in df_filtrado.columns:
    df_filtrado = df_filtrado[
        (df_filtrado["valor_total"] >= faixa_valores[0]) &
        (df_filtrado["valor_total"] <= faixa_valores[1])
    ]

if df_filtrado.empty:
    st.warning("⚠️ Nenhum dado disponível para os filtros selecionados.")
    st.stop()

# === INTERFACE PRINCIPAL ===
st.title("📊 Dashboard Clima x Vendas")
st.markdown(f"**Período:** {data_inicio.strftime('%d/%m/%Y')} a {data_fim.strftime('%d/%m/%Y')} | **Registros:** {len(df_filtrado)}")

# === ABAS ===
tabs = st.tabs([
    "📌 Visão Geral", 
    "📅 Análise Temporal", 
    "🌧️ Análise Climática", 
    "📈 Correlações",
    "🔍 Insights Avançados",
    "🌤️ Previsão do Tempo"
])

# === ABA 1: VISÃO GERAL ===
with tabs[0]:
    st.header("📌 Resumo Executivo")
    
    # Widget com condições climáticas atuais
    criar_widget_clima_atual()
    
    # Métricas principais
    stats = calcular_estatisticas_vendas(df_filtrado)
    if stats:
        criar_metricas_cards(stats)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Evolução das Vendas")
        if "valor_total" in df_filtrado.columns:
            fig = px.line(
                df_filtrado, 
                x="data", 
                y="valor_total",
                title="Faturamento Diário",
                labels={"valor_total": "Faturamento (R$)", "data": "Data"}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Distribuição de Vendas")
        if "valor_total" in df_filtrado.columns:
            fig = px.histogram(
                df_filtrado, 
                x="valor_total",
                nbins=30,
                title="Distribuição dos Valores de Venda",
                labels={"valor_total": "Faturamento (R$)", "count": "Frequência"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tabela de dados com paginação
    st.subheader("🗂️ Dados Detalhados")
    st.dataframe(
        df_filtrado.head(200), 
        use_container_width=True,
        hide_index=True
    )

# === ABA 2: ANÁLISE TEMPORAL (mantida igual) ===
with tabs[1]:
    st.header("📅 Padrões Temporais")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vendas por Dia da Semana")
        if "dia_semana_pt" in df_filtrado.columns and "valor_total" in df_filtrado.columns:
            fig = px.box(
                df_filtrado, 
                x="dia_semana_pt", 
                y="valor_total",
                color="final_de_semana",
                title="Distribuição por Dia da Semana",
                labels={"valor_total": "Faturamento (R$)", "dia_semana_pt": "Dia da Semana"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Vendas por Mês")
        if "mes" in df_filtrado.columns and "valor_total" in df_filtrado.columns:
            # Agrupa vendas por mês
            vendas_mes = df_filtrado.groupby("mes")["valor_total"].agg(['sum', 'mean']).reset_index()
            
            # Cria DataFrame com todos os meses (1-12) para garantir continuidade
            meses_completos = pd.DataFrame({"mes": range(1, 13)})
            vendas_mes_completo = meses_completos.merge(vendas_mes, on="mes", how="left").fillna(0)
            
            # Adiciona nomes dos meses em português
            meses_nomes = {
                1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
                7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
            }
            vendas_mes_completo["mes_nome"] = vendas_mes_completo["mes"].map(meses_nomes)
            
            fig = px.bar(
                vendas_mes_completo, 
                x="mes_nome", 
                y="sum",
                title="Faturamento Total por Mês",
                labels={"sum": "Faturamento Total (R$)", "mes_nome": "Mês"},
                text="sum"
            )
            
            # Formata valores no gráfico
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig.update_layout(showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)

# === ABA 3: ANÁLISE CLIMÁTICA (mantida igual) ===
with tabs[2]:
    st.header("🌧️ Impacto do Clima nas Vendas")
    
    if "choveu" in df_filtrado.columns and "valor_total" in df_filtrado.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Vendas: Chuva vs Sem Chuva")
            vendas_chuva = df_filtrado.groupby("choveu")["valor_total"].agg(['mean', 'sum', 'count']).reset_index()
            vendas_chuva["choveu_label"] = vendas_chuva["choveu"].map({True: "Com Chuva", False: "Sem Chuva"})
            
            fig = px.bar(
                vendas_chuva, 
                x="choveu_label", 
                y="mean",
                title="Faturamento Médio por Condição",
                labels={"mean": "Faturamento Médio (R$)", "choveu_label": "Condição"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Distribuição por Intensidade de Chuva")
            def classificar_chuva(valor):
                if pd.isna(valor):
                    return "Sem dados"
                elif valor == 0:
                    return "Sem chuva"
                elif valor < 5:
                    return "Chuva leve (<5 mm/h)"
                elif valor <= 30:
                    return "Chuva moderada (5-30 mm/h)"
                else:
                    return "Chuva forte (>30 mm/h)"

            df_filtrado["faixa_chuva_personalizada"] = df_filtrado["precipitacao_total"].apply(classificar_chuva)

            if "faixa_chuva_personalizada" in df_filtrado.columns:
                fig = px.violin(
                    df_filtrado, 
                    x="faixa_chuva_personalizada", 
                    y="valor_total",
                    title="Vendas por Faixa de Intensidade de Chuva",
                    labels={
                        "faixa_chuva_personalizada": "Intensidade de Chuva",
                        "valor_total": "Faturamento (R$)"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

# === ABA 4: CORRELAÇÕES (mantida igual) ===
with tabs[3]:
    st.header("📈 Análise de Correlações")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Matriz de Correlação")
        plotar_correlacao_matriz(df_filtrado)
    
    with col2:
        st.subheader("Regressão Linear")
        variavel_selecionada = st.selectbox(
            "Variável Climática:", 
            options=list(VARIAVEIS_CLIMA.keys()),
            format_func=lambda x: VARIAVEIS_CLIMA[x]
        )
        
        if variavel_selecionada in df_filtrado.columns and "valor_total" in df_filtrado.columns:
            df_reg = df_filtrado.dropna(subset=["valor_total", variavel_selecionada])
            
            if len(df_reg) > 5:  # Mínimo de pontos para regressão
                try:
                    X = sm.add_constant(df_reg[variavel_selecionada])
                    y = df_reg["valor_total"]
                    modelo = sm.OLS(y, X).fit()
                    
                    r2 = modelo.rsquared
                    p_value = modelo.pvalues[variavel_selecionada]
                    
                    st.metric("R²", f"{r2:.3f}")
                    st.metric("P-value", f"{p_value:.3f}")
                    
                    # Interpretação
                    if p_value < 0.05:
                        st.success("✅ Correlação significativa")
                    else:
                        st.warning("⚠️ Correlação não significativa")
                        
                except Exception as e:
                    st.error(f"Erro na regressão: {str(e)}")
    
    # Gráfico de dispersão
    if variavel_selecionada in df_filtrado.columns and "valor_total" in df_filtrado.columns:
        st.subheader(f"Relação: {VARIAVEIS_CLIMA[variavel_selecionada]} vs Vendas")
        fig = px.scatter(
            df_filtrado, 
            x=variavel_selecionada, 
            y="valor_total",
            trendline="ols",
            title=f"Correlação: {VARIAVEIS_CLIMA[variavel_selecionada]} vs Faturamento"
        )
        st.plotly_chart(fig, use_container_width=True)
# === ABA 5: INSIGHTS AVANÇADOS ===
with tabs[4]:
    st.header("🔍 Insights e Análises Avançadas")
    
    # Top 10 melhores e piores dias
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Melhores Dias")
        if "valor_total" in df_filtrado.columns:
            top_dias = df_filtrado.nlargest(10, "valor_total")[["data", "valor_total", "dia_semana_pt"]]
            top_dias["valor_total_fmt"] = top_dias["valor_total"].apply(
                lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            )
            st.dataframe(top_dias[["data", "valor_total_fmt", "dia_semana_pt"]], hide_index=True)
    
    with col2:
        st.subheader("📉 Top 10 Piores Dias")
        if "valor_total" in df_filtrado.columns:
            bottom_dias = df_filtrado.nsmallest(10, "valor_total")[["data", "valor_total", "dia_semana_pt"]]
            bottom_dias["valor_total_fmt"] = bottom_dias["valor_total"].apply(
                lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            )
            st.dataframe(bottom_dias[["data", "valor_total_fmt", "dia_semana_pt"]], hide_index=True)
    
    # Análise de outliers
    st.subheader("📊 Análise de Outliers")
    if "valor_total" in df_filtrado.columns:
        Q1 = df_filtrado["valor_total"].quantile(0.25)
        Q3 = df_filtrado["valor_total"].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        outliers = df_filtrado[
            (df_filtrado["valor_total"] < limite_inferior) | 
            (df_filtrado["valor_total"] > limite_superior)
        ]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Total de Outliers", len(outliers))
        with col2:
            st.metric("📈 Outliers Superiores", len(outliers[outliers["valor_total"] > limite_superior]))
        with col3:
            st.metric("📉 Outliers Inferiores", len(outliers[outliers["valor_total"] < limite_inferior]))
        
        # Gráfico de boxplot para visualizar outliers
        fig = px.box(
            df_filtrado, 
            y="valor_total",
            title="Distribuição de Vendas com Outliers",
            labels={"valor_total": "Faturamento (R$)"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Análise de tendências
    st.subheader("📈 Análise de Tendências")
    if "valor_total" in df_filtrado.columns and len(df_filtrado) > 30:
        # Média móvel de 7 dias
        df_filtrado_sorted = df_filtrado.sort_values("data").copy()
        df_filtrado_sorted["media_movel_7d"] = df_filtrado_sorted["valor_total"].rolling(window=7, center=True).mean()
        df_filtrado_sorted["media_movel_30d"] = df_filtrado_sorted["valor_total"].rolling(window=30, center=True).mean()
        
        fig = go.Figure()
        
        # Vendas diárias
        fig.add_trace(go.Scatter(
            x=df_filtrado_sorted["data"],
            y=df_filtrado_sorted["valor_total"],
            mode='markers',
            name='Vendas Diárias',
            opacity=0.6,
            marker=dict(size=4)
        ))
        
        # Média móvel 7 dias
        fig.add_trace(go.Scatter(
            x=df_filtrado_sorted["data"],
            y=df_filtrado_sorted["media_movel_7d"],
            mode='lines',
            name='Média Móvel 7 dias',
            line=dict(color='red', width=2)
        ))
        
        # Média móvel 30 dias
        fig.add_trace(go.Scatter(
            x=df_filtrado_sorted["data"],
            y=df_filtrado_sorted["media_movel_30d"],
            mode='lines',
            name='Média Móvel 30 dias',
            line=dict(color='green', width=3)
        ))
        
        fig.update_layout(
            title="Tendências de Vendas com Médias Móveis",
            xaxis_title="Data",
            yaxis_title="Faturamento (R$)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Análise de sazonalidade
    st.subheader("🔄 Análise de Sazonalidade")
    if "mes" in df_filtrado.columns and "ano" in df_filtrado.columns:
        # Vendas por mês e ano
        vendas_sazonalidade = df_filtrado.groupby(["ano", "mes"])["valor_total"].sum().reset_index()
        vendas_pivot = vendas_sazonalidade.pivot(index="mes", columns="ano", values="valor_total").fillna(0)
        
        if not vendas_pivot.empty:
            fig = px.imshow(
                vendas_pivot.values,
                x=[str(col) for col in vendas_pivot.columns],
                y=[f"Mês {idx}" for idx in vendas_pivot.index],
                aspect="auto",
                color_continuous_scale="Blues",
                title="Mapa de Calor - Sazonalidade de Vendas"
            )
            fig.update_layout(
                xaxis_title="Ano",
                yaxis_title="Mês"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Insights automáticos
    st.subheader("🤖 Insights Automáticos")
    
    insights = []
    
    # Melhor dia da semana
    if "dia_semana_pt" in df_filtrado.columns and "valor_total" in df_filtrado.columns:
        vendas_dia_semana = df_filtrado.groupby("dia_semana_pt")["valor_total"].mean()
        melhor_dia = vendas_dia_semana.idxmax()
        insights.append(f"📅 **Melhor dia da semana:** {melhor_dia} (média de R$ {vendas_dia_semana.max():,.2f})")
    
    # Impacto da chuva
    if "choveu" in df_filtrado.columns and "valor_total" in df_filtrado.columns:
        vendas_com_chuva = df_filtrado[df_filtrado["choveu"] == True]["valor_total"].mean()
        vendas_sem_chuva = df_filtrado[df_filtrado["choveu"] == False]["valor_total"].mean()
        
        if not pd.isna(vendas_com_chuva) and not pd.isna(vendas_sem_chuva):
            diferenca_pct = ((vendas_com_chuva - vendas_sem_chuva) / vendas_sem_chuva) * 100
            if diferenca_pct > 0:
                insights.append(f"🌧️ **Chuva aumenta as vendas:** +{diferenca_pct:.1f}% em dias chuvosos")
            else:
                insights.append(f"☀️ **Chuva reduz as vendas:** {diferenca_pct:.1f}% em dias chuvosos")
    
    # Tendência geral
    if "valor_total" in df_filtrado.columns and len(df_filtrado) > 10:
        primeiro_terco = df_filtrado.head(len(df_filtrado)//3)["valor_total"].mean()
        ultimo_terco = df_filtrado.tail(len(df_filtrado)//3)["valor_total"].mean()
        
        if ultimo_terco > primeiro_terco:
            crescimento = ((ultimo_terco - primeiro_terco) / primeiro_terco) * 100
            insights.append(f"📈 **Tendência crescente:** +{crescimento:.1f}% no período")
        else:
            decrescimo = ((primeiro_terco - ultimo_terco) / primeiro_terco) * 100
            insights.append(f"📉 **Tendência decrescente:** -{decrescimo:.1f}% no período")
    
    # Variabilidade
    if "valor_total" in df_filtrado.columns:
        cv = df_filtrado["valor_total"].std() / df_filtrado["valor_total"].mean()
        if cv > 0.5:
            insights.append("⚠️ **Alta variabilidade** nas vendas diárias")
        elif cv < 0.2:
            insights.append("✅ **Vendas estáveis** com baixa variabilidade")
        else:
            insights.append("📊 **Variabilidade moderada** nas vendas")
    
    # Exibir insights
    for insight in insights:
        st.markdown(insight)

# === ABA 6: PREVISÃO DO TEMPO ===
with tabs[5]:
    st.header("🌤️ Previsão do Tempo")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📍 Localização")
        st.markdown(f"**Latitude:** {RECANTO_MAESTRO_LAT}")
        st.markdown(f"**Longitude:** {RECANTO_MAESTRO_LON}")
        st.markdown("**Região:** Restinga Seca, RS")
        
        # Seleção de dias para previsão
        dias_previsao = st.selectbox(
            "Dias de previsão:",
            options=[3, 5, 7, 10, 14],
            index=2  # 7 dias por padrão
        )
    
    with col2:
        st.subheader("🔄 Atualização")
        if st.button("Atualizar Previsão", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("*Dados atualizados automaticamente a cada hora*")
    
    # Obter e exibir previsão
    previsao_df = obter_previsao_tempo(dias_previsao)
    
    if not previsao_df.empty:
        # Converter data para datetime se necessário
        previsao_df["data"] = pd.to_datetime(previsao_df["data"])
        previsao_df["data_formatada"] = previsao_df["data"].dt.strftime("%d/%m")
        previsao_df["dia_semana"] = previsao_df["data"].dt.day_name().map(DIAS_PT)
        
        # Cards com previsão
        st.subheader("📅 Previsão dos Próximos Dias")
        
        # Dividir em colunas baseado no número de dias
        n_cols = min(len(previsao_df), 7)  # Máximo 7 colunas
        cols = st.columns(n_cols)
        
        for i, row in previsao_df.head(n_cols).iterrows():
            with cols[i]:
                # Interpretar condição do tempo
                condicao = interpretar_codigo_tempo(row["codigo_tempo"])
                
                st.markdown(f"""
                <div style="
                    border: 1px solid #ddd; 
                    border-radius: 10px; 
                    padding: 10px; 
                    text-align: center;
                    background-color: #f9f9f9;
                    margin-bottom: 10px;
                ">
                    <h4>{row['data_formatada']}</h4>
                    <p><strong>{row['dia_semana']}</strong></p>
                    <div style="font-size: 24px; margin: 10px 0;">
                        {condicao.split()[0]}
                    </div>
                    <p><strong>{row['temp_max']:.1f}°C / {row['temp_min']:.1f}°C</strong></p>
                    <p>🌧️ {row['precipitacao']:.1f}mm</p>
                    <p>💨 {row['vento_max']:.1f}km/h</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Gráfico de temperaturas
        st.subheader("🌡️ Evolução das Temperaturas")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=previsao_df["data_formatada"],
            y=previsao_df["temp_max"],
            mode='lines+markers',
            name='Temperatura Máxima',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=previsao_df["data_formatada"],
            y=previsao_df["temp_min"],
            mode='lines+markers',
            name='Temperatura Mínima',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)'
        ))
        
        fig.update_layout(
            title="Previsão de Temperaturas",
            xaxis_title="Data",
            yaxis_title="Temperatura (°C)",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de precipitação
        st.subheader("🌧️ Previsão de Precipitação")
        fig2 = px.bar(
            previsao_df,
            x="data_formatada",
            y="precipitacao",
            title="Precipitação Prevista por Dia",
            labels={"precipitacao": "Precipitação (mm)", "data_formatada": "Data"},
            text="precipitacao"
        )
        
        fig2.update_traces(texttemplate='%{text:.1f}mm', textposition='outside')
        fig2.update_layout(showlegend=False)
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Tabela detalhada
        st.subheader("📋 Detalhes da Previsão")
        previsao_display = previsao_df.copy()
        previsao_display["condicao"] = previsao_display["codigo_tempo"].apply(interpretar_codigo_tempo)
        
        colunas_display = [
            "data_formatada", "dia_semana", "condicao", 
            "temp_max", "temp_min", "precipitacao", "vento_max"
        ]
        
        # Renomear colunas para exibição
        colunas_renomeadas = {
            "data_formatada": "Data",
            "dia_semana": "Dia",
            "condicao": "Condição",
            "temp_max": "Temp. Máx (°C)",
            "temp_min": "Temp. Mín (°C)",
            "precipitacao": "Chuva (mm)",
            "vento_max": "Vento (km/h)"
        }
        
        previsao_display = previsao_display[colunas_display].rename(columns=colunas_renomeadas)
        
        st.dataframe(previsao_display, use_container_width=True, hide_index=True)
        
        # Alertas meteorológicos
        st.subheader("⚠️ Alertas Meteorológicos")
        alertas = []
        
        # Alerta de chuva forte
        chuva_forte = previsao_df[previsao_df["precipitacao"] > 50]
        if not chuva_forte.empty:
            datas_chuva = ", ".join(chuva_forte["data_formatada"].tolist())
            alertas.append(f"🌧️ **Chuva forte prevista:** {datas_chuva}")
        
        # Alerta de vento forte
        vento_forte = previsao_df[previsao_df["vento_max"] > 60]
        if not vento_forte.empty:
            datas_vento = ", ".join(vento_forte["data_formatada"].tolist())
            alertas.append(f"💨 **Vento forte previsto:** {datas_vento}")
        
        # Alerta de temperatura extrema
        temp_alta = previsao_df[previsao_df["temp_max"] > 35]
        temp_baixa = previsao_df[previsao_df["temp_min"] < 5]
        
        if not temp_alta.empty:
            datas_calor = ", ".join(temp_alta["data_formatada"].tolist())
            alertas.append(f"🔥 **Temperatura alta prevista:** {datas_calor}")
        
        if not temp_baixa.empty:
            datas_frio = ", ".join(temp_baixa["data_formatada"].tolist())
            alertas.append(f"🧊 **Temperatura baixa prevista:** {datas_frio}")
        
        if alertas:
            for alerta in alertas:
                st.warning(alerta)
        else:
            st.success("✅ Nenhum alerta meteorológico para o período")
        
    else:
        st.error("❌ Não foi possível obter a previsão do tempo. Tente novamente mais tarde.")
        
        # Informações sobre a fonte dos dados
        st.info("""
        **Fonte dos dados meteorológicos:** Open-Meteo API
        
        - **Modelo:** GFS (Global Forecast System)
        - **Atualização:** A cada 6 horas
        - **Precisão:** Dados são estimativas baseadas em modelos meteorológicos
        - **Cobertura:** Global, incluindo Brasil
        """)

# === RODAPÉ ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 50px;'>
    <p>📊 Dashboard Clima x Vendas | Desenvolvido para análise de correlações meteorológicas</p>
    <p>🌤️ Dados meteorológicos fornecidos por Open-Meteo API</p>
</div>
""", unsafe_allow_html=True)
