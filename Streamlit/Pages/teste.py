import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from datetime import datetime, timedelta
import warnings

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

# === FUNÇÕES AUXILIARES ===
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
    "🔍 Insights Avançados"
])

# === ABA 1: VISÃO GERAL ===
with tabs[0]:
    st.header("📌 Resumo Executivo")
    
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

# === ABA 2: ANÁLISE TEMPORAL ===
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
            
            # Debug: mostra dados de cada mês
            st.write("📋 **Resumo por Mês:**")
            debug_col1, debug_col2 = st.columns(2)
            with debug_col1:
                st.write("Dados originais:")
                st.dataframe(vendas_mes_completo[["mes", "mes_nome", "sum"]].rename(columns={"sum": "Total"}), hide_index=True)
            with debug_col2:
                meses_com_dados = df_filtrado["mes"].value_counts().sort_index()
                st.write("Registros por mês:")
                st.dataframe(pd.DataFrame({"Mês": meses_com_dados.index, "Registros": meses_com_dados.values}), hide_index=True)
            
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

# === ABA 3: ANÁLISE CLIMÁTICA ===
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
            if "chuva_categoria" in df_filtrado.columns:
                fig = px.violin(
                    df_filtrado, 
                    x="chuva_categoria", 
                    y="valor_total",
                    title="Vendas por Intensidade de Chuva"
                )
                st.plotly_chart(fig, use_container_width=True)

# === ABA 4: CORRELAÇÕES ===
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
    
    # Debug adicional para investigar problema do mês 11
    with st.expander("🔍 Debug - Análise de Dados por Mês"):
        if "mes" in df_filtrado.columns:
            debug_mensal = df_filtrado.groupby("mes").agg({
                "valor_total": ["count", "sum", "mean"],
                "data": ["min", "max"]
            }).round(2)
            debug_mensal.columns = ["Qtd_Registros", "Total_Vendas", "Media_Vendas", "Data_Min", "Data_Max"]
            st.dataframe(debug_mensal)
            
            # Verifica especificamente novembro
            dados_nov = df_filtrado[df_filtrado["mes"] == 11]
            st.write(f"**Registros em Novembro (mês 11):** {len(dados_nov)}")
            
            if len(dados_nov) > 0:
                st.write("Primeiras 5 linhas de novembro:")
                st.dataframe(dados_nov.head()[["data", "valor_total", "mes"]])
            else:
                st.warning("⚠️ Nenhum registro encontrado para novembro nos dados filtrados!")
                
                # Verifica se existe novembro nos dados originais
                dados_nov_original = df[df["mes"] == 11] if "mes" in df.columns else pd.DataFrame()
                st.write(f"**Registros de novembro nos dados originais:** {len(dados_nov_original)}")
                
                if len(dados_nov_original) > 0:
                    st.info("💡 Novembro existe nos dados originais, mas foi removido pelos filtros aplicados.")
                    st.write("Range de datas em novembro (dados originais):")
                    st.write(f"- Mínima: {dados_nov_original['data'].min()}")
                    st.write(f"- Máxima: {dados_nov_original['data'].max()}")
    
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
    
    # Análise de tendências
    st.subheader("📊 Análise de Tendências")
    if len(df_filtrado) > 30:  # Suficientes dados para tendência
        # Média móvel de 7 dias
        df_trend = df_filtrado.sort_values("data").copy()
        df_trend["media_movel_7d"] = df_trend["valor_total"].rolling(window=7, center=True).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_trend["data"], 
            y=df_trend["valor_total"],
            mode='markers',
            name='Vendas Diárias',
            opacity=0.6
        ))
        fig.add_trace(go.Scatter(
            x=df_trend["data"], 
            y=df_trend["media_movel_7d"],
            mode='lines',
            name='Média Móvel 7 dias',
            line=dict(width=3)
        ))
        fig.update_layout(title="Tendência de Vendas com Média Móvel")
        st.plotly_chart(fig, use_container_width=True)

# === FOOTER ===
st.markdown("---")
st.markdown("**Dashboard desenvolvido com Streamlit** | Dados atualizados automaticamente")
