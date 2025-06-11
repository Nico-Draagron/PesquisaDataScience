import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.express as px
import os

# Config inicial
st.set_page_config("📊 Dashboard Clima x Vendas", layout="wide")

# === Função para carregar e preparar dados ===
@st.cache_data
def carregar_dados():
    arquivo = os.path.join(os.path.dirname(__file__), "..", "dados_unificados_completos.csv")
    df = pd.read_csv(arquivo)

    # Conversão segura da coluna de data
    df["data"] = pd.to_datetime(df["data"], errors="coerce")

    # Força o tipo datetime64[ns] para evitar erro com Arrow
    df["data"] = df["data"].astype("datetime64[ns]")

    # Conversão de tipos numéricos
    campos_numericos = [
        "valor_total", "valor_medio", "temp_max", "temp_min", "temp_media",
        "umid_max", "umid_min", "umid_mediana", "rad_min", "rad_max",
        "rad_mediana", "vento_raj_max", "vento_vel_media", "precipitacao_total"
    ]
    for col in campos_numericos:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derivadas
    df["dia_semana"] = df["data"].dt.day_name()
    ordem_dias = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["dia_semana"] = pd.Categorical(df["dia_semana"], categories=ordem_dias, ordered=True)
    df["final_de_semana"] = df["dia_semana"].isin(["Friday", "Saturday", "Sunday"])
    df["choveu"] = df["precipitacao_total"] > 0

    return df



# === Carregamento ===
df = carregar_dados()


# === Filtros Globais ===
st.sidebar.header("🎛️ Filtros")
data_inicio = st.sidebar.date_input("Data inicial", df["data"].min().date())
data_fim = st.sidebar.date_input("Data final", df["data"].max().date())

# Validação de datas
if data_inicio > data_fim:
    st.sidebar.error("⚠️ A data inicial não pode ser maior que a data final.")
    st.stop()

df = df[(df["data"] >= pd.to_datetime(data_inicio)) & (df["data"] <= pd.to_datetime(data_fim))]

# Verificação de DataFrame vazio
if df.empty:
    st.warning("⚠️ Nenhum dado disponível para o intervalo de datas selecionado.")
    st.stop()


# === Abas do dashboard ===
aba = st.selectbox("Escolha a análise:", [
    "📌 Visão Geral", 
    "📅 Dia da Semana", 
    "🌧️ Chuva vs Vendas", 
    "📈 Regressão Clima x Vendas"
])

# === VISÃO GERAL ===
if aba == "📌 Visão Geral":
    st.title("📌 Visão Geral dos Dados")
    st.subheader("📊 Tabela de dados")
    st.dataframe(df.head(100))

    st.subheader("📉 Vendas ao longo do tempo")
    fig = px.line(df, x="data", y="valor_total", title="Faturamento Diário")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Estatísticas")
    st.write(df.describe())

# === DIA DA SEMANA ===
elif aba == "📅 Dia da Semana":
    st.title("📅 Vendas por Dia da Semana")
    fig = px.box(df, x="dia_semana", y="valor_total", color="final_de_semana",
                 title="Distribuição de Vendas por Dia da Semana")
    st.plotly_chart(fig, use_container_width=True)

# === CHUVA ===
elif aba == "🌧️ Chuva vs Vendas":
    st.title("🌧️ Comparativo: Dias com e sem Chuva")

    # Filtra e remove nulos
    df_chuva = df.dropna(subset=["valor_total", "rad_mediana", "choveu"])

    # Converte booleano para string para facilitar legenda
    df_chuva["choveu_str"] = df_chuva["choveu"].map({True: "Com Chuva", False: "Sem Chuva"})

    fig, ax = plt.subplots(figsize=(10, 6))

    # Gráfico com regressão por categoria de chuva
    sns.scatterplot(
        data=df_chuva,
        x="rad_mediana", y="valor_total",
        hue="choveu_str",
        style="choveu_str",
        palette={"Com Chuva": "green", "Sem Chuva": "orange"},
        markers={"Com Chuva": "o", "Sem Chuva": "s"},
        ax=ax
    )

    # Regressão linear geral
    sns.regplot(
        data=df_chuva,
        x="rad_mediana", y="valor_total",
        scatter=False,
        color="blue",
        line_kws={"label": "Regressão Linear Geral"},
        ax=ax
    )

    ax.set_title("Chuva x Vendas com Destaques", fontsize=14, weight="bold")
    ax.set_xlabel("Radiação Mediana")
    ax.set_ylabel("Faturamento (R$)")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)


# === REGRESSÃO ===
elif aba == "📈 Regressão Clima x Vendas":
    st.title("📈 Regressão Linear: Clima vs Faturamento")

    variavel = st.selectbox("Escolha a variável climática:", [
        "temp_max", "temp_min", "temp_media", "umid_max", "umid_min",
        "rad_mediana", "vento_vel_media", "precipitacao_total"
    ])

    df_reg = df.dropna(subset=["valor_total", variavel])
    X = sm.add_constant(df_reg[variavel])
    y = df_reg["valor_total"]
    modelo = sm.OLS(y, X).fit()

    a = modelo.params["const"]
    b = modelo.params[variavel]
    r2 = modelo.rsquared
    eq = f"y = {a:.2f} + {b:.2f}x"
    r2_txt = f"R² = {r2:.3f}"

    st.markdown(f"**Equação:** {eq} | **{r2_txt}**")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(data=df_reg, x=variavel, y="valor_total", ax=ax, 
                scatter_kws={"color": "blue", "alpha": 0.6}, 
                line_kws={"color": "red"})
    ax.set_xlabel(variavel.replace("_", " ").title())
    ax.set_ylabel("Valor Total (R$)")
    ax.grid(True)
    st.pyplot(fig)
