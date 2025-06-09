import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# Configurações da página
st.set_page_config(page_title="Análise Clima vs Vendas", layout="wide")
st.title("🌦️ Análise Interativa: Clima x Vendas")

# Carregar os dados
arquivo = os.path.join(os.path.dirname(__file__), "..", "dados_unificados_completos.csv")
try:
    df = pd.read_csv(arquivo, parse_dates=["data"])
except FileNotFoundError:
    st.error("❌ Arquivo 'dados_unificados_completos.csv' não encontrado no diretório.")
    st.stop()

# Garantir colunas numéricas
campos_numericos = [
    "valor_total", "valor_medio", "temp_max", "temp_min", "temp_media",
    "umid_max", "umid_min", "umid_mediana", "rad_min", "rad_max",
    "rad_mediana", "vento_raj_max", "vento_vel_media", "precipitacao_total"
]
for col in campos_numericos:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Filtros da barra lateral
st.sidebar.header("🎛️ Filtros")
variavel = st.sidebar.selectbox(
    "Escolha a variável climática:",
    ["temp_max", "temp_min", "temp_media", "umid_max", "umid_min",
     "rad_mediana", "vento_vel_media", "precipitacao_total"]
)
data_inicio = st.sidebar.date_input("Data inicial", df["data"].min().date())
data_fim = st.sidebar.date_input("Data final", df["data"].max().date())

# Filtrar dados
df = df[(df["data"] >= pd.to_datetime(data_inicio)) & (df["data"] <= pd.to_datetime(data_fim))]
df = df.dropna(subset=[variavel, "valor_total"])

# Regressão com statsmodels
X = df[variavel]
y = df["valor_total"]
X = sm.add_constant(X)
modelo = sm.OLS(y, X).fit()
a = modelo.params["const"]
b = modelo.params[variavel]
r2 = modelo.rsquared

# Gráfico com Seaborn
st.subheader(f"📈 Relação entre {variavel.replace('_', ' ').title()} e Faturamento")
fig, ax = plt.subplots(figsize=(10, 5))
sns.regplot(
data=df,  x=variavel, y="valor_total", ax=ax, marker='s',   scatter_kws={"s": 40},   line_kws={"color": "red"} )

# Adicionar fórmula e R² no gráfico
equacao = f"y = {a:.2f} + {b:.2f}x"
r_texto = f"R² = {r2:.3f}"
ax.text(0.05, 0.95, f"{equacao}\n{r_texto}",
        transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eef", edgecolor="gray"))

ax.set_xlabel(f"{variavel.replace('_', ' ').title()} (unidade)")
ax.set_ylabel("Valor Total (R$)")
ax.grid(True)
st.pyplot(fig)

# Tabela opcional
if st.checkbox("🔍 Mostrar dados usados"):
    st.dataframe(df[["data", variavel, "valor_total"]].reset_index(drop=True))
