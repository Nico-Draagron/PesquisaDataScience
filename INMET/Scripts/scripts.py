# 📦 Importar bibliotecas
import pandas as pd
import numpy as np
import glob
import os

# 📁 Caminho relativo da pasta com os arquivos CSV
BASE_DIR = os.path.dirname(__file__)
pasta_dados = os.path.join(BASE_DIR, '..', 'Dados_INMET', 'Dados_Brutos')
pasta_saida = os.path.join(BASE_DIR, '..', 'Outros')

# Criar pasta de saída se não existir
os.makedirs(pasta_saida, exist_ok=True)

# 📜 Listar todos os arquivos CSV da pasta
arquivos_csv = glob.glob(os.path.join(pasta_dados, '*.csv'))
print(f"📁 Localizados {len(arquivos_csv)} arquivos CSV em: {pasta_dados}")

# 🏷️ Renomeações e variáveis de interesse
renomear_colunas = {
    'Data': 'data',
    'Hora UTC': 'hora',
    'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'precipitacao',
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': 'pressao',
    'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)': 'pressao_max',
    'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)': 'pressao_min',
    'RADIACAO GLOBAL (Kj/m²)': 'radiacao',
    'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'temp_inst',
    'TEMPERATURA DO PONTO DE ORVALHO (°C)': 'temp_orvalho_inst',
    'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'temp_max',
    'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'temp_min',
    'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)': 'orvalho_max',
    'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)': 'orvalho_min',
    'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)': 'umidade_max',
    'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)': 'umidade_min',
    'UMIDADE RELATIVA DO AR, HORARIA (%)': 'umidade',
    'VENTO, DIREÇÃO HORARIA (gr) (° (gr))': 'vento_dir',
    'VENTO, RAJADA MAXIMA (m/s)': 'vento_raj',
    'VENTO, VELOCIDADE HORARIA (m/s)': 'vento_vel'
}

colunas_numericas = list(renomear_colunas.values())[2:]  # exclui data e hora

# 🔄 Processar todos os arquivos CSV
dfs = []
for arquivo in arquivos_csv:
    try:
        df = pd.read_csv(arquivo, skiprows=8, encoding='latin1', delimiter=';')
        df = df.rename(columns=renomear_colunas)

        if 'hora' in df.columns:
            df['hora'] = df['hora'].str.replace(' UTC', '', regex=False).str.zfill(4)
            df['hora'] = pd.to_datetime(df['hora'], format='%H%M', errors='coerce').dt.strftime('%H:%M')

        for col in colunas_numericas:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].replace(-999, 0 if col == 'radiacao' else np.nan)

        if 'data' in df.columns:
            df['data'] = pd.to_datetime(df['data'], format='%Y/%m/%d', errors='coerce')

        colunas_finais = ['data', 'hora'] + [
            col for col in renomear_colunas.values() if col in df.columns and col not in ['data', 'hora']
        ]
        df = df[colunas_finais]

        dfs.append(df)

    except Exception as e:
        print(f"❌ Erro ao processar {arquivo}: {e}")

# 📎 Unir todos os arquivos em um só DataFrame
df_final = pd.concat(dfs, ignore_index=True)

# 🧮 Ordenar por data e hora
df_final = df_final.sort_values(by=['data', 'hora'])

# 💾 Exportar dados tratados com colunas úteis e ordenadas
colunas_exportar = [
    'data', 'hora',
    'temp_max', 'temp_min', 'temp_inst',
    'umidade_max', 'umidade_min', 'umidade',
    'radiacao', 'vento_raj', 'vento_vel', 'precipitacao'
]

df_export = df_final[colunas_exportar]
df_export.to_csv(os.path.join(pasta_saida, '24h_dados_automaticos.csv'), index=False)
print("✔️ Arquivo 24h_dados_automaticos.csv exportado com sucesso!")

# 📊 Gerar resumo diário com estatísticas agregadas
resumo_diario = df_final.groupby('data').agg({
    'temp_max': 'max',
    'temp_min': 'min',
    'temp_inst': 'mean',
    'umidade_max': 'max',
    'umidade_min': 'min',
    'umidade': 'median',
    'radiacao': ['min', 'max', 'median'],
    'vento_raj': 'max',
    'vento_vel': 'mean',
    'precipitacao': 'sum'
})

resumo_diario.columns = [
    'temp_max', 'temp_min', 'temp_media',
    'umid_max', 'umid_min', 'umid_media',
    'rad_min', 'rad_max', 'rad_media',
    'vento_raj_max', 'vento_vel_media',
    'precipitacao_total'
]

# 🔁 Resetar índice
resumo_diario = resumo_diario.reset_index()

# 📊 Cálculo de dados faltantes por grupo reduzido
variaveis_para_validacao = {
    'temp_inst': 'temp_inst_faltantes_pct',
    'umidade': 'umidade_faltantes_pct',
    'radiacao': 'radiacao_faltantes_pct',
    'vento_vel': 'vento_vel_faltantes_pct',
    'precipitacao': 'precipitacao_faltantes_pct'
}

# Cálculo da porcentagem de dados faltantes
for coluna, nova_coluna in variaveis_para_validacao.items():
    total_por_dia = df_final.groupby('data')[coluna].count()
    total_esperado = 24
    total_faltantes = total_esperado - total_por_dia
    porcentagem_faltantes = (total_faltantes / total_esperado) * 100
    resumo_diario[nova_coluna] = resumo_diario['data'].map(porcentagem_faltantes).fillna(0)

# 🔢 Arredondar todas as colunas numéricas para 2 casas decimais
for col in resumo_diario.select_dtypes(include=['float', 'float64']).columns:
    resumo_diario[col] = resumo_diario[col].round(2)

for nova_coluna in variaveis_para_validacao.values():
    resumo_diario[nova_coluna] = resumo_diario[nova_coluna].astype(str) + '%'

# 📤 Exportar CSV com resumo diário simplificado
resumo_diario.to_csv(os.path.join(pasta_saida, 'resumo_diario_climatico.csv'), index=False)
print("📈 Resumo diário simplificado gerado com sucesso!")
