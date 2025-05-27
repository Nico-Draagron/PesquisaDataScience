import pandas as pd
import glob
import os

# 📁 Caminhos relativos
BASE_DIR = os.path.dirname(__file__)
pasta_dados = os.path.join(BASE_DIR, '..', 'Dados_INMET', 'Dados_convencionais')
pasta_saida = os.path.join(BASE_DIR, '..', 'Outros')

# Criar pasta de saída se não existir
os.makedirs(pasta_saida, exist_ok=True)


arquivos_csv = glob.glob(os.path.join(pasta_dados, '*.csv'))

dfs = []

for arquivo in arquivos_csv:
    print(f'📄 Lendo: {arquivo}')
    df = pd.read_csv(arquivo, encoding='utf-8-sig', delimiter=';')
    print('📋 Colunas:', df.columns.tolist())

    # Limpar nomes de colunas
    df.columns = [col.replace('"', '').strip() for col in df.columns]

    # Renomear colunas padronizadas
    df = df.rename(columns={
        'Data': 'data',
        'Hora (UTC)': 'hora_utc',
        'Temp. [Hora] (C)': 'temp_inst',
        'Umi. (%)': 'umidade',
        'Pressao (hPa)': 'pressao',
        'Chuva [Diaria] (mm)': 'precipitacao',
        'Insolacao (h)': 'insolacao',
        'Temp. Max. [Diaria] (h)': 'temp_max',
        'Temp. Min. [Diaria] (h)': 'temp_min',
        'Vel. Vento (m/s)': 'vento_vel'
    })

    # Converter data
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['data'])

    # Converter colunas numéricas
    colunas_num = ['temp_inst', 'umidade', 'pressao', 'precipitacao', 'insolacao', 'temp_max', 'temp_min', 'vento_vel']
    for col in colunas_num:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.strip()
                .str.replace(',', '.', regex=False)
                .replace(['-', 'Tr', '', 'NaN'], '0')
                .astype(float)
            )

    dfs.append(df)

# Concatenar tudo
df_final = pd.concat(dfs, ignore_index=True)

# Agrupar por data (resumo diário)
resumo_diario = df_final.groupby('data').agg({
    'temp_max': 'max',
    'temp_min': 'min',
    'temp_inst': 'mean',
    'umidade': ['min', 'max', 'mean'],
    'pressao': ['mean'],
    'precipitacao': 'sum',
    'insolacao': 'sum',
    'vento_vel': 'mean'
})

# Ajustar nomes de colunas
resumo_diario.columns = [
    'temp_max', 'temp_min', 'temp_media',
    'umid_min', 'umid_max', 'umid_media',
    'pressao_media', 'precipitacao_total',
    'insolacao_total', 'vento_vel_media'
]

# Resetar índice
resumo_diario = resumo_diario.reset_index()

# Formatando precipitação para string "x.x mm"
resumo_diario['precipitacao_total'] = (
    resumo_diario['precipitacao_total']
    .fillna(0)
    .round(1)
    .astype(str) + ' mm'
)

# Exportar para CSV na pasta Outros
saida_csv = os.path.join(pasta_saida, 'resumo_diario_convencional.csv')
resumo_diario.to_csv(saida_csv, index=False)

print(f"✅ Resumo diário das estações convencionais exportado com sucesso para: {saida_csv}")
