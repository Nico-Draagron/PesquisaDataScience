import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. 📂 Caminhos dos arquivos  
path = r'C:\Users\usuario\Desktop\projetos\PesquisaAMF\INMET_DADOS'
arquivo_auto = fr'{path}\resumo_diario_climatico.csv'
arquivo_conv = fr'{path}\resumo_diario_convencional.csv'

# 2. 📊 Ler os arquivos
df_auto = pd.read_csv(arquivo_auto, parse_dates=['data'])
df_conv = pd.read_csv(arquivo_conv, parse_dates=['data'])

# 3. 🔗 Merge pela data
df = pd.merge(df_auto, df_conv, on='data', suffixes=('_auto', '_conv'))

# 4. ✅ Variáveis para regressão
variaveis = ['temp_media', 'umid_media', 'vento_vel_media', 'rad_mediana', 'precipitacao_total']

for var in variaveis:
    col_auto = f'{var}_auto'
    col_conv = f'{var}_conv'
    

    if col_auto in df.columns and col_conv in df.columns:
        # Remover possíveis unidades como " mm", "%"
        df[col_auto] = df[col_auto].astype(str).str.extract(r'([-+]?\d*\.\d+|\d+)', expand=False)
        df[col_conv] = df[col_conv].astype(str).str.extract(r'([-+]?\d*\.\d+|\d+)', expand=False)
        df[col_auto] = pd.to_numeric(df[col_auto], errors='coerce')
        df[col_conv] = pd.to_numeric(df[col_conv], errors='coerce')

        X = df[[col_auto]].dropna()
        y = df.loc[X.index, col_conv]

        if len(X) > 0 and len(y) > 0:
            # 6. ⚙️ Treinar modelo
            modelo = LinearRegression()
            modelo.fit(X, y)

            # 7. 📊 Coeficientes
            a = modelo.coef_[0]
            b = modelo.intercept_
            r2 = modelo.score(X, y)

            print(f'✅ Regressão Linear ({var}): y = {a:.3f} * x + {b:.3f} | R² = {r2:.3f}')

            # 8. 🔮 Previsões
            df_predict = df[['data', col_auto]].dropna()
            df_predict[f'{var}_prevista'] = modelo.predict(df_predict[[col_auto]])

            # Juntar previsões
            df = df.merge(df_predict[['data', f'{var}_prevista']], on='data', how='left')

            # 9. 📊 Gerar Gráfico
            plt.figure(figsize=(10, 6))
            plt.scatter(df[col_auto], df[col_conv], alpha=0.5, label='Observado', marker='s')
            plt.plot(df[col_auto], df[f'{var}_prevista'], color='red', label='Regressão Linear')
            plt.xlabel(f'{var.replace("_", " ").capitalize()} Automática')
            plt.ylabel(f'{var.replace("_", " ").capitalize()} Convencional')
            plt.title(f'Regressão Linear - {var.replace("_", " ").capitalize()} (Auto vs Convencional)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # 📁 Salvar gráfico
            plt.savefig(fr'{path}\regressao_{var}.png')
            plt.show()

            # 💾 Exportar resultado
            df[[ 'data', col_auto, col_conv, f'{var}_prevista']].to_csv(fr'{path}\resultado_regressao_{var}.csv', index=False)
            print(f"✅ Resultado salvo como: resultado_regressao_{var}.csv e regressao_{var}.png\n")
        else:
            print(f"⚠️ Não há dados suficientes para {var}.")
    else:
        print(f"⚠️ Colunas {col_auto} ou {col_conv} não encontradas no DataFrame.")
