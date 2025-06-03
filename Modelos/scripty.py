import os
import xarray as xr
import pandas as pd
import numpy as np

# 📁 Caminho da pasta com os arquivos GRIB2 (ajuste conforme necessário)
pasta = "C:/Users/usuario/Desktop/projetos/TESTE/NOMADS"
arquivos = [f for f in os.listdir(pasta) if "pgrb2" in f and not f.endswith(".idx")]

# 📊 Lista para guardar os dados
dados = []

# 🔁 Loop pelos arquivos
for arquivo in arquivos:
    caminho = os.path.join(pasta, arquivo)
    try:
        # Abrir dataset completo, ignorando index cache para evitar .idx
        ds = xr.open_dataset(caminho, engine="cfgrib", backend_kwargs={"indexpath": ""})

        temperatura = ds['t2m'].mean().values - 273.15 if 't2m' in ds else np.nan
        umidade = ds['r2'].mean().values if 'r2' in ds else np.nan
        pressao = ds['sp'].mean().values / 100 if 'sp' in ds else np.nan
        precipitacao = ds['tp'].mean().values if 'tp' in ds else np.nan

        data = pd.to_datetime(ds.valid_time.values)

        dados.append({
            "data": data.date(),
            "hora": data.time(),
            "temperatura_C": round(float(temperatura), 2),
            "umidade_%": round(float(umidade), 2),
            "pressao_hPa": round(float(pressao), 2),
            "precipitacao_mm": round(float(precipitacao), 2)
        })

        print(f"✅ Processado: {arquivo}")

    except Exception as e:
        print(f"⚠️ Erro em {arquivo}: {e}")

# 📄 Salvar CSV
df = pd.DataFrame(dados)
csv_path = os.path.join(pasta, "dados_meteorologicos.csv")
df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"\n📁 CSV gerado em: {csv_path}")
