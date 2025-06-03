import os
import requests
from datetime import datetime

# === Configurações iniciais ===
# Isto abaixa conforme dados de hj, foi feito ajuste pois os dados que fiz antes era de sexta feira, logo eram dados antigos. Tentei deixar automatzado.
data = datetime.utcnow().strftime('%Y%m%d')  # data de hoje em UTC
hora = '06'  # rodada do modelo: 00, 06, 12 ou 18

# Cria pasta NOMADS no mesmo diretório onde está o script
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "NOMADS")
os.makedirs(output_dir, exist_ok=True)

# URL base da API
base_url = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl'
dir_param = f'%2Fgfs.{data}%2F{hora}%2Fatmos'

# Parâmetros desejados
params = (
    '&var_TMP=on'
    '&var_TMAX=on'
    '&var_TMIN=on'
    '&var_UGRD=on'
    '&var_VGRD=on'
    '&var_RH=on'
    '&var_APCP=on'
    '&var_PRES=on'
    '&lev_2_m_above_ground=on'
    '&lev_10_m_above_ground=on'
    '&lev_surface=on'
    '&lev_mean_sea_level=on'
    '&subregion=&toplat=-29.63&leftlon=306.47&rightlon=306.57&bottomlat=-29.73'
)


for fstep in range(0, 121, 1):
    fstr = f'{fstep:03d}'
    nome_arquivo = f'gfs.t{hora}z.pgrb2.0p25.f{fstr}'
    
    url = f'{base_url}?file={nome_arquivo}{params}&dir={dir_param}'
    destino = os.path.join(output_dir, nome_arquivo)

    if os.path.exists(destino):
        print(f'✅ Já existe: {nome_arquivo}')
        continue

    print(f'⬇️ Baixando: {nome_arquivo}')
    resposta = requests.get(url, stream=True)
    
    if resposta.status_code == 200:
        with open(destino, 'wb') as f:
            for chunk in resposta.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f'✅ Salvo em: {destino}')
    else:
        print(f'❌ Erro ao baixar {nome_arquivo}: Status {resposta.status_code}')
