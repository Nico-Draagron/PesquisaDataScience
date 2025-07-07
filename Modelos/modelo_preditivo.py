import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import joblib
import json
from pathlib import Path
warnings.filterwarnings('ignore')

# Bibliotecas de ML
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             ExtraTreesRegressor, VotingRegressor)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.utils import resample

# Bibliotecas avançadas (opcionais)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class ModeloVendasBootstrap:
    """
    Modelo de machine learning para predição de vendas com:
    - Bootstrap resampling para lidar com poucos dados
    - Variáveis específicas: dia semana, temperatura (min, média, max), chuva e radiação
    - Métricas: RMSE e R²
    - Importância das features
    """
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.modelos = {}
        self.ensemble_model = None
        self.scaler = None
        self.feature_names = None
        self.bootstrap_models = []
        self.bootstrap_scores = []
        self.metricas_finais = {}
        self.importancia_features = {}  # Nova adição
        
    def _get_default_config(self):
        """Configuração padrão do modelo"""
        return {
            'test_size': 0.2,
            'random_state': 42,
            'n_bootstrap': 100,  # Número de amostras bootstrap
            'bootstrap_sample_size': 1.0,  # Proporção do tamanho original
            'scaling_method': 'standard',
            'ensemble_voting': 'soft'
        }
    
    def preparar_features(self, df, modo_treino=True):
        """
        Prepara features específicas.
        Se `modo_treino=True`, exige coluna 'valor_total'.
        """
        print("🔧 Preparando features específicas...")
        df = df.copy()
        df['data'] = pd.to_datetime(df['data'])
        df = df.sort_values('data').reset_index(drop=True)

        df['dia_semana'] = df['data'].dt.dayofweek

        df_features = pd.DataFrame()
        df_features['dia_semana'] = df['dia_semana']

        # Temperaturas
        if 'temp_min' in df.columns:
            df_features['temp_min'] = df['temp_min']
        if 'temp_media' in df.columns:
            df_features['temp_media'] = df['temp_media']
        if 'temp_max' in df.columns:
            df_features['temp_max'] = df['temp_max']

        # Chuva
        if 'precipitacao_total' in df.columns:
            df_features['chuva'] = df['precipitacao_total']

        # Radiação
        if 'radiacao' in df.columns:
            df_features['radiacao'] = df['radiacao']
        else:
            df_features['radiacao'] = np.random.uniform(15, 30, len(df))

        # Apenas se for treino: adicionar o target
        if modo_treino:
            if 'valor_total' in df.columns:
                df_features['valor_total'] = df['valor_total']
            else:
                raise ValueError("Coluna 'valor_total' não encontrada")

        df_features['data'] = df['data']
        df_features = df_features.dropna()

        print(f"✅ Features preparadas: {df_features.shape[0]} amostras")
        return df_features

    def _modelo_tem_feature_importance(self, modelo):
        """Verifica se o modelo tem capacidade de calcular importância das features"""
        return hasattr(modelo, 'feature_importances_')
    
    def _calcular_importancia_features(self, modelos_bootstrap, nome_modelo):
        """
        Calcula a importância das features para modelos que suportam
        """
        if not modelos_bootstrap:
            return None
            
        # Verificar se pelo menos um modelo tem feature_importances_
        tem_importancia = any(self._modelo_tem_feature_importance(m) for m in modelos_bootstrap)
        
        if not tem_importancia:
            return None
        
        # Coletar importâncias de todos os modelos bootstrap
        importancias_bootstrap = []
        for modelo in modelos_bootstrap:
            if self._modelo_tem_feature_importance(modelo):
                importancias_bootstrap.append(modelo.feature_importances_)
        
        if not importancias_bootstrap:
            return None
        
        # Calcular estatísticas da importância
        importancias_array = np.array(importancias_bootstrap)
        importancia_media = np.mean(importancias_array, axis=0)
        importancia_std = np.std(importancias_array, axis=0)
        
        # Mapear para nomes das features
        importancia_dict = {}
        for i, feature_name in enumerate(self.feature_names):
            importancia_dict[feature_name] = {
                'media': importancia_media[i],
                'std': importancia_std[i]
            }
        
        return importancia_dict
    
    def _agregar_importancia_por_categoria(self, importancia_dict):
        """
        Agrega a importância das features por categoria solicitada:
        - dia_semana
        - temperatura (temp_min + temp_media + temp_max)
        - chuva
        - radiacao
        """
        if not importancia_dict:
            return None
        
        categorias = {
            'dia_semana': 0.0,
            'temperatura': 0.0,
            'chuva': 0.0,
            'radiacao': 0.0
        }
        
        # Mapear features para categorias
        for feature, valores in importancia_dict.items():
            if feature == 'dia_semana':
                categorias['dia_semana'] = valores['media']
            elif feature in ['temp_min', 'temp_media', 'temp_max']:
                categorias['temperatura'] += valores['media']
            elif feature == 'chuva':
                categorias['chuva'] = valores['media']
            elif feature == 'radiacao':
                categorias['radiacao'] = valores['media']
        
        # Normalizar para que soma seja 100%
        total = sum(categorias.values())
        if total > 0:
            for categoria in categorias:
                categorias[categoria] = (categorias[categoria] / total) * 100
        
        return categorias
    
    def criar_modelos_base(self):
        """
        Cria diferentes modelos para o ensemble
        """
        modelos = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=self.config['random_state']),
            'Lasso': Lasso(alpha=1.0, random_state=self.config['random_state']),
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=self.config['random_state']),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=self.config['random_state'], n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=self.config['random_state'], n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=self.config['random_state']
            ),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'MLP': MLPRegressor(
                hidden_layer_sizes=(50, 25), max_iter=500,
                random_state=self.config['random_state'], early_stopping=True
            )
        }
        
        # Adicionar modelos avançados se disponíveis
        if XGBOOST_AVAILABLE:
            modelos['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=self.config['random_state'], n_jobs=-1, verbosity=0
            )
        
        if LIGHTGBM_AVAILABLE:
            modelos['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=self.config['random_state'], verbose=-1, n_jobs=-1
            )
        
        return modelos
    
    def bootstrap_training(self, X_train, y_train, modelo, nome_modelo):
        """
        Treina modelo usando Bootstrap resampling
        """
        n_samples = len(X_train)
        sample_size = int(n_samples * self.config['bootstrap_sample_size'])
        
        bootstrap_predictions = []
        bootstrap_scores_rmse = []
        bootstrap_scores_r2 = []
        
        print(f"   🔄 Bootstrap para {nome_modelo}: {self.config['n_bootstrap']} iterações")
        
        for i in range(self.config['n_bootstrap']):
            # Criar amostra bootstrap
            indices = resample(range(n_samples), replace=True, n_samples=sample_size, 
                             random_state=self.config['random_state'] + i)
            
            X_bootstrap = X_train.iloc[indices]
            y_bootstrap = y_train.iloc[indices]
            
            # Treinar modelo
            modelo_clone = sklearn.base.clone(modelo)
            modelo_clone.fit(X_bootstrap, y_bootstrap)
            
            # Avaliar no conjunto out-of-bag (OOB)
            oob_indices = list(set(range(n_samples)) - set(indices))
            if len(oob_indices) > 0:
                X_oob = X_train.iloc[oob_indices]
                y_oob = y_train.iloc[oob_indices]
                
                y_pred_oob = modelo_clone.predict(X_oob)
                
                rmse = np.sqrt(mean_squared_error(y_oob, y_pred_oob))
                r2 = r2_score(y_oob, y_pred_oob)
                
                bootstrap_scores_rmse.append(rmse)
                bootstrap_scores_r2.append(r2)
            
            bootstrap_predictions.append(modelo_clone)
        
        # Estatísticas do Bootstrap
        rmse_mean = np.mean(bootstrap_scores_rmse)
        rmse_std = np.std(bootstrap_scores_rmse)
        r2_mean = np.mean(bootstrap_scores_r2)
        r2_std = np.std(bootstrap_scores_r2)
        
        # Intervalos de confiança (95%)
        rmse_ci_lower = np.percentile(bootstrap_scores_rmse, 2.5)
        rmse_ci_upper = np.percentile(bootstrap_scores_rmse, 97.5)
        r2_ci_lower = np.percentile(bootstrap_scores_r2, 2.5)
        r2_ci_upper = np.percentile(bootstrap_scores_r2, 97.5)
        
        return {
            'modelos': bootstrap_predictions,
            'rmse_mean': rmse_mean,
            'rmse_std': rmse_std,
            'rmse_ci': (rmse_ci_lower, rmse_ci_upper),
            'r2_mean': r2_mean,
            'r2_std': r2_std,
            'r2_ci': (r2_ci_lower, r2_ci_upper)
        }
    
    def treinar(self, df):
        """
        Treinamento completo do modelo com Bootstrap
        """
        print("🚀 Iniciando treinamento com Bootstrap resampling...")
        
        # Preparar features específicas
        df_preparado = self.preparar_features(df)
        
        # Separar features e target
        feature_cols = ['dia_semana', 'temp_min', 'temp_media', 'temp_max', 'chuva', 'radiacao']
        X = df_preparado[feature_cols]
        y = df_preparado['valor_total']
        
        self.feature_names = feature_cols
        
        # Divisão temporal dos dados
        split_idx = int(len(X) * (1 - self.config['test_size']))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"📊 Divisão dos dados - Treino: {len(X_train)}, Teste: {len(X_test)}")
        
        # Normalização dos dados
        self.scaler = self._create_scaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Criar e treinar modelos com Bootstrap
        base_models = self.criar_modelos_base()
        self.bootstrap_models = {}
        self.metricas_finais = {}
        self.importancia_features = {}  # Resetar importâncias
        
        for nome, modelo in base_models.items():
            print(f"\n🔧 Treinando {nome} com Bootstrap...")
            
            try:
                # Treinar com Bootstrap
                resultado_bootstrap = self.bootstrap_training(
                    X_train_scaled, y_train, modelo, nome
                )
                
                self.bootstrap_models[nome] = resultado_bootstrap['modelos']
                
                # Calcular importância das features
                importancia_raw = self._calcular_importancia_features(
                    resultado_bootstrap['modelos'], nome
                )
                importancia_agregada = self._agregar_importancia_por_categoria(importancia_raw)
                self.importancia_features[nome] = importancia_agregada
                
                # Fazer predições no conjunto de teste usando ensemble dos modelos bootstrap
                y_pred_test_ensemble = np.mean([
                    m.predict(X_test_scaled) for m in resultado_bootstrap['modelos']
                ], axis=0)
                
                # Calcular métricas finais no conjunto de teste
                rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test_ensemble))
                r2_test = r2_score(y_test, y_pred_test_ensemble)
                
                self.metricas_finais[nome] = {
                    'bootstrap_rmse_mean': resultado_bootstrap['rmse_mean'],
                    'bootstrap_rmse_std': resultado_bootstrap['rmse_std'],
                    'bootstrap_rmse_ci': resultado_bootstrap['rmse_ci'],
                    'bootstrap_r2_mean': resultado_bootstrap['r2_mean'],
                    'bootstrap_r2_std': resultado_bootstrap['r2_std'],
                    'bootstrap_r2_ci': resultado_bootstrap['r2_ci'],
                    'test_rmse': rmse_test,
                    'test_r2': r2_test
                }
                
                print(f"  ✅ Bootstrap RMSE: {resultado_bootstrap['rmse_mean']:.2f} (±{resultado_bootstrap['rmse_std']:.2f})")
                print(f"     IC 95%: [{resultado_bootstrap['rmse_ci'][0]:.2f}, {resultado_bootstrap['rmse_ci'][1]:.2f}]")
                print(f"  ✅ Bootstrap R²: {resultado_bootstrap['r2_mean']:.3f} (±{resultado_bootstrap['r2_std']:.3f})")
                print(f"     IC 95%: [{resultado_bootstrap['r2_ci'][0]:.3f}, {resultado_bootstrap['r2_ci'][1]:.3f}]")
                print(f"  📊 Teste Final - RMSE: {rmse_test:.2f}, R²: {r2_test:.3f}")
                
                # Mostrar importância das features se disponível
                if importancia_agregada:
                    print(f"  🎯 Importância das Features:")
                    for categoria, valor in importancia_agregada.items():
                        print(f"     {categoria}: {valor:.1f}%")
                
            except Exception as e:
                print(f"  ❌ Erro ao treinar {nome}: {str(e)}")
        
        # Criar ensemble final
        self._criar_ensemble_final()
        
        print("\n✅ Treinamento concluído!")
        return self.metricas_finais
    
    def _create_scaler(self):
        """Cria o normalizador apropriado"""
        if self.config['scaling_method'] == 'standard':
            return StandardScaler()
        elif self.config['scaling_method'] == 'minmax':
            return MinMaxScaler()
        elif self.config['scaling_method'] == 'robust':
            return RobustScaler()
        else:
            return StandardScaler()
    
    def _criar_ensemble_final(self):
        """
        Cria ensemble com os melhores modelos baseado no R² do teste
        """
        if len(self.metricas_finais) < 2:
            return
        
        # Selecionar os 3 melhores modelos baseados no R² do teste
        sorted_models = sorted(
            self.metricas_finais.items(), 
            key=lambda x: x[1]['test_r2'], 
            reverse=True
        )[:3]
        
        self.ensemble_selecionado = [nome for nome, _ in sorted_models]
        
        print(f"\n🎯 Ensemble criado com os melhores modelos: {self.ensemble_selecionado}")
        print("   Baseado nas métricas de R² no conjunto de teste")
    
    def _calcular_importancia_ensemble(self):
        """
        Calcula a importância média das features do ensemble
        """
        if not hasattr(self, 'ensemble_selecionado') or not self.ensemble_selecionado:
            return None
        
        importancias_ensemble = {}
        for categoria in ['dia_semana', 'temperatura', 'chuva', 'radiacao']:
            importancias_ensemble[categoria] = 0.0
        
        # Média ponderada das importâncias dos modelos do ensemble
        for nome_modelo in self.ensemble_selecionado:
            if nome_modelo in self.importancia_features and self.importancia_features[nome_modelo]:
                peso = 1.0 / len(self.ensemble_selecionado)  # Peso igual para todos
                for categoria, valor in self.importancia_features[nome_modelo].items():
                    importancias_ensemble[categoria] += valor * peso
        
        return importancias_ensemble
    
    def prever(self, dados_futuros, usar_ensemble=True, retornar_intervalo=True):
        """
        Faz predições para dados futuros com intervalos de confiança
        """
        # Preparar dados
        dados_preparados = self.preparar_features(dados_futuros, modo_treino=False)
        X_futuro = dados_preparados[self.feature_names]
        X_futuro_scaled = pd.DataFrame(
            self.scaler.transform(X_futuro),
            columns=X_futuro.columns,
            index=X_futuro.index
        )
        
        if usar_ensemble and hasattr(self, 'ensemble_selecionado'):
            # Usar ensemble dos melhores modelos
            todas_predicoes = []
            
            for nome_modelo in self.ensemble_selecionado:
                modelos_bootstrap = self.bootstrap_models[nome_modelo]
                # Predições de todos os modelos bootstrap
                predicoes_modelo = np.array([
                    m.predict(X_futuro_scaled) for m in modelos_bootstrap
                ])
                todas_predicoes.extend(predicoes_modelo)
            
            todas_predicoes = np.array(todas_predicoes)
            
            # Calcular predição média e intervalos
            predicao_media = np.mean(todas_predicoes, axis=0)
            predicao_lower = np.percentile(todas_predicoes, 2.5, axis=0)
            predicao_upper = np.percentile(todas_predicoes, 97.5, axis=0)
            
            print(f"🔮 Predições feitas usando Ensemble: {self.ensemble_selecionado}")
            
        else:
            # Usar o melhor modelo individual
            melhor_modelo = max(self.metricas_finais.items(), 
                              key=lambda x: x[1]['test_r2'])
            nome_modelo = melhor_modelo[0]
            modelos_bootstrap = self.bootstrap_models[nome_modelo]
            
            # Predições de todos os modelos bootstrap
            todas_predicoes = np.array([
                m.predict(X_futuro_scaled) for m in modelos_bootstrap
            ])
            
            predicao_media = np.mean(todas_predicoes, axis=0)
            predicao_lower = np.percentile(todas_predicoes, 2.5, axis=0)
            predicao_upper = np.percentile(todas_predicoes, 97.5, axis=0)
            
            print(f"🔮 Predições feitas usando {nome_modelo}")
        
        if retornar_intervalo:
            return {
                'predicao': predicao_media,
                'intervalo_inferior': predicao_lower,
                'intervalo_superior': predicao_upper
            }
        else:
            return predicao_media
    
    def gerar_relatorio_completo(self):
        """
        Gera relatório detalhado com foco em RMSE, R² e importância das features
        """
        relatorio = {
            'configuracao': {
                'n_bootstrap': self.config['n_bootstrap'],
                'bootstrap_sample_size': self.config['bootstrap_sample_size'],
                'test_size': self.config['test_size'],
                'features_utilizadas': self.feature_names
            },
            'metricas_por_modelo': self.metricas_finais,
            'melhor_modelo': max(
                self.metricas_finais.items(), 
                key=lambda x: x[1]['test_r2']
            ),
            'ensemble_selecionado': self.ensemble_selecionado if hasattr(self, 'ensemble_selecionado') else None,
            'importancia_features': self.importancia_features,
            'importancia_ensemble': self._calcular_importancia_ensemble()
        }
        
        return relatorio
    
    def salvar_modelo(self, caminho='modelo_vendas_bootstrap.pkl'):
        """Salva o modelo completo"""
        modelo_data = {
            'bootstrap_models': self.bootstrap_models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config,
            'metricas_finais': self.metricas_finais,
            'ensemble_selecionado': self.ensemble_selecionado if hasattr(self, 'ensemble_selecionado') else None,
            'importancia_features': self.importancia_features
        }
        
        joblib.dump(modelo_data, caminho)
        print(f"✅ Modelo salvo em: {caminho}")
    
    @classmethod
    def carregar_modelo(cls, caminho='modelo_vendas_bootstrap.pkl'):
        """Carrega modelo salvo"""
        modelo_data = joblib.load(caminho)
        
        instance = cls(modelo_data['config'])
        instance.bootstrap_models = modelo_data['bootstrap_models']
        instance.scaler = modelo_data['scaler']
        instance.feature_names = modelo_data['feature_names']
        instance.metricas_finais = modelo_data['metricas_finais']
        if 'ensemble_selecionado' in modelo_data:
            instance.ensemble_selecionado = modelo_data['ensemble_selecionado']
        if 'importancia_features' in modelo_data:
            instance.importancia_features = modelo_data['importancia_features']
        
        print(f"✅ Modelo carregado de: {caminho}")
        return instance

# Importar sklearn.base para o clone
import sklearn.base

# Função para criar dados sintéticos de exemplo
def criar_dados_exemplo():
    """Cria dados sintéticos para demonstração"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-06-30', freq='D')
    
    dados = []
    for i, date in enumerate(dates):
        # Padrões sazonais e semanais
        dia_semana = date.weekday()
        dia_ano = date.dayofyear
        
        # Vendas base com padrões
        vendas_base = 50000
        if dia_semana >= 5:  # Fins de semana
            vendas_base *= 1.3
        
        # Efeito da temperatura
        temp_base = 25 + 10 * np.sin(2 * np.pi * dia_ano / 365.25)
        temp_media = temp_base + np.random.normal(0, 3)
        temp_min = temp_media - np.random.uniform(3, 7)
        temp_max = temp_media + np.random.uniform(3, 7)
        
        # Vendas aumentam com temperatura moderada (20-28°C)
        if 20 <= temp_media <= 28:
            vendas_base *= 1.1
        elif temp_media > 32 or temp_media < 15:
            vendas_base *= 0.9
        
        # Efeito da chuva
        chuva = max(0, np.random.exponential(2) if np.random.random() < 0.3 else 0)
        if chuva > 10:
            vendas_base *= 0.8
        elif 1 < chuva <= 10:
            vendas_base *= 0.95
        
        # Radiação solar (simulada)
        radiacao = max(10, 25 + 10 * np.sin(2 * np.pi * dia_ano / 365.25) + np.random.normal(0, 5))
        
        # Vendas finais com ruído
        vendas = vendas_base * (1 + np.random.normal(0, 0.1))
        
        dados.append({
            'data': date,
            'valor_total': vendas,
            'temp_min': temp_min,
            'temp_media': temp_media,
            'temp_max': temp_max,
            'precipitacao_total': chuva,
            'radiacao': radiacao
        })
    
    return pd.DataFrame(dados)

# Exemplo de uso
def exemplo_completo():
    """Exemplo completo de uso do modelo com Bootstrap"""
    print("🚀 Exemplo de Modelo com Bootstrap Resampling")
    print("=" * 60)
    
    # Criar ou carregar dados
    try:
        df = pd.read_csv('dados_unificados_completos.csv', parse_dates=['data'])
        print("✅ Dados reais carregados")
    except FileNotFoundError:
        print("⚠️ Criando dados sintéticos para demonstração...")
        df = criar_dados_exemplo()
    
    # Configurar modelo
    config = {
        'test_size': 0.2,
        'random_state': 42,
        'n_bootstrap': 100,  # 100 iterações bootstrap
        'bootstrap_sample_size': 0.8,  # 80% do tamanho original em cada amostra
        'scaling_method': 'standard'
    }
    
    # Criar e treinar modelo
    modelo = ModeloVendasBootstrap(config)
    metricas = modelo.treinar(df)
    
    # Gerar relatório
    relatorio = modelo.gerar_relatorio_completo()
    
    print("\n" + "="*60)
    print("📊 RELATÓRIO FINAL - MÉTRICAS RMSE e R²")
    print("="*60)
    
    print(f"\n🎯 Melhor Modelo: {relatorio['melhor_modelo'][0]}")
    melhor_metricas = relatorio['melhor_modelo'][1]
    print(f"   RMSE (Bootstrap): {melhor_metricas['bootstrap_rmse_mean']:.2f} ± {melhor_metricas['bootstrap_rmse_std']:.2f}")
    print(f"   R² (Bootstrap): {melhor_metricas['bootstrap_r2_mean']:.3f} ± {melhor_metricas['bootstrap_r2_std']:.3f}")
    print(f"   RMSE (Teste Final): {melhor_metricas['test_rmse']:.2f}")
    print(f"   R² (Teste Final): {melhor_metricas['test_r2']:.3f}")
    
    print(f"\n📈 Todos os Modelos Treinados:")
    for nome, metricas in relatorio['metricas_por_modelo'].items():
        print(f"\n   {nome}:")
        print(f"      RMSE: {metricas['test_rmse']:.2f}")
        print(f"      R²: {metricas['test_r2']:.3f}")
        print(f"      IC 95% R²: [{metricas['bootstrap_r2_ci'][0]:.3f}, {metricas['bootstrap_r2_ci'][1]:.3f}]")
    
    print(f"\n🔧 Features Utilizadas: {', '.join(relatorio['configuracao']['features_utilizadas'])}")
    
    # Mostrar importância das features do ensemble
    print("\n" + "="*60)
    print("🎯 IMPORTÂNCIA DAS FEATURES")
    print("="*60)
    
    if relatorio['importancia_ensemble']:
        print("\n📊 Importância das Features do Ensemble:")
        for categoria, valor in relatorio['importancia_ensemble'].items():
            print(f"   {categoria.capitalize()}: {valor:.1f}%")
        
        # Ranking das features
        importancia_sorted = sorted(relatorio['importancia_ensemble'].items(), 
                                   key=lambda x: x[1], reverse=True)
        print(f"\n🏆 Ranking de Importância:")
        for i, (categoria, valor) in enumerate(importancia_sorted, 1):
            print(f"   {i}º {categoria.capitalize()}: {valor:.1f}%")
    
    # Mostrar importância por modelo individual
    print(f"\n📋 Importância por Modelo Individual:")
    for nome_modelo, importancia in relatorio['importancia_features'].items():
        if importancia:
            print(f"\n   {nome_modelo}:")
            for categoria, valor in importancia.items():
                print(f"      {categoria}: {valor:.1f}%")
    
    # Fazer predições para próximos dias
    print("\n🔮 Fazendo predições para os próximos 7 dias...")
    
    # Criar dados futuros de exemplo
    ultimo_dia = df['data'].max()
    datas_futuras = pd.date_range(ultimo_dia + timedelta(days=1), periods=7, freq='D')
    
    dados_futuros = []
    for data in datas_futuras:
        dados_futuros.append({
            'data': data,
            'temp_min': np.random.uniform(18, 22),
            'temp_media': np.random.uniform(23, 28),
            'temp_max': np.random.uniform(28, 35),
            'precipitacao_total': max(0, np.random.exponential(2) if np.random.random() < 0.3 else 0),
            'radiacao': np.random.uniform(20, 30)
        })
    
    df_futuro = pd.DataFrame(dados_futuros)
    
    # Fazer predições com intervalos
    resultados = modelo.prever(df_futuro, usar_ensemble=True, retornar_intervalo=True)
    
    print("\nPredições com Intervalos de Confiança 95%:")
    for i, (pred, lower, upper) in enumerate(zip(
        resultados['predicao'], 
        resultados['intervalo_inferior'], 
        resultados['intervalo_superior']
    )):
        print(f"   Dia {i+1}: R$ {pred:,.2f} [IC: R$ {lower:,.2f} - R$ {upper:,.2f}]")
    
    # Salvar modelo
    modelo.salvar_modelo('modelo_vendas_bootstrap_final.pkl')
    
    print("\n✅ Modelo treinado e salvo com sucesso!")
    print("📊 Bootstrap resampling aplicado com sucesso para lidar com poucos dados")
    print("📈 Métricas RMSE e R² calculadas com intervalos de confiança")
    print("🎯 Importância das features calculada e agregada por categoria")
    
    return modelo, relatorio

if __name__ == "__main__":
    modelo, relatorio = exemplo_completo()
