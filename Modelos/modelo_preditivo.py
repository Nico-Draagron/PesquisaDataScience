import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Bibliotecas para detecção de anomalias
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

# Bibliotecas para modelagem preditiva
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Bibliotecas para visualização
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Para dados climáticos (simulação)
import requests
import json

class BusinessAnalyticsPipeline:
    """
    Pipeline automatizado para análise de negócios com detecção de anomalias
    e predição baseada em dados operacionais e climáticos.
    """
    
    def __init__(self):
        self.operational_data = None
        self.weather_data = None
        self.unified_data = None
        self.anomaly_detectors = {}
        self.predictive_models = {}
        self.column_mapping = {}
        self.scaler = StandardScaler()
        
        # Padrões para reconhecimento automático de colunas
        self.column_patterns = {
            'valor_total': [
                r'valor.*total', r'total.*valor', r'receita', r'faturamento',
                r'vendas.*valor', r'valor.*vendas', r'revenue', r'sales.*amount'
            ],
            'valor_medio': [
                r'valor.*m[eé]dio', r'ticket.*m[eé]dio', r'avg.*valor',
                r'average.*value', r'mean.*value', r'valor.*unit[aá]rio'
            ],
            'num_pedidos': [
                r'n[uú]mero.*pedidos', r'qtd.*pedidos', r'quantidade.*pedidos',
                r'num.*orders', r'order.*count', r'pedidos', r'orders'
            ],
            'quebra_caixa': [
                r'quebra.*caixa', r'cash.*break', r'deficit.*caixa',
                r'falta.*caixa', r'shortage'
            ],
            'data': [
                r'data', r'date', r'dt', r'timestamp', r'time'
            ]
        }
    
    def detect_and_standardize_columns(self, df):
        """
        Detecta automaticamente colunas baseado em padrões e padroniza nomes.
        """
        print("🔍 Detectando e padronizando colunas...")
        
        detected_columns = {}
        
        for standard_name, patterns in self.column_patterns.items():
            for col in df.columns:
                col_lower = col.lower().strip()
                for pattern in patterns:
                    if re.search(pattern, col_lower, re.IGNORECASE):
                        detected_columns[col] = standard_name
                        break
                if col in detected_columns:
                    break
        
        # Renomear colunas detectadas
        df_standardized = df.rename(columns=detected_columns)
        self.column_mapping = detected_columns
        
        print(f"✅ Colunas detectadas e mapeadas: {detected_columns}")
        return df_standardized
    
    def load_operational_data(self, data_source):
        """
        Carrega dados operacionais de diferentes fontes.
        """
        print("📊 Carregando dados operacionais...")
        
        if isinstance(data_source, str):
            # Carregar de arquivo
            if data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
            elif data_source.endswith('.xlsx') or data_source.endswith('.xls'):
                df = pd.read_excel(data_source)
            else:
                raise ValueError("Formato de arquivo não suportado")
        elif isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
        else:
            raise ValueError("Fonte de dados inválida")
        
        # Detectar e padronizar colunas
        df = self.detect_and_standardize_columns(df)
        
        # Converter coluna de data
        date_cols = [col for col in df.columns if 'data' in col.lower() or 'date' in col.lower()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            df = df.rename(columns={date_cols[0]: 'data'})
        
        self.operational_data = df
        print(f"✅ Dados operacionais carregados: {df.shape[0]} registros, {df.shape[1]} colunas")
        return df
    
    def generate_weather_data(self, start_date, end_date):
        """
        Gera dados climáticos sintéticos para o período especificado.
        Em produção, isso se conectaria a uma API real como OpenWeatherMap.
        """
        print("🌤️ Gerando dados climáticos sintéticos...")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Simulação de dados climáticos com padrões sazonais
        np.random.seed(42)
        weather_data = []
        
        for date in date_range:
            # Padrões sazonais básicos
            day_of_year = date.dayofyear
            seasonal_temp = 25 + 10 * np.sin(2 * np.pi * day_of_year / 365)
            
            weather_data.append({
                'data': date,
                'temperatura': seasonal_temp + np.random.normal(0, 5),
                'umidade': max(30, min(90, 60 + np.random.normal(0, 15))),
                'chuva': max(0, np.random.exponential(2) if np.random.random() < 0.3 else 0),
                'vento': max(0, np.random.normal(15, 5)),
                'pressao': 1013 + np.random.normal(0, 20)
            })
        
        self.weather_data = pd.DataFrame(weather_data)
        print(f"✅ Dados climáticos gerados: {len(weather_data)} registros")
        return self.weather_data
    
    def unify_temporal_data(self):
        """
        Unifica dados operacionais e climáticos por data.
        """
        print("📅 Unificando dados temporais...")
        
        if self.operational_data is None:
            raise ValueError("Dados operacionais não carregados")
        
        # Agrupar dados operacionais por dia
        if 'data' in self.operational_data.columns:
            daily_ops = self.operational_data.groupby('data').agg({
                col: 'sum' if col in ['valor_total', 'num_pedidos'] else 'mean'
                for col in self.operational_data.columns if col != 'data'
            }).reset_index()
        else:
            raise ValueError("Coluna de data não encontrada nos dados operacionais")
        
        # Gerar dados climáticos se não existirem
        if self.weather_data is None:
            start_date = daily_ops['data'].min()
            end_date = daily_ops['data'].max()
            self.generate_weather_data(start_date, end_date)
        
        # Unificar dados
        unified = pd.merge(daily_ops, self.weather_data, on='data', how='left')
        
        # Preencher valores ausentes
        unified = unified.fillna(method='ffill').fillna(method='bfill')
        
        # Adicionar features temporais
        unified['dia_semana'] = unified['data'].dt.dayofweek
        unified['mes'] = unified['data'].dt.month
        unified['dia_mes'] = unified['data'].dt.day
        unified['trimestre'] = unified['data'].dt.quarter
        unified['fim_semana'] = (unified['dia_semana'] >= 5).astype(int)
        
        self.unified_data = unified
        print(f"✅ Dados unificados: {unified.shape[0]} registros, {unified.shape[1]} colunas")
        return unified
    
    def detect_anomalies(self, contamination=0.1):
        """
        Detecta anomalias usando múltiplos algoritmos.
        """
        print("🚨 Detectando anomalias...")
        
        if self.unified_data is None:
            raise ValueError("Execute unify_temporal_data() primeiro")
        
        # Selecionar features numéricas para detecção de anomalias
        numeric_cols = self.unified_data.select_dtypes(include=[np.number]).columns
        features = self.unified_data[numeric_cols].fillna(0)
        
        # Normalizar dados
        features_scaled = self.scaler.fit_transform(features)
        
        # Múltiplos detectores de anomalia
        detectors = {
            'IsolationForest': IsolationForest(contamination=contamination, random_state=42),
            'LocalOutlierFactor': LocalOutlierFactor(contamination=contamination),
            'KNN': KNN(contamination=contamination),
            'OCSVM': OCSVM(contamination=contamination)
        }
        
        anomaly_results = {}
        
        for name, detector in detectors.items():
            print(f"  Executando {name}...")
            
            if name == 'LocalOutlierFactor':
                anomalies = detector.fit_predict(features_scaled)
            else:
                detector.fit(features_scaled)
                anomalies = detector.predict(features_scaled)
            
            # Converter para formato binário (1 = normal, -1 = anomalia)
            anomaly_results[name] = (anomalies == -1).astype(int)
        
        # Criar score de consenso
        anomaly_df = pd.DataFrame(anomaly_results)
        self.unified_data['anomalia_score'] = anomaly_df.mean(axis=1)
        self.unified_data['is_anomalia'] = (self.unified_data['anomalia_score'] > 0.5).astype(int)
        
        anomaly_count = self.unified_data['is_anomalia'].sum()
        print(f"✅ Detecção concluída: {anomaly_count} anomalias detectadas ({anomaly_count/len(self.unified_data)*100:.1f}%)")
        
        return self.unified_data[self.unified_data['is_anomalia'] == 1]
    
    def train_predictive_models(self, target_columns=None):
        """
        Treina modelos preditivos para métricas de interesse.
        """
        print("🤖 Treinando modelos preditivos...")
        
        if self.unified_data is None:
            raise ValueError("Execute unify_temporal_data() primeiro")
        
        if target_columns is None:
            target_columns = ['valor_total', 'num_pedidos', 'valor_medio']
        
        # Preparar features
        feature_cols = [
            'temperatura', 'umidade', 'chuva', 'vento', 'pressao',
            'dia_semana', 'mes', 'dia_mes', 'trimestre', 'fim_semana'
        ]
        
        available_features = [col for col in feature_cols if col in self.unified_data.columns]
        X = self.unified_data[available_features].fillna(0)
        
        # Treinar modelo para cada target
        for target in target_columns:
            if target not in self.unified_data.columns:
                print(f"  ⚠️ Target '{target}' não encontrado, pulando...")
                continue
                
            print(f"  Treinando modelo para '{target}'...")
            
            y = self.unified_data[target].fillna(0)
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Testar múltiplos modelos
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(random_state=42),
                'LinearRegression': LinearRegression()
            }
            
            best_model = None
            best_score = -np.inf
            
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                
                if score > best_score:
                    best_score = score
                    best_model = model
            
            self.predictive_models[target] = {
                'model': best_model,
                'score': best_score,
                'features': available_features
            }
            
            print(f"    ✅ Melhor modelo para '{target}': R² = {best_score:.3f}")
        
        print("✅ Treinamento de modelos concluído!")
        return self.predictive_models
    
    def predict_future(self, days_ahead=7):
        """
        Faz predições para os próximos dias.
        """
        print(f"🔮 Fazendo predições para os próximos {days_ahead} dias...")
        
        if not self.predictive_models:
            raise ValueError("Execute train_predictive_models() primeiro")
        
        # Gerar datas futuras
        last_date = self.unified_data['data'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )
        
        # Gerar dados climáticos sintéticos para o futuro
        future_weather = self.generate_weather_data(
            future_dates[0], future_dates[-1]
        )
        
        # Preparar features futuras
        future_data = future_weather.copy()
        future_data['dia_semana'] = future_data['data'].dt.dayofweek
        future_data['mes'] = future_data['data'].dt.month
        future_data['dia_mes'] = future_data['data'].dt.day
        future_data['trimestre'] = future_data['data'].dt.quarter
        future_data['fim_semana'] = (future_data['dia_semana'] >= 5).astype(int)
        
        # Fazer predições
        predictions = {'data': future_dates}
        
        for target, model_info in self.predictive_models.items():
            model = model_info['model']
            features = model_info['features']
            
            X_future = future_data[features].fillna(0)
            pred = model.predict(X_future)
            predictions[f'{target}_pred'] = pred
        
        predictions_df = pd.DataFrame(predictions)
        print("✅ Predições realizadas!")
        return predictions_df
    
    def generate_report(self):
        """
        Gera relatório completo da análise.
        """
        print("📊 Gerando relatório de análise...")
        
        if self.unified_data is None:
            raise ValueError("Execute o pipeline completo primeiro")
        
        # Estatísticas básicas
        report = {
            'resumo_dados': {
                'total_registros': len(self.unified_data),
                'periodo_analise': f"{self.unified_data['data'].min()} a {self.unified_data['data'].max()}",
                'anomalias_detectadas': self.unified_data['is_anomalia'].sum(),
                'taxa_anomalias': f"{self.unified_data['is_anomalia'].mean()*100:.1f}%"
            },
            'metricas_negocio': {},
            'performance_modelos': {},
            'insights': []
        }
        
        # Métricas de negócio
        for col in ['valor_total', 'num_pedidos', 'valor_medio']:
            if col in self.unified_data.columns:
                report['metricas_negocio'][col] = {
                    'media': self.unified_data[col].mean(),
                    'mediana': self.unified_data[col].median(),
                    'desvio_padrao': self.unified_data[col].std(),
                    'min': self.unified_data[col].min(),
                    'max': self.unified_data[col].max()
                }
        
        # Performance dos modelos
        for target, model_info in self.predictive_models.items():
            report['performance_modelos'][target] = {
                'r2_score': model_info['score'],
                'features_utilizadas': len(model_info['features'])
            }
        
        # Insights automáticos
        if 'valor_total' in self.unified_data.columns:
            weekend_avg = self.unified_data[self.unified_data['fim_semana'] == 1]['valor_total'].mean()
            weekday_avg = self.unified_data[self.unified_data['fim_semana'] == 0]['valor_total'].mean()
            
            if weekend_avg > weekday_avg * 1.1:
                report['insights'].append("💡 Vendas são significativamente maiores nos fins de semana")
            elif weekday_avg > weekend_avg * 1.1:
                report['insights'].append("💡 Vendas são maiores durante a semana")
        
        # Correlação com clima
        if 'temperatura' in self.unified_data.columns and 'valor_total' in self.unified_data.columns:
            corr = self.unified_data['temperatura'].corr(self.unified_data['valor_total'])
            if abs(corr) > 0.3:
                direction = "positiva" if corr > 0 else "negativa"
                report['insights'].append(f"🌡️ Correlação {direction} entre temperatura e vendas (r={corr:.2f})")
        
        print("✅ Relatório gerado!")
        return report
    
    def visualize_results(self):
        """
        Cria visualizações dos resultados.
        """
        print("📈 Criando visualizações...")
        
        if self.unified_data is None:
            return None
        
        # Configurar subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Série Temporal com Anomalias',
                'Distribuição de Anomalias',
                'Correlação Clima vs Negócios',
                'Predições Futuras'
            ],
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "scatter"}, {"secondary_y": True}]]
        )
        
        # Série temporal com anomalias
        if 'valor_total' in self.unified_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.unified_data['data'],
                    y=self.unified_data['valor_total'],
                    mode='lines',
                    name='Valor Total',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Destacar anomalias
            anomalies = self.unified_data[self.unified_data['is_anomalia'] == 1]
            if len(anomalies) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=anomalies['data'],
                        y=anomalies['valor_total'],
                        mode='markers',
                        name='Anomalias',
                        marker=dict(color='red', size=8, symbol='x')
                    ),
                    row=1, col=1
                )
        
        # Distribuição de anomalias por dia da semana
        if 'dia_semana' in self.unified_data.columns:
            anomaly_by_day = self.unified_data.groupby('dia_semana')['is_anomalia'].sum()
            dias = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
            
            fig.add_trace(
                go.Bar(
                    x=dias,
                    y=anomaly_by_day.values,
                    name='Anomalias por Dia',
                    marker_color='orange'
                ),
                row=1, col=2
            )
        
        # Correlação clima vs negócios
        if all(col in self.unified_data.columns for col in ['temperatura', 'valor_total']):
            fig.add_trace(
                go.Scatter(
                    x=self.unified_data['temperatura'],
                    y=self.unified_data['valor_total'],
                    mode='markers',
                    name='Temp vs Vendas',
                    marker=dict(color='green', opacity=0.6)
                ),
                row=2, col=1
            )
        
        # Layout
        fig.update_layout(
            height=800,
            title_text="Dashboard de Análise de Negócios",
            showlegend=True
        )
        
        print("✅ Visualizações criadas!")
        return fig

# Exemplo de uso completo
def exemplo_completo():
    """
    Demonstra o uso completo do pipeline.
    """
    print("🚀 Iniciando exemplo completo do pipeline...")
    
    # Criar dados sintéticos para demonstração
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
    
    operational_data = pd.DataFrame({
        'Data': dates,
        'Valor Total': np.random.normal(50000, 15000, len(dates)),
        'Número de Pedidos': np.random.poisson(100, len(dates)),
        'Valor Médio': np.random.normal(500, 100, len(dates))
    })
    
    # Adicionar algumas anomalias intencionais
    anomaly_indices = np.random.choice(len(operational_data), 10, replace=False)
    operational_data.loc[anomaly_indices, 'Valor Total'] *= 2  # Picos de venda
    
    # Inicializar pipeline
    pipeline = BusinessAnalyticsPipeline()
    
    # Executar pipeline completo
    pipeline.load_operational_data(operational_data)
    pipeline.unify_temporal_data()
    anomalies = pipeline.detect_anomalies()
    models = pipeline.train_predictive_models()
    predictions = pipeline.predict_future(days_ahead=14)
    report = pipeline.generate_report()
    
    print("\n" + "="*50)
    print("📋 RELATÓRIO FINAL")
    print("="*50)
    
    print(f"\n📊 Resumo dos Dados:")
    for key, value in report['resumo_dados'].items():
        print(f"  {key}: {value}")
    
    print(f"\n🤖 Performance dos Modelos:")
    for target, metrics in report['performance_modelos'].items():
        print(f"  {target}: R² = {metrics['r2_score']:.3f}")
    
    print(f"\n💡 Insights:")
    for insight in report['insights']:
        print(f"  {insight}")
    
    print(f"\n🔮 Predições (próximos 7 dias):")
    if len(predictions) > 0:
        print(predictions.head(7).to_string(index=False))
    
    return pipeline, report, predictions

if __name__ == "__main__":
    pipeline, report, predictions = exemplo_completo()
