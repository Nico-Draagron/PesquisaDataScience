# APP streamlit

## Configuração

1. Instale as bibliotecas necessarias:

```bash
python install streamlit pandas seaborn matplotlib statsmodels
```

2. Como rodar


No terminal, rode:

```bash
python -m streamlit run app.py
```

## Estrutura do projeto

- `app.py`: "Tela inicia"
- `Clima x Vendas.py`: "Tela com regressão linear entre clima x vendas.
- `Serie_Temporal.py`: "Tela com  serie temporal"
- `dados_unificados_completos.csv`: arquivo CSV com os dados usados na análise, contem clima e vendas unificados(deve estar no caminho configurado).
- `resumo_diario_climatico`: arquivo CSV com os dados climaticos diarios(deve estar no caminho configurado).

---

Feito com Python e Streamlit.
