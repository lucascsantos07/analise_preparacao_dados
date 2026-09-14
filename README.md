# Análise Exploratória e Preparação de Dados para Machine Learning

**Aluno:** Lucas da Costa Santos  
**Matrícula:** 20250038952  
**Disciplina:** DCT1401 - Inteligência Artificial  
**Período:** 2026.2  

**Dataset:** https://www.kaggle.com/datasets/harpartapsingh13/movie-genre-recommender-dataset/data

## Descrição

Este repositório contém o desenvolvimento da atividade prática da
disciplina DCT1401 - Inteligência Artificial, cujo objetivo é realizar uma
análise exploratória, limpeza e pré-processamento de dados para apoiar a
construção de modelos de Machine Learning.

O dataset utilizado contém informações demográficas, características de
personalidade, preferências e hábitos relacionados ao consumo de filmes.
A análise foi realizada considerando dois problemas de aprendizado
supervisionado: regressão e classificação.

## Objetivos

- Realizar uma análise exploratória dos dados (EDA);
- Identificar e tratar valores ausentes;
- Detectar e analisar possíveis outliers;
- Avaliar a correlação entre variáveis numéricas;
- Realizar a codificação de variáveis categóricas;
- Identificar uma variável-alvo para regressão;
- Identificar uma variável-alvo para classificação;
- Analisar as principais características relacionadas a cada problema;
- Recomendar algoritmos para futuras etapas de modelagem.

## Estrutura do Projeto

```text
.
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   └── processed/
│       └── dataset_clean.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_regression_analysis.ipynb
│   └── 04_classification_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   └── visualization.py
│
├── reports/
│   └── figures/
│
├── requirements.txt
└── README.md
```

# Como Executar

## 1. Clonar o repositório

```bash
git clone https://github.com/lucascsantos07/analise_preparacao_dados.git
```

## 2. Criar e ativar o ambiente virtual

No Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

## 4. Executar os notebooks

Os notebooks devem ser executados na seguinte ordem:

1. `01_eda.ipynb` — análise exploratória dos dados
2. `02_preprocessing.ipynb` — limpeza e pré-processamento
3. `03_regression_analysis.ipynb` — análise para regressão
4. `04_classification_analysis.ipynb` — análise para classificação

---

# Principais Resultados

## Análise Exploratória

O dataset possui 6.000 registros e 35 variáveis. Foram identificadas variáveis numéricas e categóricas, além de valores ausentes nas variáveis `favorite_game_genre` e `favorite_music_genre`.

Os valores ausentes representaram 6,00% e 5,93% dos registros, respectivamente, e foram tratados pela criação da categoria `Nao_Informado`.

A análise de outliers identificou possíveis valores extremos em algumas variáveis numéricas. Após a análise estatística e visual, esses valores foram mantidos por não haver evidências de erros nos registros.

As correlações entre as variáveis numéricas apresentaram valores próximos de zero, não indicando relações lineares ou monotônicas fortes entre elas.

## Regressão

A variável-alvo escolhida foi: `social_media_hours_per_day`

Foram analisadas as relações entre o alvo e as variáveis numéricas por meio das correlações de Pearson e Spearman, SelectKBest e Random Forest.

Os resultados indicaram relações muito fracas entre as variáveis numéricas e o alvo.

Na modelagem inicial, a Regressão Linear apresentou:

* MAE: 1,359
* RMSE: 1,776
* R²: 0,0004

O Random Forest Regressor apresentou:

* MAE: 1,431
* RMSE: 1,836
* R²: -0,068

Os resultados indicaram baixo poder preditivo das variáveis analisadas para explicar a variação de `social_media_hours_per_day`.

## Classificação

A variável-alvo escolhida foi: `recommended_movie_genre`

A variável possui 9 classes de gêneros de filmes.

As análises realizadas indicaram maior relevância de características como:

* `openness_score`
* `risk_tolerance`
* `stress_level`
* `romance_interest`
* `humor_preference`
* `fear_tolerance`
* `age_group`
* `favorite_game_genre`

Como modelos iniciais para uma etapa posterior, são sugeridos a Regressão Logística e o Random Forest Classifier.

## Pré-processamento

Foram utilizadas diferentes estratégias de codificação de acordo com o tipo de variável:

* **Ordinal Encoding**: para variáveis categóricas com ordem natural
* **One-Hot Encoding**: para variáveis categóricas nominais
* **Multi-Hot Encoding**: para a variável `hobbies`, que pode possuir múltiplos valores
* **Label Encoding**: para a variável-alvo `recommended_movie_genre`

As variáveis originais foram mantidas no dataset, enquanto as representações codificadas foram adicionadas como novas colunas.

---

# Relatório

O relatório técnico completo da atividade está disponível em: `docs/relatorio_final.pdf`


# Autor

Lucas da Costa Santos
