import pandas as pd
import numpy as np
from scipy import stats


def get_categorical_frequencies(df, top_n=None):
    """Retorna a frequência das categorias das variáveis categóricas."""

    categorical_columns = df.select_dtypes(include="object").columns

    frequencies = []

    for column in categorical_columns:
        counts = df[column].value_counts()

        if top_n:
            counts = counts.head(top_n)

        for category, count in counts.items():
            frequencies.append({
                "variable": column,
                "category": category,
                "frequency": count
            })

    return pd.DataFrame(frequencies)


def get_missing_values(df):
    """Retorna a quantidade e porcentagem de valores ausentes."""

    missing = df.isnull().sum()

    missing_pct = (missing / len(df) * 100).round(2)

    missing_table = pd.DataFrame({
        "qtd_ausentes": missing,
        "pct_ausentes": missing_pct
    })

    missing_table = missing_table[
        missing_table["qtd_ausentes"] > 0
    ].sort_values(
        "qtd_ausentes",
        ascending=False
    )

    return missing_table


def get_correlation_matrix(df, variables):
    """Retorna a matriz de correlação de Pearson das variáveis selecionadas."""

    return df[variables].corr(method="pearson")

def detectar_outliers_iqr(df, coluna):
    """Detecta outliers de uma variável numérica usando o método IQR."""

    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)

    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = df[
        (df[coluna] < limite_inferior) |
        (df[coluna] > limite_superior)
    ]

    return {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "limite_inferior": limite_inferior,
        "limite_superior": limite_superior,
        "quantidade_outliers": len(outliers),
        "outliers": outliers
    }

def detectar_outliers_zscore(df, coluna, limite=3):
    """Detecta outliers usando o método Z-score."""

    zscores = np.abs(stats.zscore(df[coluna]))

    outliers = df[zscores > limite].copy()

    outliers["zscore"] = zscores[zscores > limite]

    return {
        "limite": limite,
        "quantidade_outliers": len(outliers),
        "outliers": outliers
    }

def verificar_valores_invalidos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica se os valores das colunas numéricas estão dentro dos limites esperados.
    
    - Variáveis de escala: devem estar entre 1 e 10.
    - Variáveis de contagem/horas: não podem ser negativas (< 0).
    
    Returns:
        pd.DataFrame com colunas: variavel, minimo, maximo, valores_invalidos
    """
    resultados = []

    variaveis_escala = [
        "mood_score",
        "stress_level",
        "fear_tolerance",
        "risk_tolerance",
        "humor_preference",
        "openness_score",
        "romance_interest"
    ]

    for coluna in variaveis_escala:
        invalidos = ((df[coluna] < 1) | (df[coluna] > 10)).sum()
        resultados.append({
            "variavel": coluna,
            "minimo": df[coluna].min(),
            "maximo": df[coluna].max(),
            "valores_invalidos": invalidos
        })

    variaveis_nao_negativas = [
        "social_media_hours_per_day",
        "streaming_subscription_count"
    ]

    for coluna in variaveis_nao_negativas:
        resultados.append({
            "variavel": coluna,
            "minimo": df[coluna].min(),
            "maximo": df[coluna].max(),
            "valores_invalidos": (df[coluna] < 0).sum()
        })

    return pd.DataFrame(resultados)