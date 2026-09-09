import os
import matplotlib.pyplot as plt
import seaborn as sns


def save_plot(filename, output_dir="../reports/figures"):
    """Salva o gráfico atual no diretório de saída."""

    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(
        os.path.join(output_dir, filename),
        bbox_inches="tight",
        dpi=300
    )


def plot_missing_values_heatmap(df):
    """Gera um heatmap dos valores ausentes."""

    cols_com_nulos = df.columns[df.isnull().any()]

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        df[cols_com_nulos].isnull(),
        cbar=True,
        cmap="viridis",
        yticklabels=False,
        cbar_kws={"label": "0 = Preenchido | 1 = Ausente"},
    )

    plt.title("Valores Ausentes por Coluna", fontsize=12, pad=12)
    plt.xlabel("Atributos com Nulos")
    plt.ylabel("Instâncias (Linhas)")
    plt.xticks(rotation=0)

    save_plot("missing_values_heatmap.png")

    plt.show()
    plt.close()

def plot_numerical_boxplots(df):
    """Gera boxplots para todas as variáveis numéricas."""

    numerical_columns = df.select_dtypes(include="number").columns
    numerical_columns = numerical_columns.drop("person_id")

    for column in numerical_columns:
        plt.figure(figsize=(8, 5))

        sns.boxplot(x=df[column])

        plt.title(f"Boxplot — {column}")
        plt.xlabel(column)

        plt.show()
        plt.close()

def plot_correlation_matrix(matriz_corr):
    """Gera um heatmap da matriz de correlação."""

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        matriz_corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        center=0,
        linewidths=0.5
    )

    plt.title("Matriz de Correlação de Pearson")

    plt.tight_layout()

    plt.show()
    plt.close()

def plot_pairplot(df, variables):
    """Gera uma matriz de dispersão entre variáveis numéricas."""

    pair_fig = sns.pairplot(
        df[variables],
        diag_kind="kde",
        plot_kws={"alpha": 0.4, "s": 25}
    )

    pair_fig.fig.suptitle(
        "Matriz de Dispersão entre Variáveis Numéricas",
        y=1.02,
        fontsize=12
    )

    pair_fig.savefig(
        "../reports/figures/scatter_matrix_numericas.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.show()
    plt.close()

def plot_stress_vs_mood(df):
    """Gera um gráfico de dispersão entre estresse e humor."""

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="stress_level",
        y="mood_score",
        alpha=0.3
    )

    sns.regplot(
        data=df,
        x="stress_level",
        y="mood_score",
        scatter=False,
        line_kws={"linewidth": 2}
    )

    plt.title("Relação entre Nível de Estresse e Pontuação de Humor")
    plt.xlabel("Nível de Estresse (stress_level)")
    plt.ylabel("Pontuação de Humor (mood_score)")

    save_plot("scatter_stress_vs_mood.png")

    plt.show()
    plt.close()

def plot_mood_by_movie_era(df):
    """Gera um boxplot do humor por era de filme favorita."""

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="favorite_movie_era",
        y="mood_score",
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "red",
            "markeredgecolor": "red"
        }
    )

    plt.title("Distribuição do Mood Score por Era de Filme Favorita")
    plt.xlabel("Era de Filme Favorita (favorite_movie_era)")
    plt.ylabel("Pontuação de Humor (mood_score)")
    plt.xticks(rotation=15)

    save_plot("boxplot_mood_vs_movie_era.png")

    plt.show()
    plt.close()

def plot_stress_by_recommended_genre(df):
    """Gera um boxplot do estresse por gênero de filme recomendado."""

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=df,
        x="recommended_movie_genre",
        y="stress_level",
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "darkred",
            "markeredgecolor": "darkred"
        }
    )

    plt.title("Distribuição de Nível de Estresse por Gênero Recomendado")
    plt.xlabel("Gênero Recomendado (recommended_movie_genre)")
    plt.ylabel("Nível de Estresse (stress_level)")
    plt.xticks(rotation=30)

    save_plot("boxplot_stress_vs_recommended_genre.png")

    plt.show()
    plt.close()

def plot_recommended_genre_frequency(df):
    """Gera um gráfico de frequência dos gêneros de filmes recomendados."""

    plt.figure(figsize=(10, 5))

    order = df["recommended_movie_genre"].value_counts().index

    sns.countplot(
        data=df,
        x="recommended_movie_genre",
        order=order
    )

    plt.title("Distribuição das Frequências por Gênero de Filme Recomendado")
    plt.xlabel("Gênero Recomendado (recommended_movie_genre)")
    plt.ylabel("Quantidade de Clientes")
    plt.xticks(rotation=40)

    save_plot("count_recommended_genre.png")

    plt.show()
    plt.close()

def plot_stress_distribution(df):
    """Gera um histograma da distribuição do nível de estresse."""

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="stress_level",
        kde=True,
        bins=20
    )

    plt.title("Distribuição da Variável 'Nível de Estresse'")
    plt.xlabel("Nível de Estresse (stress_level)")
    plt.ylabel("Frequência Absoluta")

    save_plot("hist_stress_level.png")

    plt.show()
    plt.close()