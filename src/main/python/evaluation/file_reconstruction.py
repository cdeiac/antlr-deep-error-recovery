import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def load_combined_parse_statistics():
    df_combined = pd.concat([
        preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_001/parse_statistics0.json"), 0.001),
        preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_001/parse_statistics1.json"), 0.001),
        preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_001/parse_statistics2.json"), 0.001),
        preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_005/parse_statistics0.json"), 0.005),
        #preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_005/parse_statistics1.json"), 0.005),
        #preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_005/parse_statistics2.json"), 0.005),
        #preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_010/parse_statistics0.json"), 0.010),
        #preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_010/parse_statistics1.json"), 0.010),
        #preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_010/parse_statistics2.json"), 0.010),

    ], ignore_index=True)
    return df_combined

def load_combined_sequence_statistics():
    df_combined = pd.concat([
        pd.read_json("src/main/python/data/generated/cv/00_001/sequence_statistics0.json"),
        pd.read_json("src/main/python/data/generated/cv/00_001/sequence_statistics1.json"),
        pd.read_json("src/main/python/data/generated/cv/00_001/sequence_statistics2.json"),
        pd.read_json("src/main/python/data/generated/cv/00_005/sequence_statistics0.json"),
        #pd.read_json("src/main/python/data/generated/cv/00_005/sequence_statistics1.json"),
        #pd.read_json("src/main/python/data/generated/cv/00_005/sequence_statistics2.json"),
        #pd.read_json("src/main/python/data/generated/cv/00_010/sequence_statistics0.json"),
        #pd.read_json("src/main/python/data/generated/cv/00_010/sequence_statistics1.json"),
        #pd.read_json("src/main/python/data/generated/cv/00_010/sequence_statistics2.json"),
    ], ignore_index=True)
    return df_combined


def preprocess_df(dataframe, noise_level):
    dataframe['noiseLevel'] = noise_level
    # base
    dataframe['baseNumOfCompilationErrors'] = dataframe['baseCompilationErrors'].apply(lambda x: len(x))
    dataframe['baseCompiles'] = dataframe['baseNumOfCompilationErrors'] == 0
    # antlr ER
    dataframe['antlrNumOfCompilationErrors'] = dataframe['antlrCompilationErrors'].apply(lambda x: len(x))
    # deep ER on-error
    dataframe['derOnErrorNumOfCompilationErrors'] = dataframe['derOnErrorCompilationErrors'].apply(lambda x: len(x))
    # deep ER on-file
    dataframe['derOnFileNumOfCompilationErrors'] = dataframe['derOnFileCompilationErrors'].apply(lambda x: len(x))
    return dataframe.drop(columns=[col for col in dataframe.columns if 'bail' in col.lower()])


def count_errors(arr, error_type):
    return sum(1 for elem in arr if elem.get('type') == error_type)


def extract_first_type(arr):
    if len(arr) > 0:
        return arr[0].get('type', None)
    else:
        return None


def clean_up_control_tokens(df):
    df['original'] = df['original'].apply(lambda x: x[1:-1] if isinstance(x, list) and len(x) > 2 else x)
    df['noisy'] = df['noisy'].apply(lambda x: x[1:-1] if isinstance(x, list) and len(x) > 2 else x)
    return df


def violinplot_levenshtein(df, filename):
    melted_df = df.melt(id_vars=['noiseLevel'], value_vars=['levenshtein_antlr', 'levenshtein_derOnError'],
                        var_name='Type', value_name='Distance')

    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.violinplot(x='noiseLevel', y='Distance', hue='Type', data=melted_df, split=True, gap=.1, fill=False, cut=0, inner="quart")

    for line in ax.lines:
        line.set_linestyle('-')
        line.set_color(line.get_color())


    legend = plt.legend(title='Error Recovery', loc='upper center', fancybox=False, shadow=False, ncol=2)
    legend.get_texts()[0].set_text('ANTLR')
    legend.get_texts()[1].set_text('Deep')
    legend.set_bbox_to_anchor((0.1, -0.1))
    plt.ylabel("Norm. Levenshtein Distance")
    plt.xlabel("Noise Level (%)", labelpad=15)
    plt.grid(True, axis='y', linestyle='--')
    plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


if __name__ == '__main__':
    sns.set_style('whitegrid', rc={
        'xtick.bottom': True,
        'ytick.left': True,
    })
    df = load_combined_parse_statistics()
    df_seq = load_combined_sequence_statistics()
    df_joined = df.join(df_seq)
    df_joined = df_joined[df_joined['baseCompiles'] != True]
    #df = df[df['baseCompiles'] != True]

    df_joined = clean_up_control_tokens(df_joined)
    violinplot_levenshtein(df_joined, "figure/levenshtein_total.pdf")