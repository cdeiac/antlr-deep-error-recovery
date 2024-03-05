import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ast import literal_eval
import json

from evaluation.reconstruction import load_combined_parse_statistics, load_combined_sequence_statistics


def normalize_positions(row, error_col):
    lst =  [1 if error['position'] < 0 else error['position'] / (len(row['noisy'])-2) for error in row[error_col]]
    return np.mean(lst)

def normalize_antlr_positions(row, error_col):
    lst = [1 if error['position'] < 0 else error['position']  / (len(row['antlrErPrediction'])) for error in row[error_col]]
    return np.mean(lst)
def normalize_der_positions(row, error_col):
    lst =  [1 if error['position'] < 0 else error['position']  / (len(row['derOnErrorPrediction'])) for error in row[error_col]]
    return np.mean(lst)
# Normalizing positions and preparing for plotting
def normalize_and_prepare(df, error_col):
    normalized_positions = []

    for index, row in df.iterrows():
        length_noisy = len(row['noisy'])-2
        for error in row[error_col]:
            normalized_position = error['position'] / length_noisy
            normalized_positions.append((row['noiseLevel'], normalized_position))

    return pd.DataFrame(normalized_positions, columns=['noiseLevel', 'NormalizedPosition'])

def normalize_and_prepare_antlr(df, error_col):
    normalized_positions = []

    for index, row in df.iterrows():
        length_noisy = len(row['antlrErPrediction'])
        for error in row[error_col]:
            normalized_position = error['position'] / length_noisy
            normalized_positions.append((row['noiseLevel'], normalized_position))

    return pd.DataFrame(normalized_positions, columns=['noiseLevel', 'NormalizedPosition'])


def normalize_and_prepare(df, error_col):
    normalized_positions = []

    for index, row in df.iterrows():
        length_noisy = len(row['derOnErrorPrediction'])
        for error in row[error_col]:
            normalized_position = error['position'] / length_noisy
            normalized_positions.append((row['noiseLevel'], normalized_position))

    return pd.DataFrame(normalized_positions, columns=['noiseLevel', 'NormalizedPosition'])


if __name__ == '__main__':
    sns.set_style('whitegrid', rc={
        'xtick.bottom': True,
        'ytick.left': True,
    })
    # plot_token_distribution(noisy_010)
    df = load_combined_parse_statistics()
    df_seq = load_combined_sequence_statistics()
    df_joined = df.join(df_seq)
    df_joined = df_joined[df_joined['baseCompiles'] != True]
    df_joined['normalized_base'] = df_joined.apply(lambda row: normalize_positions(row, "baseCompilationErrors"), axis=1)
    df_joined['normalized_antlr'] = df_joined.apply(lambda row: normalize_antlr_positions(row, 'antlrCompilationErrors'), axis=1)

    # Prepare data for plotting
    plot_data = pd.melt(df_joined, id_vars=['noiseLevel'], value_vars=['normalized_base', 'normalized_antlr'],
                        var_name='ErrorType', value_name='NormalizedPosition')

    # Convert list of values into separate rows
    plot_data = plot_data.explode('NormalizedPosition')

    # Adjust the 'ErrorType' for readability
    plot_data['ErrorType'] = plot_data['ErrorType'].map({'normalized_base': 'Base', 'normalized_antlr': 'ANTLR'})

    # Plotting the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='noiseLevel', y='NormalizedPosition', hue='ErrorType', data=plot_data, gap=.1, split=True, inner="quart", palette='Set2')
    plt.title('Normalized Error Positions by Noise Level and Error Type')
    plt.xlabel('Noise Level')
    plt.ylabel('Normalized Position')
    plt.legend(title='Error Type')
    plt.show()
