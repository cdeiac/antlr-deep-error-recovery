import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import Levenshtein as lev


def load_combined_parse_statistics():
    df_combined = pd.concat([
        preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_001/parse_statistics0.json"), 0.001),
        preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_001/parse_statistics1.json"), 0.001),
        preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_001/parse_statistics2.json"), 0.001),
        preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_005/parse_statistics0.json"), 0.005),
        preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_005/parse_statistics1.json"), 0.005),
        preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_005/parse_statistics2.json"), 0.005),
        preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_010/parse_statistics0.json"), 0.010),
        preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_010/parse_statistics1.json"), 0.010),
        preprocess_df(pd.read_json("src/main/python/data/generated/cv/00_010/parse_statistics2.json"), 0.010),

    ], ignore_index=True)
    return df_combined

def load_combined_sequence_statistics():
    df_combined = pd.concat([
        pd.read_json("src/main/python/data/generated/cv/00_001/sequence_statistics0.json"),
        pd.read_json("src/main/python/data/generated/cv/00_001/sequence_statistics1.json"),
        pd.read_json("src/main/python/data/generated/cv/00_001/sequence_statistics2.json"),
        pd.read_json("src/main/python/data/generated/cv/00_005/sequence_statistics0.json"),
        pd.read_json("src/main/python/data/generated/cv/00_005/sequence_statistics1.json"),
        pd.read_json("src/main/python/data/generated/cv/00_005/sequence_statistics2.json"),
        pd.read_json("src/main/python/data/generated/cv/00_010/sequence_statistics0.json"),
        pd.read_json("src/main/python/data/generated/cv/00_010/sequence_statistics1.json"),
        pd.read_json("src/main/python/data/generated/cv/00_010/sequence_statistics2.json"),
    ], ignore_index=True)
    return df_combined


def preprocess_df(dataframe, noise_level):
    #dataframe['originalData'] = dataframe['originalData'].astype('string')
    #dataframe['noisyData'] = dataframe['noisyData'].astype('string')
    #dataframe['numOfcompilationErrors'] = dataframe['compilationErrors'].apply(lambda x: len(x))
    #dataframe['hasNoise'] = dataframe['noiseOperations'].apply(lambda arr: not all(x == 0 for x in arr))
    dataframe['noiseLevel'] = noise_level
    # base
    dataframe['baseNumOfCompilationErrors'] = dataframe['baseCompilationErrors'].apply(lambda x: len(x))
    #dataframe['baseNumOfMissingErrors'] = dataframe['baseCompilationErrors'].apply(lambda x: count_errors(x, 'MissingError'))
    #dataframe['baseNumOfMismatchedErrors'] = dataframe['baseCompilationErrors'].apply(lambda x: count_errors(x, 'MismatchedError'))
    #dataframe['baseNumOfExtraneousInputErrors'] = dataframe['baseCompilationErrors'].apply(lambda x: count_errors(x, 'ExtraneousInputError'))
    dataframe['baseCompiles'] = dataframe['baseNumOfCompilationErrors'] == 0
    # antlr ER
    dataframe['antlrNumOfCompilationErrors'] = dataframe['antlrCompilationErrors'].apply(lambda x: len(x))
    #dataframe['antlrNumOfMissingErrors'] = dataframe['antlrCompilationErrors'].apply(lambda x: count_errors(x, 'MissingError'))
    #dataframe['antlrNumOfMismatchedErrors'] = dataframe['antlrCompilationErrors'].apply(lambda x: count_errors(x, 'MismatchedError'))
    #dataframe['antlrNumOfExtraneousInputErrors'] = dataframe['antlrCompilationErrors'].apply(lambda x: count_errors(x, 'ExtraneousInputError'))
    # deep ER on-error
    dataframe['derOnErrorNumOfCompilationErrors'] = dataframe['derOnErrorCompilationErrors'].apply(lambda x: len(x))
    #dataframe['derOnErrorNumOfMissingErrors'] = dataframe['derOnErrorCompilationErrors'].apply(lambda x: count_errors(x, 'MissingError'))
    #dataframe['derOnErrorNumOfMismatchedErrors'] = dataframe['derOnErrorCompilationErrors'].apply(lambda x: count_errors(x, 'MismatchedError'))
    #dataframe['derOnErrorNumOfExtraneousInputErrors'] = dataframe['derOnErrorCompilationErrors'].apply(lambda x: count_errors(x, 'ExtraneousInputError'))
    # deep ER on-file
    dataframe['derOnFileNumOfCompilationErrors'] = dataframe['derOnFileCompilationErrors'].apply(lambda x: len(x))
    #dataframe['derOnFileNumOfMissingErrors'] = dataframe['derOnFileCompilationErrors'].apply(lambda x: count_errors(x, 'MissingError'))
    #dataframe['derOnFileNumOfMismatchedErrors'] = dataframe['derOnFileCompilationErrors'].apply(lambda x: count_errors(x, 'MismatchedError'))
    #dataframe['derOnFileNumOfExtraneousInputErrors'] = dataframe['derOnFileCompilationErrors'].apply(lambda x: count_errors(x, 'ExtraneousInputError'))
    return dataframe.drop(columns=[col for col in dataframe.columns if 'bail' in col.lower()])


def plot_on_file_difference(df):
    df['normalized_diff'] = df.apply(
        lambda row: abs(len(row['original']) - len(row['derOnFilePrediction'])) / len(row['original']) if len(
            row['original']) > 0 else 0, axis=1)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='noiseLevel', y='normalized_diff', data=df)
    plt.title('Normalized Difference in Length by Noise Level')
    plt.ylabel('Normalized Length Difference')
    plt.xlabel('Noise Level')
    plt.show()


if __name__ == '__main__':
    df = load_combined_parse_statistics()
    df_seq = load_combined_sequence_statistics()
    df_joined = df.join(df_seq)
    df_joined = df_joined[df_joined['baseCompiles'] != True]

    df_joined['normalized_diff'] = df_joined.apply(
        lambda row: abs(len(row['original']) - len(row['derOnFilePrediction'])) / len(row['original']) if len(
            row['original']) > 0 else 0, axis=1)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='noiseLevel', y='normalized_diff', data=df_joined)
    plt.title('Normalized Difference in Length by Noise Level')
    plt.ylabel('Normalized Length Difference')
    plt.xlabel('Noise Level')
    plt.show()