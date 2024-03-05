import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import Levenshtein as lev
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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


def count_errors(arr, error_type):
    return sum(1 for elem in arr if elem.get('type') == error_type)


def extract_first_type(arr):
    if len(arr) > 0:
        return arr[0].get('type', None)
    else:
        return None


def calculate_levenshtein(row, col1, col2):
    return lev.distance(row[col1], row[col2])


def normalized_levenshtein(row, col1, col2):
    distance = lev.distance(row[col1], row[col2])
    # Normalize by the maximum length of the two strings
    max_length = max(len(row[col1]), len(row[col2]))
    return distance / max_length if max_length > 0 else 0


def compute_normalize_levenshtein(original, prediciton, positions):
    normalized_distances = []
    for pos in positions:
        if pos < len(original) and pos < len(prediciton):
            distance = lev.distance(original[pos], prediciton[pos])
            max_length = max(len(original[pos]), len(prediciton[pos]))
            normalized_distance = distance / max_length if max_length > 0 else 0
            normalized_distances.append(normalized_distance)
        else:
            pass # position out of range
    return normalized_distances


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


def violinplot_reconstruction(df, y, filename, base=False):
    plt.figure(figsize=(10, 6), dpi=300)
    if base:
        ax = sns.violinplot(data=df, x="noiseLevel", y=y, fill=False, cut=0, inner="quart")
    else:
        ax = sns.violinplot(data=df, x="noiseLevel", y=y, hue="baseCompiles",
                            split=True, gap=.1, fill=False, cut=0, inner="quart", palette='Set1')

    for line in ax.lines:
        line.set_linestyle('-')
        line.set_color(line.get_color())

    if not base:
        legend = plt.legend(title='compiles initially', loc='upper center', fancybox=False, shadow=False, ncol=2)
        #legend.get_texts()[0].set_text('no')
        #legend.get_texts()[1].set_text('yes')
        legend.set_bbox_to_anchor((0.1, -0.1))
    plt.ylabel("Reconstruction")
    plt.xlabel("Noise Level (%)", labelpad=15)
    plt.gca().set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.grid(True, axis='y', linestyle='--')
    #plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def violinplot_accuracy(df, y, filename, base=False):
    plt.figure(figsize=(10, 6), dpi=300)
    if base:
        ax = sns.violinplot(data=df, x="noiseLevel", y=y, fill=False, cut=0, inner="quart")
    else:
        ax = sns.violinplot(data=df, x="noiseLevel", y=y, hue="baseCompiles",
                            split=True, gap=.1, fill=False, cut=0, inner="quart", palette='Set2')

    for line in ax.lines:
        line.set_linestyle('-')
        line.set_color(line.get_color())

    if not base:
        legend = plt.legend(title='compiles initially', loc='upper center', fancybox=False, shadow=False, ncol=2)
        #legend.get_texts()[0].set_text('no')
        #legend.get_texts()[1].set_text('yes')
        legend.set_bbox_to_anchor((0.1, -0.1))
    plt.ylabel("Accuracy")
    plt.xlabel("Noise Level (%)", labelpad=15)
    plt.gca().set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.grid(True, axis='y', linestyle='--')
    #plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def violinplot_accuracy_split(df, filename):
    plt.figure(figsize=(10, 6), dpi=300)

    # Reshape the DataFrame, keeping the 'category' column as a part of the reshaped DataFrame
    df_long = pd.melt(df, id_vars=['noiseLevel'], value_vars=['accuracy_antlr', 'accuracy_derOnError'], var_name='Variable',
                      value_name='Value')

    ax = sns.violinplot(data=df_long, x="noiseLevel", y="Value", hue="Variable", fill=False, cut=0,
                        gap=.1, inner="quart", palette='Set2')

    for line in ax.lines:
        line.set_linestyle('-')
        line.set_color(line.get_color())

    plt.ylabel("Accuracy")
    plt.xlabel("Noise Level (%)", labelpad=15)
    plt.gca().set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.grid(True, axis='y', linestyle='--')
    #plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def violinplot_reconstruction_split(df, filename):
    melted_df = df.melt(id_vars=['noiseLevel'], value_vars=['antlrReconstruction', 'derOnErrorReconstruction'],
                        var_name='Type', value_name='Distance')

    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.violinplot(x='noiseLevel', y='Distance', hue='Type', data=melted_df, split=True, gap=.1, fill=False, cut=0, palette="Set2", inner="quart")

    for line in ax.lines:
        line.set_linestyle('-')
        line.set_color(line.get_color())


    legend = plt.legend(title='Error Recovery', loc='upper center', fancybox=False, shadow=False, ncol=2)
    legend.get_texts()[0].set_text('ANTLR')
    legend.get_texts()[1].set_text('Deep')
    legend.set_bbox_to_anchor((0.1, -0.1))
    plt.ylabel("Reconstruction")
    plt.xlabel("Noise Level (%)", labelpad=15)
    plt.grid(True, axis='y', linestyle='--')
    plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def violinplot_on_file_reconstruction_split(df, filename):
    melted_df = df.melt(id_vars=['noiseLevel'], value_vars=['antlrReconstruction', 'derOnFileReconstruction'],
                        var_name='Type', value_name='Distance')

    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.violinplot(x='noiseLevel', y='Distance', hue='Type', data=melted_df, split=True, gap=.1, fill=False, cut=0, palette="Set2", inner="quart")

    for line in ax.lines:
        line.set_linestyle('-')
        line.set_color(line.get_color())


    legend = plt.legend(title='Error Recovery', loc='upper center', fancybox=False, shadow=False, ncol=2)
    legend.get_texts()[0].set_text('ANTLR')
    legend.get_texts()[1].set_text('Deep')
    legend.set_bbox_to_anchor((0.1, -0.1))
    plt.ylabel("Reconstruction")
    plt.xlabel("Noise Level (%)", labelpad=15)
    plt.grid(True, axis='y', linestyle='--')
    plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def violinplot_levenshtein(df, filename):
    melted_df = df.melt(id_vars=['noiseLevel'], value_vars=['levenshtein_antlr', 'levenshtein_derOnError'],
                        var_name='Type', value_name='Distance')

    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.violinplot(x='noiseLevel', y='Distance', hue='Type', data=melted_df, split=True, gap=.1, fill=False, palette="Set2", cut=0, inner="quart")

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


def violinplot_on_file_levenshtein(df, filename):
    melted_df = df.melt(id_vars=['noiseLevel'], value_vars=['levenshtein_antlr', 'levenshtein_derOnFile'],
                        var_name='Type', value_name='Distance')

    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.violinplot(x='noiseLevel', y='Distance', hue='Type', data=melted_df, split=True, gap=.1, fill=False, palette="Set2", cut=0, inner="quart")

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


def describe_to_latex(df, column):
    stats = df[column].describe()
    latex_stats = stats.to_latex()
    print(latex_stats)


def describe_selected_columns_modified(df, columns):
    valid_columns = [col for col in columns if col in df.columns]
    description_df = df[valid_columns].describe()
    description_df = description_df.drop(['count', '25%', '75%'])
    description_df = description_df.rename(index={'50%': 'median'})
    latex_code = description_df.to_latex()
    print(description_df)
    print(latex_code)
    return description_df


def extract_words(row):
    positions = [error['position'] for error in row['baseCompilationErrors']]
    extracted_original = [row['original'][pos] for pos in positions if pos < min(len(row['original']), len(row['noisy']), len(row['antlrErPrediction']), len(row['derOnErrorPrediction']))]
    extracted_noisy = [row['noisy'][pos] for pos in positions if
                          pos < min(len(row['original']), len(row['noisy']), len(row['antlrErPrediction']),
                                    len(row['derOnErrorPrediction']))]

    extracted_antlr = [row['antlrErPrediction'][pos] for pos in positions if pos < min(len(row['original']), len(row['noisy']), len(row['antlrErPrediction']), len(row['derOnErrorPrediction']))]
    extracted_derOnError = [row['derOnErrorPrediction'][pos] for pos in positions if pos < min(len(row['original']), len(row['noisy']), len(row['antlrErPrediction']), len(row['derOnErrorPrediction']))]
    return pd.Series([extracted_original, extracted_noisy, extracted_antlr, extracted_derOnError])


def plot_error_occurrence(df):
    # Counting the occurrences of each unique value using Counter
    value_counts = Counter(df['baseNumOfCompilationErrors'])
    # Sorting the counts for better visualization (optional)
    sorted_counts = dict(sorted(value_counts.items()))
    print(sorted_counts)
    # Plotting the bar chart
    plt.figure(figsize=(10, 6))  # Adjust the figure size
    # df.baseNumOfCompilationErrors.hist(bins=len(sorted_counts), color='blue', alpha=0.7)
    plt.bar(sorted_counts.keys(), sorted_counts.values(), color='blue', alpha=0.7)  # Create a bar chart
    plt.title('Noise Level = 0.001')  # Adding a title
    plt.xlabel('Value')  # Labeling the x-axis
    plt.ylabel('Frequency')  # Labeling the y-axis
    plt.xscale("log")
    plt.grid(axis='y')
    plt.xlim(0, 100)
    plt.show()


def compute_accuracy(cm):
    # Sum of diagonal elements (true positives and true negatives)
    correct_predictions = np.trace(cm)
    # Total number of predictions
    total_predictions = np.sum(cm)
    # Compute accuracy
    accuracy = correct_predictions / total_predictions if total_predictions else 0
    return accuracy


def calculate_row_accuracy(row):
    # Extract words lists
    original_words = row['extracted_original']
    noisy_words = row['extracted_noisy']
    antlr_words = row['extracted_antlr']
    derOnError_words = row['extracted_derOnError']

    # Calculate matches and accuracy for antlr
    base_matches = sum(o == a for o, a in zip(original_words, noisy_words))
    base_accuracy = base_matches / len(original_words) if original_words else 0

    antlr_matches = sum(o == a for o, a in zip(original_words, antlr_words))
    antlr_accuracy = antlr_matches / len(original_words) if original_words else 0

    # Calculate matches and accuracy for derOnError
    derOnError_matches = sum(o == d for o, d in zip(original_words, derOnError_words))
    derOnError_accuracy = derOnError_matches / len(original_words) if original_words else 0

    return pd.Series([base_accuracy, antlr_accuracy, derOnError_accuracy])

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

def normalize_matching_tokens(row, column_name):
    operations = row[column_name]
    total_operations = len(operations)
    if total_operations == 0:
        return None  # Avoid division by zero
    # Count matches where tokenAfter is not None and equals tokenBefore
    matching_count = sum(1 for op in operations if op['tokenAfter'] is not None and op['tokenAfter'] == op['tokenBefore'])
    # Normalize the count by the total number of operations
    normalized_count = matching_count / total_operations
    return normalized_count


def count_matches_and_entries(df, column_name):
    total_matches = 0
    total_entries = 0

    for operations in df[column_name]:
        total_entries += len(operations)  # Count all operations
        total_matches += sum(
            1 for op in operations if op['tokenAfter'] is not None and op['tokenAfter'] == op['tokenOriginal'])

    return total_matches, total_entries


def count_matches_for_lowest_position(df, column_name):
    total_matches = 0
    total_entries = 0

    for operations in df[column_name]:
        if not operations:  # Skip empty lists
            continue

        # Find the operation with the lowest position
        lowest_position_op = min(operations, key=lambda op: op['position'])

        total_entries += 1  # Only count the lowest position operation
        if lowest_position_op['tokenAfter'] is not None and lowest_position_op['tokenAfter'] == lowest_position_op[
            'tokenOriginal']:
            total_matches += 1

    return total_matches, total_entries


def extract_tokens(operations_list, token_after_list, token_original_list):
    for operation in operations_list:
        if 'tokenAfter' in operation and 'tokenBefore' in operation:
            token_after_list.append(operation['tokenAfter'])
            token_original_list.append(operation['tokenBefore'])


def plot_operations(tokenOriginal, tokenAfter, title):
    labels = np.unique(tokenOriginal + tokenAfter)  # Define the label set
    cm = confusion_matrix(tokenOriginal, tokenAfter, labels=labels, normalize='true')
    report = classification_report(tokenOriginal, tokenAfter, labels=labels, target_names=labels, zero_division=0.0)
    # plot
    plt.figure(figsize=(30, 25))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, linewidths=.5,
                linecolor='black')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix for {title}')
    plt.show()
    print(f"\nClassification Report for {title}:")
    print(report)


def clean_up_control_tokens(df):
    df['original'] = df['original'].apply(lambda x: x[1:-1] if isinstance(x, list) and len(x) > 2 else x)
    df['noisy'] = df['noisy'].apply(lambda x: x[1:-1] if isinstance(x, list) and len(x) > 2 else x)
    return df


if __name__ == '__main__':
    sns.set_style('whitegrid', rc={
        'xtick.bottom': True,
        'ytick.left': True,
    })
    #plot_token_distribution(noisy_010)
    df = load_combined_parse_statistics()
    df_seq = load_combined_sequence_statistics()
    df_seq = clean_up_control_tokens(df_seq)
    df_joined = df.join(df_seq)
    df_joined = df_joined[df_joined['baseCompiles'] != True]
    df = df[df['baseCompiles'] != True]

    # calculate total fix % and per-file fix %
    #der_matches, der_entries = count_matches_and_entries(df, 'derOnErrorRecoveryOperations')
    #antlr_matches, antlr_entries = count_matches_and_entries(df, 'antlrRecoveryOperations')
    #der_percentage = (der_matches / der_entries * 100) if der_entries > 0 else 0
    #antlr_percentage = (antlr_matches / antlr_entries * 100) if antlr_entries > 0 else 0
    #print(f"DER On Error Recovery Operations Match Percentage: {der_percentage:.2f}%")
    #print(f"ANTLR Error Recovery Operations Match Percentage: {antlr_percentage:.2f}%")
#
    ## fix % of FIRST error
    #antlr_matches, antlr_entries = count_matches_for_lowest_position(df, 'antlrRecoveryOperations')
    #der_percentage = (der_matches / der_entries * 100) if der_entries > 0 else 0
    #antlr_percentage = (antlr_matches / antlr_entries * 100) if antlr_entries > 0 else 0
    #print(f"DER On Error Recovery Operations (Lowest Position) Match Percentage: {der_percentage:.2f}%")
    #print(f"ANTLR Error Recovery Operations (Lowest Position) Match Percentage: {antlr_percentage:.2f}%")
#
    #tokenAfter_derOnError = []
    #tokenOriginal_derOnError = []
    #tokenAfter_antlr = []
    #tokenOriginal_antlr = []
#
    #for _, row in df.iterrows():
    #    extract_tokens(row['derOnErrorRecoveryOperations'], tokenAfter_derOnError, tokenOriginal_derOnError)
    #    extract_tokens(row['antlrRecoveryOperations'], tokenAfter_antlr, tokenOriginal_antlr)
#
    ## Analyze Recovery Operations
    #plot_operations(tokenOriginal_derOnError, tokenAfter_derOnError, 'derOnError')
    #plot_operations(tokenOriginal_antlr, tokenAfter_antlr, 'antlr')

    violinplot_reconstruction_split(df, 'figures/reconstruction_total_overview.pdf')
    violinplot_on_file_reconstruction_split(df, "figures/on_file_reconstruction.pdf")
    #df.to_csv("parse_statistics.csv", index=False)
    #df.to_csv("parse_statistics_idx.csv", index=True)
    #df_joined.to_csv("joined.csv", index=False)
    #df_joined.to_csv("joined_idx.csv", index=True)
    #df = pd.read_csv("parse_statistics.csv")
    #df.baseCompilationErrors = df.baseCompilationErrors.astype('object')
    #df_joined = pd.read_csv("joined.csv")
    #df_joined.baseCompilationErrors = df_joined.baseCompilationErrors.astype('object')
    df_summary = df[df['baseCompiles'] != True]
    summary_table_by_noise_level = df[["noiseLevel", "baseReconstruction", "antlrReconstruction", "derOnErrorReconstruction"]].groupby('noiseLevel').describe().transpose().unstack()
    summary_table_by_noise_level = summary_table_by_noise_level.round(4)
    #print(summary_table_by_noise_level.to_latex())
    print(summary_table_by_noise_level.to_csv())


    #extracted_df = df.apply(extract_words, axis=1)
    #df_joined[['extracted_original', 'extracted_noisy', 'extracted_antlr', 'extracted_derOnError']] = extracted_df

    # levenshtein accuracy
    df_joined['levenshtein_base'] = df_joined.apply(normalized_levenshtein, col1='original', col2='noisy', axis=1)
    df_joined['levenshtein_antlr'] = df_joined.apply(normalized_levenshtein, col1='original', col2='antlrErPrediction', axis=1)
    df_joined['levenshtein_derOnError'] = df_joined.apply(normalized_levenshtein, col1='original', col2='derOnErrorPrediction', axis=1)
    df_joined['levenshtein_derOnFile'] = df_joined.apply(normalized_levenshtein, col1='original',
                                                          col2='derOnFilePrediction', axis=1)
    violinplot_levenshtein(df_joined, "figures/levenshtein_distribution.pdf")
    violinplot_on_file_levenshtein(df_joined, "figures/levenshtein_on_file_distribution.pdf")
    levenshtein_table_by_noise_level = df_joined[["noiseLevel", "levenshtein_base", "levenshtein_antlr", "levenshtein_derOnError", "levenshtein_derOnFile"]].groupby('noiseLevel').describe().transpose().unstack()
    levenshtein_table_by_noise_level = levenshtein_table_by_noise_level.round(4)
    # print(summary_table_by_noise_level.to_latex())
    print(levenshtein_table_by_noise_level.to_csv())

    # distances for samples where DER partially reduces comp.errs
    df_reduced = df_joined[df_joined["baseNumOfCompilationErrors"] > df_joined['derOnErrorNumOfCompilationErrors']]
    violinplot_on_file_reconstruction_split(df_reduced, 'figures/reconstruction_on_file_partial_overview.pdf')
    violinplot_reconstruction_split(df_reduced, 'figures/reconstruction_partial_overview.pdf')

    df_reduced['levenshtein_base'] = df_reduced.apply(normalized_levenshtein, col1='original', col2='noisy', axis=1)
    df_reduced['levenshtein_antlr'] = df_reduced.apply(normalized_levenshtein, col1='original', col2='antlrErPrediction',
                                                     axis=1)
    df_reduced['levenshtein_derOnError'] = df_reduced.apply(normalized_levenshtein, col1='original',
                                                          col2='derOnErrorPrediction', axis=1)
    df_reduced['levenshtein_derOnFile'] = df_reduced.apply(normalized_levenshtein, col1='original',
                                                         col2='derOnFilePrediction', axis=1)
    violinplot_levenshtein(df_reduced, "figures/levenshtein_partial_distribution.pdf")
    violinplot_on_file_levenshtein(df_reduced, "figures/levenshtein_on_file_partial_distribution.pdf")
    levenshtein_table_by_noise_level = df_reduced[
        ["noiseLevel", "levenshtein_base", "levenshtein_antlr", "levenshtein_derOnError",
         "levenshtein_derOnFile"]].groupby('noiseLevel').describe().transpose().unstack()
    levenshtein_table_by_noise_level = levenshtein_table_by_noise_level.round(4)
    # print(summary_table_by_noise_level.to_latex())
    print(levenshtein_table_by_noise_level.to_csv())

    # calculate accuracy distribution
    #df_joined[['accuracy_base', 'accuracy_antlr', 'accuracy_derOnError']] = df_joined.apply(calculate_row_accuracy, axis=1)
    #violinplot_accuracy_split(df, "figures/base_accuracy.pdf")
    #violinplot_accuracy(df_joined, "accuracy_base", "figures/base_accuracy.pdf", base=True)
    #violinplot_accuracy(df_joined, "accuracy_antlr", "figures/antlr_accuracy.pdf", base=True)
    #violinplot_accuracy(df_joined, "accuracy_derOnError", "figures/accuracy_derOnError.pdf", base=True)

    # compute confusion matrices
    #df_grouped = df.groupby("noiseLevel")

    #for group, df_per_noise in df_grouped:
    #    original_flat = [word for sublist in df_per_noise['extracted_original'].tolist() for word in sublist]
    #    noisy_flat = [word for sublist in df_per_noise['extracted_noisy'].tolist() for word in sublist]
    #    antlr_flat = [word for sublist in df_per_noise['extracted_antlr'].tolist() for word in sublist]
    #    derOnError_flat = [word for sublist in df_per_noise['extracted_derOnError'].tolist() for word in sublist]
    #    labels = list(set(original_flat + noisy_flat + antlr_flat + derOnError_flat))
    #    #labels = list(set(original_flat) | set(noisy_flat) | set(antlr_flat) | set(derOnError_flat))
#
    #    cm_base = confusion_matrix(original_flat, noisy_flat, labels=labels, normalize='true')
    #    cm_antlr = confusion_matrix(original_flat, antlr_flat, labels=labels, normalize='true')
    #    cm_derOnError = confusion_matrix(original_flat, derOnError_flat, labels=labels, normalize='true')
#
    #    report_base = pd.DataFrame(classification_report(original_flat, noisy_flat, zero_division=0.0, output_dict=True)).transpose()
    #    report_antlr = pd.DataFrame(classification_report(original_flat, antlr_flat, zero_division=0.0, output_dict=True)).transpose()
    #    report_der = pd.DataFrame(classification_report(original_flat, derOnError_flat, zero_division=0.0, output_dict=True)).transpose()
    #    # Saving the DataFrame to a CSV file
    #    base_label = str(group).replace('.', '_')
    #    report_base.to_csv(f'figures/base_{base_label}_classification_report.csv', index=True)
    #    report_antlr.to_csv(f'figures/antlr_{base_label}_classification_report.csv', index=True)
    #    report_der.to_csv(f'figures/der_{base_label}_classification_report.csv', index=True)
#
    #    # ANTLR
    #    plt.figure(figsize=(30, 25))  # Adjust the figure size as necessary
    #    sns.heatmap(cm_base, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels, linewidths=.5, linecolor='black')
    #    plt.xlabel('Predicted')
    #    plt.ylabel('True')
    #    plt.title(f'Baseline Original vs Noisy on Noise Level: {group}')
    #    plt.grid(True)
    #    plt.savefig(f"figures/base_{base_label}_confmatrix.pdf", dpi=300)
    #    plt.show()
#
    #    # ANTLR
    #    plt.figure(figsize=(30, 25))  # Adjust the figure size as necessary
    #    sns.heatmap(cm_antlr, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels, linewidths=.5, linecolor='black')
    #    plt.xlabel('Predicted')
    #    plt.ylabel('True')
    #    plt.title(f'Original vs ANTLR ER Prediction on Noise Level: {group}')
    #    plt.grid(True)
    #    plt.savefig(f"figures/antlr_{base_label}_confmatrix.pdf", dpi=300)
    #    plt.show()
    #    # DER
    #    plt.figure(figsize=(30, 25))  # Adjust the figure size as necessary
    #    sns.heatmap(cm_derOnError, annot=False, fmt="d", cmap='Blues', xticklabels=labels, yticklabels=labels, linewidths=.5, linecolor='black')
    #    plt.xlabel('Predicted')
    #    plt.ylabel('True')
    #    plt.title(f'Original vs Deep Error Recovery Prediction on Noise Level: {group}')
    #    plt.grid(True)
    #    plt.savefig(f"figures/der_{base_label}_confmatrix.pdf", dpi=300)
    #    plt.show()
#
    #    # accuracy
    #    accuracy_base = compute_accuracy(cm_base)
    #    accuracy_antlr = compute_accuracy(cm_antlr)
    #    accuracy_derOnError = compute_accuracy(cm_derOnError)
#
    #    print(f"Accuracy Base on Noise {group}: {accuracy_base:.4f}")
    #    print(f"Accuracy ANTLR Prediction on Noise {group}: {accuracy_antlr:.4f}")
    #    print(f"Accuracy DerOnError Prediction on Noise {group}: {accuracy_derOnError:.4f}")



