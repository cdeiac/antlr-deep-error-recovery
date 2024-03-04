import itertools

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import Levenshtein as lev
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import re

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
        #pd.read_json("src/main/python/data/generated/cv/00_005/sequence_statistics1.json"),
        #pd.read_json("src/main/python/data/generated/cv/00_005/sequence_statistics2.json"),
        #pd.read_json("src/main/python/data/generated/cv/00_010/sequence_statistics0.json"),
        #pd.read_json("src/main/python/data/generated/cv/00_010/sequence_statistics1.json"),
        #pd.read_json("src/main/python/data/generated/cv/00_010/sequence_statistics2.json"),
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


def clean_up_control_tokens(df):
    df['original'] = df['original'].apply(lambda x: x[1:-1] if isinstance(x, list) and len(x) > 2 else x)
    df['noisy'] = df['noisy'].apply(lambda x: x[1:-1] if isinstance(x, list) and len(x) > 2 else x)
    return df


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


def extract_token_values(dataframe, row_name):
    tokenOriginal_list = []
    tokenAfter_list = []

    content_list = dataframe.loc[row_name, 'content']
    if content_list:
        for item in content_list:
            tokenOriginal_list.append(item['tokenOriginal'])
            tokenAfter_list.append(item['tokenAfter'])
    return tokenOriginal_list, tokenAfter_list


def extract_token_values_grouped(dataframe, column_name):
    # Group the DataFrame by the "noiseLevel" column
    grouped = dataframe.groupby('noiseLevel')

    # Initialize an empty dictionary to store tokenOriginal and tokenAfter lists for each group
    token_values_by_group = {}

    # Iterate over each group
    for group_name, group_data in grouped:
        # Initialize empty lists to store tokenOriginal and tokenAfter values for the current group
        tokenOriginal_list = []
        tokenAfter_list = []

        # Extract the content list from the specified column in the current group
        content_list = group_data[column_name].tolist()

        # Iterate over each list in the content_list
        for sublist in content_list:
            # Check if the sublist is not empty
            if sublist:
                # Iterate over each dictionary in the sublist
                for item in sublist:
                    # Check if tokenOriginal is None and replace it with 'EOF'
                    if item['tokenOriginal'] is None:
                        tokenOriginal_list.append('EOF')
                    else:
                        tokenOriginal_list.append(item['tokenOriginal'])

                    # Check if tokenAfter is None and replace it with 'EOF'
                    if item['tokenAfter'] is None:
                        tokenAfter_list.append('EOF')
                    else:
                        tokenAfter_list.append(item['tokenAfter'])

        # Store tokenOriginal and tokenAfter lists for the current group in the dictionary
        token_values_by_group[group_name] = {
            'tokenOriginal': tokenOriginal_list,
            'tokenAfter': tokenAfter_list
        }

    # Return the dictionary containing tokenOriginal and tokenAfter lists for each group
    return token_values_by_group


def compute_classification_report(target_list, prediction_list):
    # Ensure both lists are of the same length
    if len(target_list) != len(prediction_list):
        raise ValueError("Length of target_list and prediction_list must be the same")

    # Compute the classification report
    return classification_report(target_list, prediction_list, zero_division=0.0)


def plot_confusion_matrix(target_list, prediction_list, labels, type, noise):
    # Compute the confusion matrix
    cm = confusion_matrix(target_list, prediction_list, labels=labels, normalize='true')

    # Plot the confusion matrix
    plt.figure(figsize=(30, 25))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, linewidths=.5,
                linecolor='black')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig("figures/" + type + "_" + str(noise).replace('.', '_') + "_cmat_")
    plt.show()


def process_classification_report(classification_report_str):
    # Split the classification report string into lines
    lines = classification_report_str.split('\n')

    # Initialize dictionaries to store f1-scores and support values
    f1_scores = {}
    supports = {}

    # Iterate over each line and extract f1-scores and support values
    for line in lines[2:-5]:  # Skip header and footer lines
        data = re.split(r'\s+', line.strip())
        label = data[0]
        precision = float(data[1])
        recall = float(data[2])
        f1 = float(data[3])
        support = int(data[4])

        f1_scores[label] = f1
        supports[label] = support

    # Extract overall accuracy
    accuracy_line = lines[-3]
    accuracy_data = re.split(r'\s+', accuracy_line.strip())
    #accuracy = float(accuracy_data[1])

    # Filter labels with support >= 100
    top_labels = [label for label, support in supports.items() if support >= 100]

    # Sort top labels by f1-score in descending order
    top_labels = sorted(top_labels, key=lambda label: f1_scores[label], reverse=True)[:10]

    # Extract f1-scores for the top labels
    top_f1_scores = {label: f1_scores[label] for label in top_labels}

    return top_labels, top_f1_scores




if __name__ == '__main__':
    sns.set_style('whitegrid', rc={
        'xtick.bottom': True,
        'ytick.left': True,
    })
    #plot_token_distribution(noisy_010)
    df = load_combined_parse_statistics()
    #df_seq = load_combined_sequence_statistics()
    #df_seq = clean_up_control_tokens(df_seq)
    #df_joined = df.join(df_seq)
    #df_joined = df_joined[df_joined['baseCompiles'] != True]
    df = df[df['baseCompiles'] != True]

    antlr = extract_token_values_grouped(df, 'antlrRecoveryOperations')
    der = extract_token_values_grouped(df, 'derOnErrorRecoveryOperations')

    for key in antlr:
        print(key)
        value = antlr[key]
        report = compute_classification_report(value["tokenOriginal"], value["tokenAfter"])
        print(report)
        labels = list(set(value["tokenOriginal"] + value["tokenAfter"]))
        plot_confusion_matrix(value["tokenOriginal"], value["tokenAfter"], labels, "antlr", key)
        top_labels, top_f1_scores = process_classification_report(report)
        print("Top 10 labels with highest f1-score:", top_labels)
        print("F1-scores for the top 10 labels:", top_f1_scores)
        print("========================================")

    for key in der:
        print(key)
        value = der[key]
        report = compute_classification_report(value["tokenOriginal"], value["tokenAfter"])
        print(report)
        labels = list(set(value["tokenOriginal"] + value["tokenAfter"]))
        plot_confusion_matrix(value["tokenOriginal"], value["tokenAfter"], labels, "der", key)
        top_labels, top_f1_scores = process_classification_report(report)
        print("Top 10 labels with highest f1-score:", top_labels)
        print("F1-scores for the top 10 labels:", top_f1_scores)
        print("========================================")