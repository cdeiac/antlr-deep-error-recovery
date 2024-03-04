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
    return df_combined.drop(columns=[col for col in df_combined.columns if 'bail' in col.lower()])


def preprocess_df(dataframe, noise_level):
    dataframe['noiseLevel'] = noise_level
    # base
    dataframe['baseNumOfCompilationErrors'] = dataframe['baseCompilationErrors'].apply(lambda x: len(x))
    dataframe['baseNumOfMissingErrors'] = dataframe['baseCompilationErrors'].apply(lambda x: count_errors(x, 'MissingError'))
    dataframe['baseNumOfMismatchedErrors'] = dataframe['baseCompilationErrors'].apply(lambda x: count_errors(x, 'MismatchedError'))
    dataframe['baseNumOfExtraneousInputErrors'] = dataframe['baseCompilationErrors'].apply(lambda x: count_errors(x, 'ExtraneousInputError'))
    dataframe['baseCompiles'] = dataframe['baseNumOfCompilationErrors'] == 0
    # antlr ER
    dataframe['antlrNumOfCompilationErrors'] = dataframe['antlrCompilationErrors'].apply(lambda x: len(x))
    dataframe['antlrNumOfMissingErrors'] = dataframe['antlrCompilationErrors'].apply(lambda x: count_errors(x, 'MissingError'))
    dataframe['antlrNumOfMismatchedErrors'] = dataframe['antlrCompilationErrors'].apply(lambda x: count_errors(x, 'MismatchedError'))
    dataframe['antlrNumOfExtraneousInputErrors'] = dataframe['antlrCompilationErrors'].apply(lambda x: count_errors(x, 'ExtraneousInputError'))
    # deep ER on-error
    dataframe['derOnErrorNumOfCompilationErrors'] = dataframe['derOnErrorCompilationErrors'].apply(lambda x: len(x))
    dataframe['derOnErrorNumOfMissingErrors'] = dataframe['derOnErrorCompilationErrors'].apply(lambda x: count_errors(x, 'MissingError'))
    dataframe['derOnErrorNumOfMismatchedErrors'] = dataframe['derOnErrorCompilationErrors'].apply(lambda x: count_errors(x, 'MismatchedError'))
    dataframe['derOnErrorNumOfExtraneousInputErrors'] = dataframe['derOnErrorCompilationErrors'].apply(lambda x: count_errors(x, 'ExtraneousInputError'))
    # deep ER on-file
    dataframe['derOnFileNumOfCompilationErrors'] = dataframe['derOnFileCompilationErrors'].apply(lambda x: len(x))
    dataframe['derOnFileNumOfMissingErrors'] = dataframe['derOnFileCompilationErrors'].apply(lambda x: count_errors(x, 'MissingError'))
    dataframe['derOnFileNumOfMismatchedErrors'] = dataframe['derOnFileCompilationErrors'].apply(lambda x: count_errors(x, 'MismatchedError'))
    dataframe['derOnFileNumOfExtraneousInputErrors'] = dataframe['derOnFileCompilationErrors'].apply(lambda x: count_errors(x, 'ExtraneousInputError'))
    return dataframe.drop(columns=[col for col in dataframe.columns if 'bail' in col.lower()])


def count_errors(arr, error_type):
    return sum(1 for elem in arr if elem.get('type') == error_type)


def process_row(row):
    antlr_map, der_map = {}, {}
    antlr_count, der_count = {"match": 0, "no_match": 0}, {"match": 0, "no_match": 0}

    base_positions = [error['position'] for error in row['baseCompilationErrors']]
    antlr_positions = [error['position'] for error in row['antlrCompilationErrors']]
    der_positions = [error['position'] for error in row['derOnErrorCompilationErrors']]

    for pos in base_positions:
        # For antlrErPrediction
        if pos in antlr_positions:
            antlr_count["match"] += 1
        else:
            antlr_count["no_match"] += 1
        try:
            antlr_map[pos] = (row['original'][pos], row['antlrErPrediction'][pos])
        except IndexError:
            # position not present anymore due to deletions
            pass

        # For derPrediction
        if pos in der_positions:
            der_count["match"] += 1
        else:
            der_count["no_match"] += 1
        try:
            der_map[pos] = (row['original'][pos], row['derOnErrorPrediction'][pos])
        except IndexError:
            # position not present anymore due to deletions
            pass
    return antlr_map, der_map, antlr_count, der_count


def process_data(row):
    position_details = {}

    # Extract lists of positions for antlr and der errors
    antlr_positions = {error['position'] for error in row['antlrCompilationErrors']}
    der_positions = {error['position'] for error in row['derOnErrorCompilationErrors']}

    for error in row['baseCompilationErrors']:
        pos = error['position']
        # Initialize the details dictionary for this position
        details = {
            'original': row['original'][pos] if pos < len(row['original']) else None,
            'antlrErPrediction': row['antlrErPrediction'][pos] if pos in antlr_positions and pos < len(row['antlrErPrediction']) else None,
            'derOnErrorPrediction': row['derOnErrorPrediction'][pos] if pos in der_positions and pos < len(row['derOnErrorPrediction']) else None,
        }
        position_details[pos] = details

    return position_details


def normalize_positions(df, error_col, text_col, series_name):
    normalized_positions = []

    for _, row in df.iterrows():
        positions = [error['position'] for error in row[error_col]]
        text_length = len(row[text_col])
        normalized = [pos / text_length for pos in positions if text_length > 0]
        normalized_positions.extend(normalized)

    # Return a DataFrame with normalized positions and the series name
    return pd.DataFrame({'Normalized Position': normalized_positions, 'Series': series_name})


def normalize_errors(row):
    base_positions = {error['position'] for error in row['baseCompilationErrors']}
    antlr_positions = {error['position'] for error in row['antlrCompilationErrors']}
    der_positions = {error['position'] for error in row['derOnErrorCompilationErrors']}

    base_total = len(base_positions)
    if base_total == 0:
        return pd.Series([None, None])  # Avoid division by zero

    # count how often baseCompilationError positions appear in other error lists
    antlr_overlap = len(base_positions.intersection(antlr_positions)) / base_total
    der_overlap = len(base_positions.intersection(der_positions)) / base_total

    # Subtract normalized values from 1
    return pd.Series([1 - antlr_overlap, 1 - der_overlap])


def plot_avg_errors(df):
    base_avg_errors = df['baseNumOfCompilationErrors'].mean()
    antlr_avg_errors = df['antlrNumOfCompilationErrors'].mean()
    der_avg_errors = df['derOnErrorNumOfCompilationErrors'].mean()
    categories = ['Baseline', 'ANTLR', 'Deep']
    values = [base_avg_errors, antlr_avg_errors, der_avg_errors]
    plt.figure(figsize=(8, 6))  # Optional: Adjusts the figure size
    sns.barplot(x=categories, y=values, palette='Set2')  # You can customize the colors
    # Adding title and labels
    plt.xlabel('Categories')
    plt.ylabel('Avg. Syntax Errors')
    plt.show()


def plot_avg_errors_by_noise_level(df):
    # Reshape the DataFrame to long format
    df_long = df.melt(id_vars=['noiseLevel'],
                      value_vars=['baseNumOfCompilationErrors', 'antlrNumOfCompilationErrors',
                                  'derOnErrorNumOfCompilationErrors'],
                      var_name='Category', value_name='AvgSyntaxErrors')

    # Map the original column names to more readable category names
    category_mapping = {
        'baseNumOfCompilationErrors': 'Baseline',
        'antlrNumOfCompilationErrors': 'ANTLR',
        'derOnErrorNumOfCompilationErrors': 'Deep'
    }

    df_long['Category'] = df_long['Category'].map(category_mapping)

    # Now plot with seaborn
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x='noiseLevel', y='AvgSyntaxErrors', hue='Category', data=df_long, palette='Set2')

    plt.xlabel('Noise Level (%)')
    plt.ylabel('Avg. Syntax Errors')
    legend = plt.legend(title='Error Recovery', loc='upper center', fancybox=False, shadow=False, ncol=3)
    legend.set_bbox_to_anchor((0.2, -0.05))
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("figures/compilation_errors_overview.pdf")
    plt.show()


def plot_avg_extraneous_errors_by_noise_level(df):
    # Reshape the DataFrame to long format
    df_long = df.melt(id_vars=['noiseLevel'],
                      value_vars=['baseNumOfExtraneousInputErrors', 'antlrNumOfExtraneousInputErrors',
                                  'derOnErrorNumOfExtraneousInputErrors'],
                      var_name='Category', value_name='AvgSyntaxErrors')

    # Map the original column names to more readable category names
    category_mapping = {
        'baseNumOfExtraneousInputErrors': 'Baseline',
        'antlrNumOfExtraneousInputErrors': 'ANTLR',
        'derOnErrorNumOfExtraneousInputErrors': 'Deep'
    }

    df_long['Category'] = df_long['Category'].map(category_mapping)

    # Now plot with seaborn
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x='noiseLevel', y='AvgSyntaxErrors', hue='Category', data=df_long, palette='Set2')

    plt.xlabel('Noise Level (%)')
    plt.ylabel('Avg. Extraneous Input Syntax Errors')
    plt.legend(title='Error Recovery')
    plt.savefig("figures/extraneous_errors_overview.pdf")
    plt.show()


def plot_avg_mismatched_errors_by_noise_level(df):
    # Reshape the DataFrame to long format
    df_long = df.melt(id_vars=['noiseLevel'],
                      value_vars=['baseNumOfMismatchedErrors', 'antlrNumOfMismatchedErrors',
                                  'derOnErrorNumOfMismatchedErrors'],
                      var_name='Category', value_name='AvgSyntaxErrors')

    # Map the original column names to more readable category names
    category_mapping = {
        'baseNumOfMismatchedErrors': 'Baseline',
        'antlrNumOfMismatchedErrors': 'ANTLR',
        'derOnErrorNumOfMismatchedErrors': 'Deep'
    }

    df_long['Category'] = df_long['Category'].map(category_mapping)

    # Now plot with seaborn
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x='noiseLevel', y='AvgSyntaxErrors', hue='Category', data=df_long, palette='Set2')

    plt.xlabel('Noise Level (%)')
    plt.ylabel('Avg. Mismatched Symbol Syntax Errors')
    plt.legend(title='Error Recovery')
    plt.savefig("figures/mismatched_errors_overview.pdf")
    plt.show()


def plot_avg_missing_errors_by_noise_level(df):

    # Reshape the DataFrame to long format
    df_long = df.melt(id_vars=['noiseLevel'],
                      value_vars=['baseNumOfMissingErrors', 'antlrNumOfMissingErrors',
                                  'derOnErrorNumOfMissingErrors'],
                      var_name='Category', value_name='AvgSyntaxErrors')

    # Map the original column names to more readable category names
    category_mapping = {
        'baseNumOfMissingErrors': 'Baseline',
        'antlrNumOfMissingErrors': 'ANTLR',
        'derOnErrorNumOfMissingErrors': 'Deep'
    }

    df_long['Category'] = df_long['Category'].map(category_mapping)

    # Now plot with seaborn
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x='noiseLevel', y='AvgSyntaxErrors', hue='Category', data=df_long, palette='Set2')

    plt.xlabel('Noise Level (%)')
    plt.ylabel('Avg. Missing Symbol Syntax Errors')
    plt.legend(title='Error Recovery')
    plt.savefig("figures/missing_errors_overview.pdf")
    plt.show()


def violinplot_errors_split(df, filename):
    melted_df = df.melt(id_vars=['noiseLevel'], value_vars=['baseNumOfCompilationErrors', 'antlrNumOfCompilationErrors',
                                  'derOnErrorNumOfCompilationErrors'],
                        var_name='Type', value_name='Count')

    fig, ax = plt.subplots(1, 3, figsize=(30, 6), dpi=300)

    sns.violinplot(x='Type', y='Count', data=melted_df[melted_df['noiseLevel'] == 0.001], ax=ax[0], palette='Set2', fill=False, cut=0, inner="quart")
    sns.violinplot(x='Type', y='Count', data=melted_df[melted_df['noiseLevel'] == 0.005], ax=ax[1], palette='Set2', fill=False, cut=0, inner="quart")
    sns.violinplot(x='Type', y='Count', data=melted_df[melted_df['noiseLevel'] == 0.010], ax=ax[2], palette='Set2', fill=False, cut=0, inner="quart")

    for a in ax:
        for line in a.lines:
            line.set_linestyle('-')
            line.set_color(line.get_color())


    legend = plt.legend(title='Error Recovery', loc='upper center', fancybox=False, shadow=False, ncol=3)
    #legend.get_texts()[0].set_text('Baseline')
    #legend.get_texts()[1].set_text('ANTLR')
    #legend.get_texts()[2].set_text('Deep')
    #legend.set_bbox_to_anchor((0.1, -0.1))
    plt.ylabel("Number of Syntax Errors")
    plt.xlabel("Noise Level (%)", labelpad=15)
    plt.grid(True, axis='y', linestyle='--')
    plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def compute_der_fix_percentage(group):
    # Number of times col is lower than base
    #condition_lower = df[col] < df['baseNumOfCompilationErrors']
    #count_lower = condition_lower.sum()
    ## Number of times col is 0
    #condition_zero = df[col] == 0
    #count_zero = condition_zero.sum()
#
    #total_rows = len(df)
    #score_error_reduction = count_lower / total_rows
    #score_fix_percentage = count_zero / total_rows
    #return score_error_reduction, score_fix_percentage

    col1 = 'baseNumOfCompilationErrors'
    col2 = 'derOnErrorNumOfCompilationErrors'

    # Condition 1: Number of times col2 is lower than col1
    condition_lower = group[col2] < group[col1]
    count_lower = condition_lower.sum()

    # Condition 2: Number of times col2 is 0
    condition_zero = group[col2] == 0
    count_zero = condition_zero.sum()

    # Total number of rows in the group
    total_rows = len(group)

    # Compute scores
    score1 = count_lower / total_rows
    score2 = count_zero / total_rows

    return pd.Series({'error_reduction': score1, 'fix_percentage': score2})

def compute_antlr_fix_percentage(group):
    col1 = 'baseNumOfCompilationErrors'
    col2 = 'antlrNumOfCompilationErrors'

    # Condition 1: Number of times col2 is lower than col1
    condition_lower = group[col2] < group[col1]
    count_lower = condition_lower.sum()

    # Condition 2: Number of times col2 is 0
    condition_zero = group[col2] == 0
    count_zero = condition_zero.sum()

    # Total number of rows in the group
    total_rows = len(group)

    # Compute scores
    score1 = count_lower / total_rows
    score2 = count_zero / total_rows

    return pd.Series({'error_reduction': score1, 'fix_percentage': score2})


def filter_der_successful_fixes(group):
    filtered_group = group[group['baseNumOfCompilationErrors'] > group['derOnErrorNumOfCompilationErrors']]
    filtered_group['Absolute Difference'] =  filtered_group['baseNumOfCompilationErrors'] - filtered_group['derOnErrorNumOfCompilationErrors']
    filtered_group['Percentage Difference'] = (filtered_group['Absolute Difference'] / filtered_group[
        'baseNumOfCompilationErrors']) * 100
    return filtered_group[['Absolute Difference', 'Percentage Difference']]


def plot_fixed_errors(df):
    # prepare data
    df_long = pd.melt(df, id_vars=['noiseLevel'], value_vars=['antlr_overlap_norm', 'der_overlap_norm'],
                      var_name='Type', value_name='Value')

    # plot
    plt.figure(figsize=(10, 6), dpi=300)

    ax = sns.violinplot(data=df_long, x="noiseLevel", y="Value", hue="Type",
                        split=True, gap=.1, fill=False, cut=0, inner="quart", palette='Set2')

    for line in ax.lines:
        line.set_linestyle('-')
        line.set_color(line.get_color())

    legend = plt.legend(title='Error Recovery', loc='upper center', fancybox=False, shadow=False, ncol=2)
    legend.get_texts()[0].set_text('ANTLR')
    legend.get_texts()[1].set_text('Deep')
    legend.set_bbox_to_anchor((0.1, -0.1))
    plt.ylabel("Successful Recovery of Total Syntax Errors")
    plt.xlabel("Noise Level (%)", labelpad=15)
    plt.gca().set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.grid(True, axis='y', linestyle='--')
    # plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig("figures/successful_recovery.pdf", dpi=300)
    plt.show()


def plot_file_size_of_part_recovery(df):
    df['size'] = df['original'].apply(len)

    # Plotting
    plt.figure(figsize=(10, 6), dpi=300)
    sns.violinplot(x='noiseLevel', y='size', data=df, fill=False, cut=0, inner="quart")
    plt.xlabel('Noise Level (%)')
    plt.ylabel('File Size')
    plt.grid(True, axis='y', linestyle='--')
    # plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig("figures/file_size_of_part_recovery.pdf")
    plt.show()


def plot_file_size_of_succ_recovery(df):
    df['size'] = df['original'].apply(len)

    # Plotting
    plt.figure(figsize=(10, 6), dpi=300)
    sns.violinplot(x='noiseLevel', y='size', data=df, fill=False, cut=0, inner="quart")
    plt.xlabel('Noise Level (%)')
    plt.ylabel('File Size')
    plt.grid(True, axis='y', linestyle='--')
    # plt.ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig("figures/file_size_of_succ_recovery.pdf")
    plt.show()


def plot_succ_recovery_by_error_type(df):
    df_long = pd.melt(df, id_vars=['noiseLevel'],
                      value_vars=['baseNumOfMissingErrors', 'baseNumOfMismatchedErrors',
                                  'baseNumOfExtraneousInputErrors'],
                      var_name='ErrorType', value_name='ErrorCount')

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.barplot(x='noiseLevel', y='ErrorCount', hue='ErrorType', data=df_long, palette="Set1")
    plt.xlabel('Noise Level (%)')
    plt.ylabel('Avg. Number of Errors')
    legend = plt.legend(title='Syntax Error Type', loc='upper right')
    legend.get_texts()[0].set_text('Missing Symbol')
    legend.get_texts()[1].set_text('Mismatched Symbol')
    legend.get_texts()[2].set_text('Extraneous Input')
    plt.tight_layout()
    plt.savefig("figures/succ_recovery_by_err_type.pdf")
    plt.show()


if __name__ == '__main__':
    sns.set_style('whitegrid', rc={
        'xtick.bottom': True,
        'ytick.left': True,
    })
    df = load_combined_parse_statistics()
    #df_seq = load_combined_sequence_statistics()
    #df_joined = df.join(df_seq)
    #df_joined = df_joined[df_joined['baseCompiles'] != True]
    df = df[df['baseCompiles'] != True]
#
    # number of compilation errors
    #plot_avg_errors(df)
    #plot_avg_errors_by_noise_level(df)
    #plot_avg_missing_errors_by_noise_level(df)
    #plot_avg_mismatched_errors_by_noise_level(df)
    #plot_avg_extraneous_errors_by_noise_level(df)

    plot_succ_recovery_by_error_type(df[df["baseNumOfCompilationErrors"] > df['derOnErrorNumOfCompilationErrors']])
    plot_succ_recovery_by_error_type(df[df['derOnErrorNumOfCompilationErrors'] == 0])

    # per-file fixes
    antlr_err_scores = df.groupby('noiseLevel').apply(compute_antlr_fix_percentage)
    print(f"ANTLR Error Scores: {antlr_err_scores}")
    der_err_scores = df.groupby('noiseLevel').apply(compute_der_fix_percentage)
    print(f"DER Error Scores: {der_err_scores}")
    df_der_succ_fixes = df.groupby('noiseLevel').apply(filter_der_successful_fixes)


    #df_der_outperforms_base = df[df['baseNumOfCompilationErrors'] > df['derOnErrorNumOfCompilationErrors']]
    # how large is the discrepancy between the columns? How many errors were actually fixed?
    #df_der_outperforms_antlr = df[(df['baseNumOfCompilationErrors'] > df['derOnErrorNumOfCompilationErrors']) & (df['antlrNumOfCompilationErrors'] < df['derOnErrorNumOfCompilationErrors'])]

    #plot_file_size_of_part_recovery(df_joined[df_joined["baseNumOfCompilationErrors"] > df_joined['derOnErrorNumOfCompilationErrors']][['noiseLevel', 'original']])
    #plot_file_size_of_succ_recovery(df_joined[df_joined['derOnErrorNumOfCompilationErrors'] == 0][['noiseLevel', 'original']])

    #breakdown by type for succ. fixed
    # per-file fix %
    df[['antlr_overlap_norm', 'der_overlap_norm']] = df.apply(normalize_errors, axis=1)

    # how often are base comp. errors present in antlr comp. errs and der (1- x)
    print(df.groupby('noiseLevel')['antlr_overlap_norm'].describe().to_csv())
    print(df.groupby('noiseLevel')['der_overlap_norm'].describe().to_csv())
    plot_fixed_errors(df)

    ## Initialize an empty dictionary to store aggregated results
    #error_positions = {}
    ## Iterate through each row in the DataFrame and update the aggregated dictionary
    #for index, row in df_joined.iterrows():
    #    row_details = process_data(row)
    #    for pos, details in row_details.items():
    #        if pos not in error_positions:
    #            error_positions[pos] = []
    #        error_positions[pos].append(details)
#
    ## Initialize counters for each property
    #count_original = 0
    #count_antlrErPrediction = 0
    #count_derPrediction = 0
#
    ## Iterate through each position and its list of details
    #for pos, details_list in error_positions.items():
    #    for details in details_list:
    #        # Count non-None occurrences for each property
    #        if details['original'] is not None:
    #            count_original += 1
    #        if details['antlrErPrediction'] is not None:
    #            count_antlrErPrediction += 1
    #        if details['derOnErrorPrediction'] is not None:
    #            count_derPrediction += 1
#
    #accuracy_antlrErPrediction = count_antlrErPrediction / count_original if count_original > 0 else 0
    #accuracy_derPrediction = count_derPrediction / count_original if count_original > 0 else 0
#
    #print(f"Count of non-None 'original': {count_original}")
    #print(f"Count of non-None 'antlrErPrediction': {count_antlrErPrediction}")
    #print(f"Count of non-None 'derPrediction': {count_derPrediction}")
    #print(f"Accuracy for 'antlrErPrediction': {accuracy_antlrErPrediction:.4f}")
    #print(f"Accuracy for 'derPrediction': {accuracy_derPrediction:.4f}")
#
    ##error_positions = df_joined.apply(process_data, axis=1)
#
    #print(len(error_positions))





    # Normalize positions for each series
    #df_base = normalize_positions(df_joined, 'baseCompilationErrors', 'original', 'Base')
    #df_antlr = normalize_positions(df_joined, 'antlrCompilationErrors', 'antlrErPrediction', 'ANTLR')
    #df_der = normalize_positions(df_joined, 'derOnErrorCompilationErrors', 'derOnErrorPrediction', 'DER')
#
    #df_combined = pd.concat([df_base, df_antlr, df_der])
#
    ## Plotting the violin plot
    ##plt.figure(figsize=(10, 6))
    ##sns.violinplot(x='Series', y='Normalized Position', data=df_combined)
    ##plt.title('Distribution of Normalized Error Positions')
    ##plt.show()
    #plt.figure(figsize=(10, 6))
    #sns.stripplot(x='Series', y='Normalized Position', data=df_combined, jitter=True, dodge=True)
    #plt.title('Scatter Plot of Normalized Error Positions')
    #plt.show()
