import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from evaluation.compilation_errors import load_combined_parse_statistics


def compare_errors(group):
    return (group['derOnErrorNumOfCompilationErrors'] < group['antlrNumOfCompilationErrors']).sum() / len(group)

def categorize_errors(value):
    if value == 1:
        return '1'
    elif value < 5:
        return '5'
    elif value < 10:
        return '10'
    elif value < 100:
        return '100'
    else:
        return '101'


def plot_error_category(df, column):
    df['ErrorCategory'] = df[column].apply(categorize_errors)
    category_order = ['1', '5', '10', '100', '101']
    df['ErrorCategory'] = pd.Categorical(df['ErrorCategory'], categories=category_order, ordered=True)

    # Group by 'noiseLevel' and 'ErrorCategory', count, and compute percentages
    grouped = df.groupby(['noiseLevel', 'ErrorCategory']).size().reset_index(name='Counts')
    percentages = grouped.groupby('noiseLevel')['Counts'].apply(lambda x: x / x.sum() * 100).reset_index(
        name='Percentage')
    grouped['Percentage'] = percentages['Percentage']

    # Plot using Seaborn's barplot
    plt.figure(figsize=(10, 6), dpi=300)

    sns.barplot(x='noiseLevel', y='Percentage', hue='ErrorCategory', data=grouped, palette='Set1')
    plt.xlabel('Noise Level (%)')
    plt.ylabel('Syntax Errors (%)')
    custom_labels = ['1', '< 5', '< 10', '< 100', '> 100']
    legend = plt.legend(title='Recovered')
    for text, label in zip(legend.texts, custom_labels):
        text.set_text(label)
    plt.savefig("figures/recoveries_per_file.pdf")
    plt.show()


if __name__ == '__main__':
    sns.set_style('whitegrid', rc={
        'xtick.bottom': True,
        'ytick.left': True,
    })
    df = load_combined_parse_statistics()
    df = df[df['baseCompiles'] != True]
    df = df[['noiseLevel', 'baseNumOfCompilationErrors', 'antlrNumOfCompilationErrors', 'derOnErrorNumOfCompilationErrors']]

    # filter rows where DER fixes at least one error
    filtered_df = df[df['baseNumOfCompilationErrors'] > df['derOnErrorNumOfCompilationErrors']]
    # of the files that DER reduces errors, how often does DER outperform ANTLR?
    comparison_counts = filtered_df.groupby('noiseLevel').apply(compare_errors)

    #der_better_at_fixing_than_antlr = (df['derOnErrorNumOfCompilationErrors'] < df['antlrNumOfCompilationErrors']).sum() / len(filtered_df)

    # Compute the difference in absolute and percentage terms
    filtered_df['diff_base_der_abs'] = filtered_df['baseNumOfCompilationErrors'] - filtered_df['derOnErrorNumOfCompilationErrors']
    filtered_df['diff_base_der_perc'] = (filtered_df['diff_base_der_abs'] / filtered_df['baseNumOfCompilationErrors'])

    # Using original indexes to compute the same metric for 'antlrNumOfCompilationErrors'
    filtered_df['diff_base_antlr_abs'] = filtered_df['baseNumOfCompilationErrors'] - filtered_df['antlrNumOfCompilationErrors']
    filtered_df['diff_base_antlr_perc'] = (filtered_df['diff_base_antlr_abs'] / filtered_df['baseNumOfCompilationErrors'])

    # plot number of fixes per category
    plot_error_category(filtered_df, 'diff_base_der_abs')

    # Reshape for plotting
    melted_df = pd.melt(filtered_df, id_vars=['noiseLevel'], value_vars=['diff_base_der_abs', 'diff_base_der_perc', 'diff_base_antlr_abs', 'diff_base_antlr_perc'],
                        var_name='Metric', value_name='Value')

    # Filter melted_df for percentage differences only
    percentage_diff_df = melted_df[melted_df['Metric'].str.contains('perc')]

    # Filter melted_df for absolute differences only
    absolute_diff_df = melted_df[melted_df['Metric'].str.contains('abs')]

    ## Plotting percentage differences
    #plt.figure(figsize=(10, 6))
    #sns.violinplot(x='noiseLevel', y='Value', hue='Metric', data=percentage_diff_df, split=True, gap=.1, fill=False, palette='Set2')
    #plt.title('Percentage Differences of Compilation Errors by Noise Level')
    #plt.ylabel('Percentage Difference')
    #plt.xlabel('Noise Level')
    #plt.legend(title='Metric', loc='upper left')
    #plt.tight_layout()
    #plt.show()
#
    ## Plotting absolute differences
    #plt.figure(figsize=(10, 6))
    #sns.violinplot(x='noiseLevel', y='Value', hue='Metric', data=absolute_diff_df, split=True, gap=.1, fill=False, palette='Set2')
    #plt.title('Absolute Differences of Compilation Errors by Noise Level')
    #plt.ylabel('Absolute Difference')
    #plt.xlabel('Noise Level')
    #plt.legend(title='Metric', loc='upper left')
    #plt.tight_layout()
    #plt.show()