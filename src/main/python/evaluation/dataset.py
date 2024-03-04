import json

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

sns.set_theme(style='whitegrid')

def count_string_occurrences(input_list):
    return Counter(input_list)

def count_word_occurrences(sentences):
    word_counts = Counter()
    for sentence in sentences:
        words = sentence.split()  # Split the sentence into words
        word_counts.update(words)  # Update word counts
    return word_counts

def plot_token_distribution(filename):
    with open(filename, 'r') as f:
        dataset = json.load(f)

    repo_list = []
    for i in range(len(dataset)):
        repo_list.append(dataset[i]['source']['file']['repo'])
    print(count_string_occurrences(repo_list))

    original_data = []
    noisy_data = []
    for i in range(len(dataset)):
        original_data.append(dataset[i]['source']['source'])
        noisy_data.append(dataset[i]['source']['sourceWithNoise'])
    # count token occurrences across all sentences
    original_occurrences = count_word_occurrences(original_data)
    sorted_originals = sorted(original_occurrences.keys())
    noisy_occurrences = count_word_occurrences(noisy_data)
    sorted_noisy = sorted(noisy_occurrences.keys())
    # plot
    fig, axs = plt.subplots(2, 1, figsize=(30, 10))
    sns.barplot(ax=axs[0], x=sorted_originals, y=[original_occurrences[key] for key in sorted_originals])
    axs[0].set_ylabel('count (log)')
    axs[0].tick_params(axis='x', labelrotation=90)
    axs[0].set_yscale('log')
    axs[0].set_ylim(1, 10**7)

    sns.barplot(ax=axs[1], x=sorted_noisy, y=[noisy_occurrences[key] for key in sorted_noisy])
    axs[1].set_xlabel('tokens')
    axs[1].set_ylabel('count (log)')
    axs[1].tick_params(axis='x', labelrotation=90)
    axs[1].set_yscale('log')
    axs[1].set_ylim(1, 10 ** 7)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.8)
    plt.savefig('figures/noisy_occurrences_001.pdf', dpi=300)
    plt.show()





if __name__ == '__main__':
    # noisy files
    noisy_001 = "src/main/resources/generated/00_010/noisy_jhetas_clean.json"
    noisy_002 = "src/main/resources/generated/00_020/noisy_jhetas_clean.json"
    noisy_004 = "src/main/resources/generated/00_040/noisy_jhetas_clean.json"
    noisy_008 = "src/main/resources/generated/00_080/noisy_jhetas_clean.json"
    noisy_010 = "src/main/resources/generated/00_100/noisy_jhetas_clean.json"
    #plot_token_distribution(noisy_010)



