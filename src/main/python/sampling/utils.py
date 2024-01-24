import numpy as np
from collections import Counter

from matplotlib import pyplot as plt


def flatten_nested_list(nested_list):
    nested_list = [string.split() for string in nested_list]
    return [item for sublist in nested_list for item in sublist]


def find_top_tokens(data, top_n):
    flattened_list = flatten_nested_list(data)
    counts = Counter(flattened_list)
    return [item[0] for item in counts.most_common(top_n)]


def compute_percentage_of_top_n_tokens(lst, top_tokens):
    total_count = len(lst)
    string_counts = Counter(lst.split())
    return sum(string_counts[string] for string in top_tokens) / total_count * 100


def sort_by_percentage(nested_list, top_tokens):
    return sorted(range(len(nested_list)), key=lambda x: compute_percentage_of_top_n_tokens(nested_list[x], top_tokens))


def plot_token_occurrences(data, data_type):
    data = [string.split() for string in data]
    flatlist = [string for sublist in data for string in sublist]
    token_counts = Counter(flatlist)
    unique_tokens, counts = zip(*token_counts.items())
    # plot
    plt.figure(figsize=(26, 12))
    plt.bar(unique_tokens, counts)
    plt.yscale('log')  # Apply log scale on the y-axis
    plt.xlabel('tokens')
    plt.ylabel('count (log)')
    plt.title(f'Token Occurrences ({data_type})')
    plt.xticks(rotation=90)
    plt.show()


def count_unique_tokens_above_threshold(data, threshold):
    flatlist = flatten_nested_list(data)
    token_counts = Counter(flatlist)
    unique_tokens_above_threshold = {string for string, count in token_counts.items() if count > threshold}
    total_unique_tokens_above_threshold = len(unique_tokens_above_threshold)
    return total_unique_tokens_above_threshold


def sort_dataset_by_rare_tokens_with_occurrence_below_threshold(x_data, y_data, threshold):
    tokens_above_threshold = count_unique_tokens_above_threshold(x_data, threshold)
    top_tokens = find_top_tokens(x_data, top_n=tokens_above_threshold)
    sorted_nested_idx = sort_by_percentage(x_data, top_tokens)
    return [x_data[i] for i in sorted_nested_idx], [y_data[i] for i in sorted_nested_idx]


def compute_normalized_inverse_class_weights(tensors_list):
    # Flatten the matrices and count the occurrences of each class
    flat_data = np.concatenate([tensor.flatten() for tensor in tensors_list])
    class_counts = np.bincount(flat_data.astype(int))

    # Add missing classes with count 0
    num_classes = max(len(class_counts), 1)
    class_frequencies = np.zeros(num_classes)
    class_frequencies[:len(class_counts)] = class_counts

    # Compute class frequencies and then inverse class weights
    total_samples = np.sum(class_frequencies)
    inverse_class_weights = np.zeros_like(class_frequencies)
    non_zero_indices = class_frequencies != 0
    inverse_class_weights[non_zero_indices] = total_samples / class_frequencies[non_zero_indices] if total_samples > 0 else class_frequencies[non_zero_indices]

    # Normalize inverse class weights to sum up to 1 (excluding weights with value 0)
    non_zero_indices = inverse_class_weights != 0
    normalized_inverse_weights = np.zeros_like(inverse_class_weights)
    normalized_inverse_weights[non_zero_indices] = inverse_class_weights[non_zero_indices] / np.sum(inverse_class_weights[non_zero_indices]) if np.sum(inverse_class_weights[non_zero_indices]) > 0 else inverse_class_weights[non_zero_indices]

    return normalized_inverse_weights
