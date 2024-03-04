import numpy as np
from matplotlib import pyplot as plt

from Runner import Runner


def init_countmap_of_tensors(tensor_list):
    # Flatten the list of arrays and concatenate them into a single NumPy array
    flattened = [tensor.flatten().numpy() for tensor in tensor_list]
    all_values = np.concatenate(flattened)
    # Use numpy's unique function to get unique values and their counts
    unique_values, counts = np.unique(all_values, return_counts=True)
    return {unique_value: 0 for unique_value in unique_values}
    # Create a dictionary with unique values as keys and their counts as values
    #return dict(zip(unique_values, counts))


def add_counts_exceeds_max_count(tensor, count_map, max_count):
    temp_map = count_map.copy()
    for tensor_value in tensor:
        val = tensor_value.item()
        temp_map[val] = temp_map[val] + 1
    return max(temp_map.values()) > max_count


def add_counts_to_map(tensor, count_map):
    for tensor_value in tensor:
        val = tensor_value.item()
        count_map[val] = count_map[val] + 1
    return count_map


def create_balanced_dataset(source_list, target_list, max_elements, max_count):
    count_map = init_countmap_of_tensors(target_list)
    selected_source_samples = []
    selected_target_samples = []
    for i in range(len(target_list)):
        if len(selected_target_samples) == max_elements:
            break
        if not add_counts_exceeds_max_count(target_list[i], count_map, max_count):
            selected_target_samples.append(target_list[i])
            count_map = add_counts_to_map(target_list[i], count_map)
            selected_source_samples.append(source_list[i])
            if max(count_map.values()) >= max_count:
                break
    return selected_source_samples, selected_target_samples


def resample(src, tgt, identifier):
    map = dict()
    for i in range(len(tgt)):
        for j in tgt[i]:
            val = j.item()
            if val == identifier:
                if i not in map:
                    map[i] = 0
                else:
                    map[i] += (1/len(tgt[i]))
    sorted_items = sorted(map.items(), key=lambda x: x[1])
    result_keys = [key for key, value in sorted_items[:10]]

    #src, targ = create_balanced_dataset(train_seqs, train_target_seqs, 10, 200)
    #plot_tensor_histogram(targ)
    src = [src[x] for x in result_keys]
    tgt = [tgt[x] for x in result_keys]

    temp_x = []
    temp_y = []
    for i in range(len(src)):
        if len(src[i]) < 2000:
            temp_x.append(src[i])
            temp_y.append(tgt[i])
    src = temp_x
    tgt = temp_y

    return src, tgt


def plot_tensor_histogram(tensor_list):
    flattened = [tensor.flatten().numpy() for tensor in tensor_list]
    all_values = np.concatenate(flattened)
    unique_values, counts = np.unique(all_values, return_counts=True)

    # Plot histogram
    plt.hist(flattened, bins=len(unique_values))
    plt.xlabel('Unique Values')
    plt.ylabel('Occurrences')
    plt.title('Histogram of Unique Values')
    plt.show()


if __name__ == '__main__':

    runner = Runner()
    runner.train_model()

