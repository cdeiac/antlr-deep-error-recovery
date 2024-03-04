import json

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def extract_average_scores(lc_dict):
    lc_dict = json.load(open(lc_001))
    fold0_scores = lc_dict['0']["test_accs"]
    fold1_scores = lc_dict['1']["test_accs"]
    fold2_scores = lc_dict['2']["test_accs"]
    return [(x + y + z) / 3 for x, y, z in zip(fold0_scores, fold1_scores, fold2_scores)]


if __name__ == '__main__':
    # noisy files
    lc_010 = "src/main/python/logs/00_010/scores.json"
    lc_005 = "src/main/python/logs/00_005/scores.json"
    lc_001 = "src/main/python/logs/00_001/scores.json"

    data_010 = extract_average_scores(lc_010)
    data_005 = extract_average_scores(lc_005)
    data_001 = extract_average_scores(lc_001)
    epochs = list(range(1, 11))

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=epochs, y=data_010, marker='o')
    plt.title('Learning Curves for Three Models')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.xticks(epochs)  # Ensure all epochs are shown
    plt.legend(title='Model')
    plt.show()
