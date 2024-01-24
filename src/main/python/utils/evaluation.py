from typing import List
from torch import tensor
from dataset.token import Token


def compute_accuracy(predicted, target):
    total_non_padded = 0.0
    correct = 0.0
    #target = target.tolist()
    #print(f"Targ: {target.tolist()}")
    #print(f"Pred: {predicted}")
    for i in range(len(target)): # rest is padding which we ignore
        if target[i] != Token.PAD:
            total_non_padded += 1
            if predicted[i] == target[i]:
                correct += 1
    return correct/total_non_padded


def aggregated_accuracy(predictions: List[tensor], target: tensor, stride: int) -> float:
    seq_len = len(predictions)
    #stride = max(1, stride-1)
    predicted_sequence = []
    for i, prediction in enumerate(predictions):
        scalars = prediction.flatten().tolist()
        #if i == seq_len - 1:
            #predicted_sequence.extend(scalars)
        #    predicted_sequence.extend(scalars[:stride])
        #else:
        #    if stride == len(prediction):
                # stride is full window length: ignore EOS prediction for windows that are not the last one
        #        predicted_sequence.extend(scalars)
        #    else:
        predicted_sequence.extend(scalars[:stride])
    return compute_accuracy(predicted_sequence, target)
