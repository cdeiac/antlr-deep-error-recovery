package antlr.utils;

import org.apache.commons.math3.util.Precision;

public class TokenUtils {

    public static double computeSuccessfulCompilationPercentage(int skippedTokenIndexes, int totalTokens) {
        if (totalTokens == 0) {
            return 0.0000;
        }
        else if (skippedTokenIndexes == 0) {
            return 1.0000;
        }
        else {
            return Precision.round((1.0 - ((double) skippedTokenIndexes / (double) (totalTokens))),4 );
        }
    }

    public static double computeSequenceEquality(int[] inputSequence,
                                                 int[] targetSequence) {
        int count = 0;
        int totalElements = targetSequence.length;

        try {
            for (int i = 0; i < targetSequence.length; i++) {
                if (inputSequence[i] == targetSequence[i]) {
                    count++;
                }
            }
        } catch (Exception e) {
            e.printStackTrace(); // TODO: Check why base reconstruction with bail and ER are different!
        }
        return Precision.round((double) count / totalElements, 4);
    }

    public static double computeSuccessfulCompilationPercentageWithBail(int errorIndex,
                                                                        int totalNodes) {
        if (errorIndex == 0) {
            return 0.0000;
        }
        if (errorIndex == totalNodes) {
            return 1.0000;
        }
        // error position index starts at 0, so we need to add 1
        return Precision.round((double) errorIndex / (double) (totalNodes), 4);
    }

    public static double computeSequenceEqualityWithBail(int bailIndex,
                                                         int[] inputSequence,
                                                         int[] targetSequence) {

        // pre-processing
        int[] bailInput = getFirstNElements(inputSequence, bailIndex);

        int count = 0;
        int totalElements = targetSequence.length;

        try { // TODO: How is bail index larger than input sequence???
            for (int i = 0; i < Math.min(bailInput.length, targetSequence.length); i++) {
                if (bailInput[i] == targetSequence[i]) {
                    count++;
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            e.printStackTrace();
        }

        return Precision.round((double) count / totalElements, 4);
    }

    public static double computeSequenceEqualityForModel(int[] inputSequence, int[] targetSequence) {
        int count = 0;
        int totalElements = 0;

        for (int i = 0; i < inputSequence.length; i++) {
            if (inputSequence[i] == 0 && targetSequence[i] == 0) {
                continue;
            }
            if (targetSequence[i] == 1) {
                // stop at EOS
                break;
            }
            totalElements += 1;
            if (inputSequence[i] == targetSequence[i]) { // input is shifted
                count++;
            }
        }
        return Precision.round((double) count / totalElements, 4);
    }

    public static double computeSequenceEqualityForModelWithBail(int bailIndex,
                                                                 int[] inputSequence,
                                                                 int[] targetSequence) {

        // pre-processing
        int[] bailInput = getFirstNElements(inputSequence, bailIndex);

        int count = 0;
        int totalElements = targetSequence.length-2; // account for SOS, EOS

        for (int i = 0; i < bailInput.length; i++) {
            if (bailInput[i] == targetSequence[i+1]) { // input is shifted
                count++;
            }
        }
        return Precision.round((double) count / totalElements, 4);
    }

    public static int[] getFirstNElements(int[] array, int n) {
        // Check if the array length is less than n, return the entire array
        if (n >= array.length) {
            return array.clone(); // Return a copy of the original array
        }

        // Create a new array of size n
        int[] result = new int[n];

        // Copy the first n elements from the original array
        System.arraycopy(array, 0, result, 0, n);

        return result;
    }
}
