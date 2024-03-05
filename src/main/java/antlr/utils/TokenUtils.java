package antlr.utils;

import antlr.JavaLexer;
import antlr.converters.ANTLRDataConverter;
import antlr.converters.ANTLRPlaceholderToken;
import antlr.errorrecovery.ErrorPosition;
import antlr.errorrecovery.ErrorRecovery;
import antlr.evaluation.CompilationError;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.Token;
import org.apache.commons.math3.util.Precision;

import java.util.ArrayList;
import java.util.List;

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
        for (int i = 0; i < targetSequence.length; i++) {
            if (i == inputSequence.length) {
                break;
            }
            if (inputSequence[i] == targetSequence[i]) {
                count++;
            }
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

        for (int i = 0; i < Math.min(bailInput.length, targetSequence.length); i++) {
            if (i == inputSequence.length) {
                break;
            }
            if (bailInput[i] == targetSequence[i]) {
                count++;
            }
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

    public static List<ErrorPosition> getErrorPositions(int[] sequence, List<CompilationError> compilationErrors) {
        List<Token> tokenList = getAntlrTokenList(sequence);
        int length = tokenList.size();
        List<ErrorPosition> errorPositions = new ArrayList<>();
        for (CompilationError error : compilationErrors) {
            errorPositions.add(new ErrorPosition(Precision.round((double) error.getPosition() / length, 4)));
        }
        return errorPositions;
    }

    public static List<ErrorRecovery> getErrorRecoveries(int[] input, int[] modification, int[] target, List<CompilationError> errorList) {

        List<Token> inputTokens = getAntlrTokenList(input);
        List<Token> modificationTokens = getAntlrTokenList(modification);
        List<Token> targetTokens = getAntlrTokenList(target);
        List<ErrorRecovery> recoveryList = new ArrayList<>();
        for (CompilationError error : errorList) {
            int errorPosition = error.getPosition();
            int inputToken = errorPosition < inputTokens.size() ? inputTokens.get(errorPosition).getType() : -1;
            int modificationToken = errorPosition < modificationTokens.size() ? modificationTokens.get(errorPosition).getType() : -1;
            int targetToken = errorPosition < targetTokens.size() ? targetTokens.get(errorPosition).getType() : -1;
            if (targetToken == 125) {
                targetToken = errorPosition+1 < targetTokens.size() ? targetTokens.get(errorPosition+1).getType() : -1;
            }
            recoveryList.add(new ErrorRecovery(
                    toAntlrToken(inputToken), toAntlrToken(modificationToken), toAntlrToken(targetToken))
            );
        }
        return recoveryList;
    }

    public static String toAntlrToken(int type) {
        if (type == -1) {
            return "EOF";
        }
        return ANTLRDataConverter.mapIdsToToken(new int[] {type});
    }

    public static List<Token> getAntlrTokenList(int[] tokenIds) {
        JavaLexer lLexer = new JavaLexer(CharStreams.fromString(ANTLRPlaceholderToken.replaceSourceWithDummyTokens(tokenIds)));
        CommonTokenStream tokens = new CommonTokenStream(lLexer);
        tokens.fill();
        return tokens.getTokens();
    }

    public static int[] getCleanSequence(List<Token> tokenList) {
        return tokenList.stream()
                // filter out WS && EOF
                .filter(t -> t.getType() != 125 && t.getType() != -1)
                .map(Token::getType)
                .mapToInt(Integer::intValue)
                .toArray();
    }
}
