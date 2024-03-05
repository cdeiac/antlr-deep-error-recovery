package antlr.evaluation;

import antlr.converters.ANTLRDataConverter;
import antlr.converters.ANTLRPlaceholderToken;
import org.antlr.v4.runtime.*;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static antlr.converters.ANTLRDataConverter.ANTLR_INDEX_2_TOKEN;

public class JavaErrorListener extends BaseErrorListener {

    private List<Token> tokenList;
    private List<Token> reconstructedList;
    private int[] original;
    private List<CompilationError> compilationErrorList = new ArrayList<>();
    private List<RecoveryOperation> operations = new ArrayList<>();


    private HashMap<Integer, Token> deletions = new HashMap<>();
    private HashMap<Integer, Token> additions = new HashMap<>();
    private HashMap<Integer, Token> modifications = new HashMap<>();


    public JavaErrorListener(List<Token> tokenStream, int[] original) {
        this.tokenList = tokenStream.stream()
                .filter(t -> t.getType() != 125 && t.getType() != -1)
                .toList();
        reconstructedList = new ArrayList<>(tokenList);
        this.original = original;
    }


    @Override
    public void syntaxError(Recognizer<?, ?> recognizer,
                            Object offendingSymbol,
                            int line,
                            int charPositionInLine,
                            String msg,
                            RecognitionException e) {

        int actualTokenIndex = getErrorTokenPositionOfTokenStream((Token) offendingSymbol);
        int adjustedPosition = getAdjustedPosition(actualTokenIndex);
        if (!isSpuriousError(actualTokenIndex)) {
            this.compilationErrorList.add(new CompilationError(msg, actualTokenIndex, adjustedPosition));
            replaceTokenFromMessage(msg, actualTokenIndex);
        }
    }

    public int getErrorTokenPositionOfTokenStream(Token currentToken) {
        if (currentToken.getType() == -1) {
            return -1;
        }
        return IntStream.range(0, reconstructedList.size())
                .filter(i -> reconstructedList.get(i).getTokenIndex() == (currentToken.getTokenIndex()))
                .findFirst() // Find the first matching index
                .orElse(-1);
    }

    private boolean isSpuriousError(int position) {
        return !this.compilationErrorList.stream()
                .filter(e -> e.getPosition() == position)
                .toList()
                .isEmpty();
    }

    public List<CompilationError> getCompilationErrorList() {
        return new ArrayList<>(this.compilationErrorList.stream()
                .collect(Collectors.toMap(
                        CompilationError::getPosition,
                        Function.identity(),
                        (existing, replacement) -> existing))
                .values());
    }

    public void replaceTokenFromMessage(String message, int tokenIndex) {
        String tokenText;
        if (message == null) {
            return;
        }
        if (message.toLowerCase().contains("extraneous")) {
            deleteTypeInStream(tokenIndex);
        }
        else if (message.contains("missing")) {
            tokenText = replaceMissingToken(message);
            insertTokenTypeInStream(tokenText, tokenIndex);
        } else if (message.contains("mismatched")) {
            tokenText = replaceExpectedToken(message);
            replaceTokenTypeInStream(tokenText, tokenIndex);
        }
    }

    public String replaceExpectedToken(String input) {
        int index = input.indexOf("expecting");
        if (index != -1) {
            String subString = input.substring(index + "expecting".length()).trim();
            subString = subString.substring(1, subString.length() - 1).trim();
            String[] elements = subString.split(",");
            String firstElement = elements[0].trim();
            return firstElement.replaceAll("^'|'$", "");
        } else {
            return null;
        }
    }
    public String replaceMissingToken(String message) {
        String extractedTokenText = "";
        int startIndex = message.indexOf("'");
        int endIndex = message.indexOf("'", startIndex + 1);
        if (startIndex != -1 && endIndex != -1) {
            extractedTokenText = message.substring(startIndex + 1, endIndex);
        }
        return extractedTokenText;

    }

    private void deleteTypeInStream(int tokenIndex) {
        if (tokenIndex == -1) {
            // EOF: nothing to do
            return;
        }
        deletions.put(tokenIndex, null);
    }


    private void replaceTokenTypeInStream(String tokenName, int tokenIndex) {
        int replacementTokenId;
        if (tokenName == null || tokenName.isEmpty()) {
            replacementTokenId = -1;
        }
        else {
            try {
                // get tokenId of given token name
                replacementTokenId = ANTLRDataConverter.mapTokenToIds(tokenName.toUpperCase())[0];
            } catch (Exception e) {
                replacementTokenId = ANTLRPlaceholderToken.reversePlaceholders.get(tokenName);
            }
        }
        if (tokenIndex == -1) {
            CommonToken newToken = new CommonToken(tokenList.get(tokenList.size()-1));
            newToken.setText(tokenName);
            newToken.setType(replacementTokenId);
            newToken.setCharPositionInLine(newToken.getCharPositionInLine()+1);
            //tokenList.add(new CommonToken(replacementTokenId, tokenName));
            //netModifications += 1;
            modifications.put(tokenList.size()-1, newToken);
        }
        else {
            // replace in tokenstream
            CommonToken replacement = new CommonToken(reconstructedList.get(tokenIndex));
            replacement.setText(tokenName);
            replacement.setType(replacementTokenId);
            //reconstructedList.set(tokenIndex, replacement);
            modifications.put(tokenIndex, replacement);
        }
    }

    private void insertTokenTypeInStream(String tokenName, int tokenIndex) {
        int replacementTokenId;
        if (tokenName.isEmpty()) {
            replacementTokenId = -1;
        }
        else {
            try {
                // get tokenId of given token name
                replacementTokenId = ANTLRDataConverter.mapTokenToIds(tokenName.toUpperCase())[0];
            } catch (Exception e) {
                replacementTokenId = ANTLRPlaceholderToken.reversePlaceholders.get(tokenName);
            }

        }
        CommonToken newToken;
        if (tokenIndex == -1) {
            newToken = new CommonToken(replacementTokenId);
            tokenIndex = 99999;
            newToken.setTokenIndex(99999);
        }
        // add in tokenstream
        else {
            newToken = new CommonToken(reconstructedList.get(tokenIndex));// + netModifications));
        }
        newToken.setText(tokenName);
        newToken.setType(replacementTokenId);
        additions.put(tokenIndex, newToken);
    }

    public int[] getReconstructedSequence() {
        for (Integer index : modifications.keySet()) {
            Token tokenBefore = reconstructedList.get(index);
            reconstructedList.set(index, modifications.get(index));
            int adjustedPosition = getAdjustedPosition(index);
            operations.add(
                    new RecoveryOperation(
                            "MOD",
                            index,
                            adjustedPosition,
                            ANTLR_INDEX_2_TOKEN.get(tokenBefore.getType()),
                            ANTLR_INDEX_2_TOKEN.get(modifications.get(index).getType()), // get new token
                            adjustedPosition < original.length ? ANTLR_INDEX_2_TOKEN.get(original[adjustedPosition]) : "EOF"
                    )
            );
        }
        for (Integer index : deletions.keySet()) {
            Token tokenBefore = reconstructedList.get(index);
            reconstructedList.set(index, deletions.get(index));
            int adjustedPosition = getAdjustedPosition(index);
            operations.add(
                    new RecoveryOperation(
                            "DEL",
                            index,
                            adjustedPosition,
                            ANTLR_INDEX_2_TOKEN.get(tokenBefore.getType()),
                            index+1 < reconstructedList.size() && reconstructedList.get(index+1) != null ? ANTLR_INDEX_2_TOKEN.get(reconstructedList.get(index+1).getType()) : "EOF", // ANTLR_INDEX_2_TOKEN.get(getNextNonWhitespaceToken(adjustedPosition)),
                            adjustedPosition < original.length ? ANTLR_INDEX_2_TOKEN.get(original[adjustedPosition]) : "EOF"
                    )
            );
        }
        TreeMap<Integer, Token> reverseAdditions = new TreeMap<>(Collections.reverseOrder());
        reverseAdditions.putAll(additions);

        for (Integer index : reverseAdditions.keySet()) {
            Token tokenBefore;
            if (index >= reconstructedList.size() || index == -1) {
                tokenBefore = new CommonToken(-1);
                reconstructedList.add(reverseAdditions.get(index));
            }
            else {
                tokenBefore = reconstructedList.get(index);
                reconstructedList.add(index, reverseAdditions.get(index));
            }
            int adjustedPosition = getAdjustedPosition(index);
            operations.add(
                    new RecoveryOperation(
                            "INS",
                            index,
                            adjustedPosition,
                            ANTLR_INDEX_2_TOKEN.get(tokenBefore.getType() > 0 ? tokenBefore.getType() : "EOF"),
                            ANTLR_INDEX_2_TOKEN.get(additions.get(index).getType()),
                            adjustedPosition < original.length ? ANTLR_INDEX_2_TOKEN.get(original[adjustedPosition]) : "EOF"
                    )
            );
        }

        return reconstructedList.stream()
                // filter out WS && EOF
                .filter(t -> t != null && t.getType() != 125 && t.getType() != -1)
                .map(Token::getType)
                .mapToInt(Integer::intValue)
                .toArray();
    }

    private int getAdjustedPosition(int position) {
        if (position >= reconstructedList.size()) {
            return reconstructedList.size();
        }
        int deletionsCount = 0;
        int additionsCount = 0;
        for (Integer key : deletions.keySet()) {
            if (key < position) {
                deletionsCount++;
            }
        }
        for (Integer key : additions.keySet()) {
            if (key < position) {
                additionsCount++;
            }
        }
        return position - deletionsCount + additionsCount;
    }

    public List<RecoveryOperation> getOperations() {
        return operations;
    }
}
