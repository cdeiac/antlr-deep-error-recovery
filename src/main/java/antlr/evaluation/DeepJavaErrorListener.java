package antlr.evaluation;

import antlr.converters.ANTLRDataConverter;
import antlr.converters.ANTLRPlaceholderToken;
import org.antlr.v4.runtime.*;

import java.util.ArrayList;
import java.util.List;

public class JavaDeepErrorListener extends BaseErrorListener {

    private final List<Token> tokenList;
    private List<Token> reconstructedList = new ArrayList<>();
    private List<CompilationError> compilationErrorList = new ArrayList<>();
    private int netModifications = 0;


    public JavaDeepErrorListener(List<Token> tokenStream) {
        this.tokenList = tokenStream.stream()
                // filter out WS && EOF
                .filter(t -> t.getType() != 125 && t.getType() != -1)
                .toList();
        reconstructedList.addAll(tokenList);
    }


    @Override
    public void syntaxError(Recognizer<?, ?> recognizer,
                            Object offendingSymbol,
                            int line,
                            int charPositionInLine,
                            String msg,
                            RecognitionException e) {

        int actualTokenIndex = this.getErrorTokenPositionOfTokenStream((Token) offendingSymbol);
        this.compilationErrorList.add(new CompilationError(msg, actualTokenIndex));

        replaceTokenFromMessage(msg, actualTokenIndex);
    }

    public int getErrorTokenPositionOfTokenStream(Token currentToken) {
        return reconstructedList.indexOf(currentToken);
    }

    public int getTotalNumberOfTokens() {
        return this.tokenList.size();
    }

    public List<CompilationError> getCompilationErrorList() {
        return this.compilationErrorList;
    }

    public void replaceTokenFromMessage(String message, int tokenIndex) {
        String tokenText;
        if (message == null) {
            return;
        }
        if (message.contains("expecting")) {
            tokenText = replaceExpectedToken(message);
            replaceTokenTypeInStream(tokenText, tokenIndex);
        }
        else if (message.contains("missing")){
            tokenText = replaceMissingToken(message);
            insertTokenTypeInStream(tokenText, tokenIndex);
        }


    }

    public String replaceExpectedToken(String input) {
        // Find the index of "expecting"
        int index = input.indexOf("expecting");
        if (index != -1) {
            // Extract the substring after "expecting"
            String subString = input.substring(index + "expecting".length()).trim();
            // Remove surrounding curly braces
            subString = subString.substring(1, subString.length() - 1).trim();
            // Split the substring by commas and get the first element
            String[] elements = subString.split(",");
            String firstElement = elements[0].trim();
            // Remove single quotes around the element
            return firstElement.replaceAll("^'|'$", "");
        } else {
            return null;
        }
    }
    public String replaceMissingToken(String message) {
        String extractedTokenText = "";
        // Find the positions of the single quotes
        int startIndex = message.indexOf("'");
        int endIndex = message.indexOf("'", startIndex + 1);

        // Extract the substring between the single quotes
        if (startIndex != -1 && endIndex != -1) {
            extractedTokenText = message.substring(startIndex + 1, endIndex);
        }
        return extractedTokenText;

    }

    private void replaceTokenTypeInStream(String tokenName, int tokenIndex) {
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
        if (tokenIndex == -1) {
            reconstructedList.add(new CommonToken(replacementTokenId, tokenName));
            netModifications += 1;
        }
        else {
            // replace in tokenstream
            reconstructedList.set(tokenIndex, new CommonToken(replacementTokenId, tokenName));
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
        // add in tokenstream
        reconstructedList.add(tokenIndex + netModifications, new CommonToken(replacementTokenId, tokenName));
        netModifications += 1;
    }

    public int[] getReconstructedSequence() {
        return reconstructedList.stream()
                .map(Token::getType)
                .mapToInt(Integer::intValue)
                .toArray();
    }

}
