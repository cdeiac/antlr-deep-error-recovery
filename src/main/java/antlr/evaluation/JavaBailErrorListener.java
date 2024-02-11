package antlr.evaluation;

import antlr.converters.ANTLRDataConverter;
import antlr.converters.ANTLRPlaceholderToken;
import org.antlr.v4.runtime.*;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class JavaBailErrorListener extends BaseErrorListener {

    private List<Token> tokenList = new ArrayList<>();
    private List<Token> reconstructedList;
    private List<CompilationError> compilationErrorList = new ArrayList<>();
    private int netModifications = 0;
    private boolean replaceOnError;
    private boolean bailed = false;


    public JavaBailErrorListener(List<Token> tokenStream, boolean replaceOnError) {
        this.tokenList.addAll(tokenStream);//.stream()
        // filter out WS && EOF
        //.filter(t -> t.getType() != 125 && t.getType() != -1)
        //.toList();
        this.replaceOnError = replaceOnError;
        this.reconstructedList = new ArrayList<>(tokenList).stream()
                // filter out WS && EOF
                .filter(t -> t.getType() != 125 && t.getType() != -1)
                .collect(Collectors.toList());

    }

    @Override
    public void syntaxError(Recognizer<?, ?> recognizer,
                            Object offendingSymbol,
                            int line,
                            int charPositionInLine,
                            String msg,
                            RecognitionException e) {

        if (bailed) {
            return;
        }
        //Optional<Token> matchedToken = reconstructedList.stream().filter(t -> t.getTokenIndex() == ((Token) offendingSymbol).getTokenIndex()+netModifications).findFirst();
        //int actualTokenIndex = matchedToken.map(this::getErrorTokenPositionOfTokenStream).orElse(-1);
        int actualTokenIndex = getErrorTokenPositionOfTokenStream((Token) offendingSymbol);
        this.compilationErrorList.add(new CompilationError(msg, actualTokenIndex));
        if (replaceOnError) {
            replaceTokenFromMessage(msg, actualTokenIndex);
        }
        bailed = true;
    }

    public int getErrorTokenPositionOfTokenStream(Token currentToken) {
        //Optional<Token> matchedToken =  reconstructedList.stream().filter(t -> t.getTokenIndex() == currentToken.getTokenIndex()+netModifications).findFirst();
        //return matchedToken.isPresent() ? reconstructedList.indexOf((Token) currentToken) : -1;
        //reconstructedList.indexOf(currentToken);
        if (currentToken.getText().equals("<EOF>")) {
            return getReconstructedSequence().length;
        }
        int matchedTokenIdx = IntStream.range(0, tokenList.size())
                .filter(i -> tokenList.get(i).equals(currentToken))
                .findFirst() // Find the first matching index
                .orElse(-1);
        if (matchedTokenIdx < 0) {
            matchedTokenIdx = IntStream.range(0, tokenList.size())
                    .filter(i -> tokenList.get(i).getCharPositionInLine() == (currentToken.getCharPositionInLine()))
                    .findFirst() // Find the first matching index
                    .orElse(-1); // EOF
        }
        return matchedTokenIdx;
    }

    public int getBailedPosition(Token token) {
        //Optional<Token> matchedToken =  reconstructedList.stream().filter(t -> t.getTokenIndex() == currentToken.getTokenIndex()+netModifications).findFirst();
        //return matchedToken.isPresent() ? reconstructedList.indexOf((Token) currentToken) : -1;
        //reconstructedList.indexOf(currentToken);
        int[] reconstructedSequence = getReconstructedSequence();
        if (token.getText().equals("<EOF>")) {
            return reconstructedSequence.length;
        }
        int indexToMatch = token.getTokenIndex();
        int match = -999;
        while (match < 0) {
            int finalIndexToMatch = indexToMatch;
            match = IntStream.range(0, reconstructedSequence.length)
                    .filter(i -> reconstructedList.get(i).getTokenIndex() == finalIndexToMatch)
                    .findFirst() // Find the first matching index
                    .orElse(-999);
            indexToMatch -= 1; //token might not exist anymore, so we match the last token
        }
        // TODO: might be EOF or missing symbol???
        return match;
    }

    public List<CompilationError> getCompilationErrorList() {
        return this.compilationErrorList;
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
        // how it was before:
        //tokenText = replaceExpectedToken(message);
        //replaceTokenTypeInStream(tokenText, tokenIndex);

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

    private void deleteTypeInStream(int tokenIndex) {
        if (tokenIndex == -1) {
            // EOF: nothing to do
            return;
        }
        tokenList.remove(tokenIndex); // + netModification
        netModifications -= 1;/*
        for (int i = tokenIndex + netModifications+1; i < tokenList.size(); i++) {
            Token nextToken = tokenList.get(i);
            CommonToken newToken = new CommonToken(nextToken);
            newToken.setTokenIndex(nextToken.getTokenIndex()-1);
            newToken.setText(nextToken.getText());
            tokenList.set(i, newToken);
        }
       */
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
            tokenList.add(new CommonToken(replacementTokenId, tokenName));
            netModifications += 1;
        }
        else {
            // replace in tokenstream
            CommonToken replacement = new CommonToken(tokenList.get(tokenIndex));
            replacement.setText(tokenName);
            replacement.setType(replacementTokenId);
            tokenList.set(tokenIndex, replacement);
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
        // TODO: IndexOutOfBound!!
        CommonToken newToken = new CommonToken(tokenList.get(tokenIndex));// + netModifications));
        newToken.setText(tokenName);
        newToken.setType(replacementTokenId);
        tokenList.add(tokenIndex, newToken); //+ netModifications, newToken);
        netModifications += 1;
        /*
        for (int i = tokenIndex; i < reconstructedList.size(); i++) {
            Token nextToken = reconstructedList.get(i);
            CommonToken nextTokenWithText = new CommonToken(nextToken);
            nextTokenWithText.setTokenIndex(nextToken.getTokenIndex()+1);
            nextTokenWithText.setText(nextToken.getText());
            reconstructedList.set(i, nextTokenWithText);
        }*/

    }

    public int[] getReconstructedSequence() {
        /*
        return reconstructedList.stream()
                .map(Token::getType)
                .mapToInt(Integer::intValue)
                .toArray();

         */
        return tokenList.stream()
                // filter out WS && EOF
                .filter(t -> t.getType() != 125 && t.getType() != -1)
                .map(Token::getType)
                .mapToInt(Integer::intValue)
                .toArray();
    }

}
