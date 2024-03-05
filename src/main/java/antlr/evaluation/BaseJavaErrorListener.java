package antlr.evaluation;

import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;
import org.antlr.v4.runtime.Token;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class BaseJavaErrorListener extends BaseErrorListener {

    private final List<Token> tokenList;
    private List<Token> reconstructedList = new ArrayList<>();
    private List<CompilationError> compilationErrorList = new ArrayList<>();


    public BaseJavaErrorListener(List<Token> tokenStream) {
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

        int actualTokenIndex = getErrorTokenPositionOfTokenStream((Token) offendingSymbol);
        this.compilationErrorList.add(new CompilationError(msg, actualTokenIndex, -1));
    }

    public int getErrorTokenPositionOfTokenStream(Token currentToken) {
        int match = IntStream.range(0, reconstructedList.size())
                .filter(i -> reconstructedList.get(i).getTokenIndex() == (currentToken.getTokenIndex()))
                .findFirst() // Find the first matching index
                .orElse(-1);
        if (match == -1) {
            // EOF matched
            return getErrorTokenPositionOfTokenStream(reconstructedList.get(reconstructedList.size()-1));
        }
        return match;
    }

    public int getTotalNumberOfTokens() {
        return this.tokenList.size();
    }

    public List<CompilationError> getCompilationErrorList() {
        return new ArrayList<>(this.compilationErrorList.stream().collect(Collectors.toMap(
                        CompilationError::getPosition, // Key mapper
                        Function.identity(), // Value mapper (identity function)
                        (existing, replacement) -> existing, // Merge function to keep existing objects
                        HashMap::new // Use LinkedHashMap to preserve the order
                ))
                .values());
    }
    public int[] getReconstructedSequence() {
        return reconstructedList.stream()
                .map(Token::getType)
                .mapToInt(Integer::intValue)
                .toArray();
    }

    public int findPreviousIndex(int positionIndex) {
        int match = IntStream.range(0, reconstructedList.size())
                .filter(i -> reconstructedList.get(i).getTokenIndex() > positionIndex)
                .findFirst() // Find the first matching index
                .orElse(-1);
        return match-1;
    }

}
