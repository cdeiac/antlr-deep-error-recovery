package antlr.evaluation;

import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;
import org.antlr.v4.runtime.Token;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class JavaBaseErrorListener extends BaseErrorListener {

    private final List<Token> tokenList;
    private List<Token> reconstructedList = new ArrayList<>();
    private List<CompilationError> compilationErrorList = new ArrayList<>();


    public JavaBaseErrorListener(List<Token> tokenStream) {
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
        this.compilationErrorList.add(new CompilationError(msg, actualTokenIndex));
    }

    public int getErrorTokenPositionOfTokenStream(Token currentToken) {
        return IntStream.range(0, reconstructedList.size())
                .filter(i -> reconstructedList.get(i).getTokenIndex() == (currentToken.getTokenIndex()))
                .findFirst() // Find the first matching index
                .orElse(-1);
    }

    public int getTotalNumberOfTokens() {
        return this.tokenList.size();
    }

    public List<CompilationError> getCompilationErrorList() {
        return this.compilationErrorList;
    }

    public int[] getReconstructedSequence() {
        return reconstructedList.stream()
                .map(Token::getType)
                .mapToInt(Integer::intValue)
                .toArray();
    }

}
