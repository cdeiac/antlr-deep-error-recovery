package antlr.extensions;

import org.antlr.v4.runtime.DefaultErrorStrategy;
import org.antlr.v4.runtime.Parser;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.misc.IntervalSet;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class CustomErrorStrategy extends DefaultErrorStrategy {

    private List<Integer> skippedTokenIndexes = new ArrayList<>();


    @Override
    public Token recoverInline(Parser recognizer) {
        Token tokenBefore = recognizer.getCurrentToken();
        Token tokenAfter = super.recoverInline(recognizer);
        skippedTokenIndexes.addAll(
                IntStream.range(tokenBefore.getTokenIndex(), tokenAfter.getTokenIndex()+1)
                        .boxed()
                        .toList()
        );
        return tokenAfter;
    }

    @Override
    public void recover(Parser recognizer, RecognitionException e) {
        Token tokenBefore = recognizer.getCurrentToken();
        super.recover(recognizer, e);
        Token tokenAfter = recognizer.getCurrentToken();
        skippedTokenIndexes.addAll(
                IntStream.range(tokenBefore.getTokenIndex(), tokenAfter.getTokenIndex()+1)
                        .boxed()
                        .toList()
        );
    }

    protected boolean singleTokenInsertion(Parser recognizer) {
        boolean insertionSuccessful = super.singleTokenInsertion(recognizer);
        return insertionSuccessful;
    }
    @Override
    protected Token singleTokenDeletion(Parser recognizer) {
        Token deletedToken = super.singleTokenDeletion(recognizer);
        if (deletedToken != null) {
            //System.out.println("Delete: " + recognizer.getCurrentToken().getText());
        }
        return deletedToken;
    }
    @Override
    protected void consumeUntil(Parser recognizer, IntervalSet set) {
        //System.out.println("Consume: " + recognizer.getCurrentToken().getText());
        super.consumeUntil(recognizer, set);
    }

    @Override
    protected Token getMissingSymbol(Parser recognizer) {
        Token missingToken = super.getMissingSymbol(recognizer);
        return missingToken;
    }

    public List<Integer> getSkippedTokenIndexes() {
        return skippedTokenIndexes.stream().distinct().collect(Collectors.toList());
    }
}
