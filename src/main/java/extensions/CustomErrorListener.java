package extensions;

import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.ATNConfigSet;
import org.antlr.v4.runtime.dfa.DFA;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.BitSet;

import static extensions.TokenGenerator.getRandomString;

public class CustomErrorListener extends BaseErrorListener {

    private static final Logger logger = LoggerFactory.getLogger(CustomErrorListener.class);
    private final TokenStreamRewriter rewriter;


    public CustomErrorListener(CommonTokenStream tokens) {
        rewriter = new TokenStreamRewriter(tokens);
    }


    @Override
    public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol, int line, int charPositionInLine, String msg, RecognitionException e) {
        logger.info("Offending symbol: {} at {}:{} | message: {} | exception:{}",
                offendingSymbol.toString(), line, charPositionInLine, msg, e != null ? e.getMessage(): "");
        this.replaceToken((Parser) recognizer,1);
    }

    @Override
    public void reportAmbiguity(Parser recognizer, DFA dfa, int startIndex, int stopIndex, boolean exact, BitSet ambigAlts, ATNConfigSet configs) {
        logger.info("Ambiguity detected at index: {}-{} | exact: {} | alternatives: {} | config: {}",
                startIndex, stopIndex, exact, ambigAlts.toString(), configs);
        logger.info(recognizer.getTokenStream().getText());
        this.replaceToken(recognizer,stopIndex-startIndex);
    }

    @Override
    public void reportAttemptingFullContext(Parser recognizer, DFA dfa, int startIndex, int stopIndex, BitSet conflictingAlts, ATNConfigSet configs) {
        logger.info("Attempting full context at index: {}-{} | conflicting alternatives: {} | config: {}",
                startIndex, stopIndex, conflictingAlts.toString(), configs);
        this.replaceToken(recognizer, stopIndex-startIndex);
    }

    @Override
    public void reportContextSensitivity(Parser recognizer, DFA dfa, int startIndex, int stopIndex, int prediction, ATNConfigSet configs) {
        logger.info("Context sensitivity detected at index: {}-{} | prediction: {} | config: {}",
                startIndex, stopIndex, prediction, configs);
        this.replaceToken(recognizer,stopIndex-startIndex);
    }

    private void replaceToken(Parser recognizer, int length) {

        String replacement = getRandomString(length);

        if (length == 0) {
            // do nothing
            logger.info("Skip replace");
        }
        else if (length == 1) {
            logger.info("Replace '{}' with '{}'", recognizer.getCurrentToken().getText(), replacement);
            rewriter.replace(recognizer.getContext().start, replacement);
        }
        else {
            logger.info("Replace '{}' with '{}'", recognizer.getCurrentToken().getText(), replacement);
            rewriter.replace(recognizer.getContext().start, recognizer.getContext().stop, getRandomString(length));
        }
    }

    public String getReplacedCode() {
        return rewriter.getText();
    }
}
