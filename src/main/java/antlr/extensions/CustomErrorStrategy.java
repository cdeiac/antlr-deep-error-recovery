package antlr.extensions;

import org.antlr.v4.runtime.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CustomErrorStrategy extends DefaultErrorStrategy {

    private static final Logger LOGGER = LoggerFactory.getLogger(CustomErrorStrategy.class);


    @Override
    public Token recoverInline(Parser recognizer) {

        LOGGER.info("Recover Inline");
        return super.recoverInline(recognizer);
    }

    @Override
    public void recover(Parser recognizer, RecognitionException e) {

        LOGGER.info("Recover");
        super.recover(recognizer, e);
    }

}
