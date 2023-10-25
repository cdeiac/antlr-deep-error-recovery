package dataset.noise;

import dataset.tokenizer.Tokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

public class Insertion implements NoiseStrategy {

    private final static Logger logger = LoggerFactory.getLogger(Insertion.class);
    private final Tokenizer tokenizer;


    public Insertion(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }


    @Override
    public String[] apply(String token) {
        String[] generatedTokens = new String[]{tokenizer.getRandomToken(), token};
        logger.debug("Insert Token(s): {}", Arrays.toString(generatedTokens));
        return generatedTokens;
    }
}