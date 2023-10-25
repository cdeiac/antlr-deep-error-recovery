package dataset.noise;

import dataset.tokenizer.Tokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Modification implements NoiseStrategy {

    private final static Logger logger = LoggerFactory.getLogger(Insertion.class);
    private final Tokenizer tokenizer;


    public Modification(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }


    @Override
    public String[] apply(String token) {
        String[] generatedToken = new String[]{tokenizer.getRandomToken()};
        logger.debug("Replace Token {} with {}", token, generatedToken);
        return generatedToken;
    }
}