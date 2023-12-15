package dataset.tokenizer;

import dataset.noise.NoiseGenerator;
import org.antlr.v4.runtime.Lexer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

public class Tokenizer {

    private final Lexer lexer;
    private final Random random = new Random();
    private final int numberOfTokens;
    private final static Logger logger = LoggerFactory.getLogger(NoiseGenerator.class);


    public Tokenizer(Lexer lexer) {
        this.lexer = lexer;
        this.numberOfTokens = lexer.getVocabulary().getMaxTokenType();
    }


    public String getToken(int tokenNumber) {
        return lexer.getVocabulary().getDisplayName(tokenNumber);
    }

    /**
     *  Retrieve a random token from the lexer's vocabulary
     *  @return random token
     */
    public String getRandomToken() {
        int randomToken = -1;
        String token = null;
        while (true) {
            if (randomToken == -1 || randomToken == 125) {
                randomToken = this.random.ints(1, numberOfTokens + 1)
                        .findFirst()
                        .getAsInt();
                // workaround for 'token'
                token = this.lexer.getVocabulary().getSymbolicName(randomToken); //getLiteralName(randomNumber); // getDisplayName(randomNumber);
                if (token != null) {
                    token = token.replace("'", "");
                }
            }
            else {
                logger.debug("RANDOM TOKEN: {}", token);
                break;
            };
        }
        return token;
    }
}
