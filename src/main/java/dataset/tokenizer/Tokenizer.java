package dataset.tokenizer;

import org.antlr.v4.runtime.Lexer;
import java.util.Random;

public class Tokenizer {

    private final Lexer lexer;
    private final Random random = new Random();
    private final int numberOfTokens;


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
        int randomNumber = this.random.ints(1, numberOfTokens + 1)
                .findFirst()
                .getAsInt();
        // workaround for 'token'
        String token = this.lexer.getVocabulary().getLiteralName(randomNumber); // getDisplayName(randomNumber);
        if (token != null) {
            token = token.replace("'", "");
        }
        return token;
    }
}
