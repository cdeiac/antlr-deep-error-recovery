import antlr.JavaLexer;
import antlr.JavaParser;
import antlr.extensions.CustomVisitor;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.Vocabulary;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class LexerTest {

    @Test
    public void identifier() {
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(""));
        Vocabulary vocabulary = javaLexer.getVocabulary();
        int maxTokenType = vocabulary.getMaxTokenType();
        int randomNumber = new Random().ints(1, maxTokenType).findFirst().getAsInt();
        String randomToken = vocabulary.getSymbolicName(randomNumber);
        List<? extends Token> tokens = javaLexer.getAllTokens();
        Map<String, Integer> tokenMap = javaLexer.getTokenTypeMap();
        assertFalse(tokenMap.isEmpty());
    }

    @Test
    public void test_with_single_token() {
        String javaClassContent = "int";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        assertNotNull(tokens);
    }

    @Test
    public void test_get_token_by_name() {
        String javaClassContent = "String IDENTIFIER = STRING_LITERAL";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        String strings = javaLexer.getTokenNames()[5];
        assertNotNull(strings);
    }
}
