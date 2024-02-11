import antlr.JavaLexer;
import antlr.JavaParser;
import antlr.converters.ANTLRDataConverter;
import antlr.converters.ANTLRPlaceholderToken;
import org.antlr.v4.runtime.*;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

 @Disabled
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
        String javaClassContent = "IDENTIFIER DOT ASSIGN LT LPARAM";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenFactory factory = new CommonTokenFactory();
        javaLexer.setTokenFactory(factory);

        //CommonTokenStream tokenStream = new CommonTokenStream(new CommonToken(1).getInputStream());
        //Token newToken = javaLexer.getTokenFactory().create(1, "ABSTRACT");
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        var lexedTokens = javaLexer.getAllTokens();

        JavaParser parser = new JavaParser(tokens);

        assertNotNull(tokens);
    }

     @Test
     public void test_with_encoded_ids() {
        String originalData = "LINE_COMMENT LINE_COMMENT LINE_COMMENT LINE_COMMENT LINE_COMMENT CLASS IDENTIFIER LBRACE PUBLIC INT IDENTIFIER LPAREN INT IDENTIFIER RPAREN LBRACE WHILE LPAREN IDENTIFIER GE DECIMAL_LITERAL RPAREN LBRACE INT IDENTIFIER ASSIGN DECIMAL_LITERAL SEMI WHILE LPAREN IDENTIFIER GT DECIMAL_LITERAL RPAREN LBRACE IDENTIFIER ADD_ASSIGN IDENTIFIER MOD DECIMAL_LITERAL SEMI IDENTIFIER DIV_ASSIGN DECIMAL_LITERAL SEMI RBRACE IDENTIFIER ASSIGN IDENTIFIER SEMI RBRACE RETURN IDENTIFIER SEMI RBRACE RBRACE";
        int[] input = ANTLRDataConverter.mapTokenToIds(originalData);
        //List<Integer> ints = List.of(35, 9, 128, 80, 81); // public class Dummy { }
        String dummyInput = ANTLRPlaceholderToken.replaceSourceWithDummyTokens(input);
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(dummyInput));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);

        JavaParser parser = new JavaParser(tokens);
        parser.compilationUnit();

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
