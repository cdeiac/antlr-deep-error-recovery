import antlr.evaluation.JavaProgramEvaluator;
import antlr.utils.TestData;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertNotNull;

public class JavaProgramEvaluatorTest {

    JavaProgramEvaluator evaluator = new JavaProgramEvaluator("00_1", 0);


    @Test
    void testPipeline() {
        String noisy = "SOS CLASS IDENTIFIER LBRACE PUBLIC INT IDENTIFIER LPAREN INT LBRACK RBRACK IDENTIFIER COMMA " +
                "INT IDENTIFIER RPAREN LBRACE CONST ASSIGN DECIMAL_LITERAL SEMI FOR LPAREN SEMI IDENTIFIER LT IDENTIFIER DOT " +
                "IDENTIFIER SEMI IDENTIFIER INC RPAREN LBRACE IF LPAREN IDENTIFIER LBRACK IDENTIFIER RBRACK GE IDENTIFIER " +
                "RPAREN LPAREN BREAK SEMI RBRACE RBRACE RETURN IDENTIFIER SEMI RBRACE RBRACE EOS";
        String original = "SOS CLASS IDENTIFIER LBRACE PUBLIC INT IDENTIFIER LPAREN INT LBRACK RBRACK IDENTIFIER COMMA " +
                "INT IDENTIFIER RPAREN LBRACE INT IDENTIFIER ASSIGN DECIMAL_LITERAL SEMI FOR LPAREN SEMI IDENTIFIER LT" +
                " IDENTIFIER DOT IDENTIFIER SEMI IDENTIFIER INC RPAREN LBRACE IF LPAREN IDENTIFIER LBRACK IDENTIFIER " +
                "RBRACK GE IDENTIFIER RPAREN LBRACE BREAK SEMI RBRACE RBRACE RETURN IDENTIFIER SEMI RBRACE RBRACE EOS";
        // Act
        evaluator.parseAndEvaluateAllOn("test", List.of(noisy), List.of(original));
        // Assert
        assertNotNull(noisy);

    }

    private TestData buildTestData(String noisy, String original) {
        TestData td = new TestData();
        td.setOriginalData(original);
        td.setNoisyData(noisy);
        return td;
    }
}
