package dataset.tokenizer;

import antlr.JavaLexer;
import org.antlr.v4.runtime.CharStreams;

public class JavaTokenizer extends Tokenizer {
    public JavaTokenizer() {
        super(new JavaLexer(CharStreams.fromString("")));
    }
}
