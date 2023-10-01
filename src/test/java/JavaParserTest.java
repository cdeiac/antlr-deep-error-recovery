import com.antlr.grammars.java.JavaLexer;
import com.antlr.grammars.java.JavaParser;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.tree.ParseTree;
import org.junit.jupiter.api.Test;


public class JavaParserTest {

    @Test
    public void method_name_starts_with_number_causes_error() {
        String javaClassContent = "public class SampleClass { void 1doSomething(){} }";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        JavaParser javaParser = new JavaParser(tokens);
        javaParser.setTrace(true);
        ParseTree parseTree = javaParser.compilationUnit();
    }
}