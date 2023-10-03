import com.antlr.grammars.java.JavaLexer;
import com.antlr.grammars.java.JavaParser;
import extensions.CustomErrorListener;
import extensions.EvaluateExpressionVisitor;
import extensions.TokenStreamListener;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.TokenStreamRewriter;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeWalker;
import org.junit.jupiter.api.Test;

import java.util.logging.Logger;


public class JavaParserTest {

    private static final Logger logger = Logger.getLogger(JavaParserTest.class.getName());

    @Test
    public void method_name_starts_with_number_causes_error() {
        String javaClassContent = "public class SampleClass { void 1doSomething(){} }";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        JavaParser javaParser = new JavaParser(tokens);
        javaParser.setTrace(true);

        TokenStreamListener listener = new TokenStreamListener(tokens);
        ParseTreeWalker.DEFAULT.walk(listener, javaParser.compilationUnit());
        //ParseTree parseTree = javaParser.compilationUnit();
        logger.info("Replaced code: " + listener.getReplacedCode());
        //javaLexer.setLine(0);
        //javaLexer.setCharPositionInLine(0);
        //javaLexer.reset();
        //javaParser.setInputStream(tokens);
        //javaParser.setTokenStream(tokens);
        //logger.info("index " + tokens.index());
        //javaParser.reset();
        //ParseTreeWalker.DEFAULT.walk(listener, javaParser.compilationUnit());
    }

    @Test
    public void replace_input_identifier() {
        String javaClassContent = "public class SampleClass { void 1doSomething(){;;;;} }";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        JavaParser javaParser = new JavaParser(tokens);
        CustomErrorListener errorListener = new CustomErrorListener(tokens);
        javaParser.setTrace(true);
        javaParser.removeErrorListeners();
        javaParser.addErrorListener(errorListener);
        //TokenStreamListener listener = new TokenStreamListener(tokens);
        //ParseTreeWalker.DEFAULT.walk(listener, javaParser.compilationUnit());
        javaParser.compilationUnit();


        //ParseTree parseTree = javaParser.compilationUnit();
        logger.info("Corrected input sequence: " + errorListener.getReplacedCode());
        logger.info("======================================");

        javaLexer = new JavaLexer(CharStreams.fromString(errorListener.getReplacedCode()));
        tokens = new CommonTokenStream(javaLexer);
        javaParser = new JavaParser(tokens);
        errorListener = new CustomErrorListener(tokens);
        javaParser.setTrace(true);
        javaParser.removeErrorListeners();
        javaParser.addErrorListener(errorListener);
        //listener = new TokenStreamListener(tokens);
        //ParseTreeWalker.DEFAULT.walk(listener, javaParser.compilationUnit());
        javaParser.compilationUnit();
    }
}