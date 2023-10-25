import antlr.JavaLexer;
import antlr.JavaParser;
import antlr.extensions.CustomErrorListener;
import antlr.extensions.CustomErrorStrategy;
import antlr.extensions.CustomVisitor;
import antlr.extensions.TreeUtils;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.tree.ParseTree;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.logging.Logger;


public class JavaParserTest {

    private static final Logger logger = Logger.getLogger(JavaParserTest.class.getName());


    @Test
    public void identifier() {
        String javaClassContent = "int int i;ab;c";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        JavaParser javaParser = new JavaParser(tokens);
        javaParser.setTrace(true);
        javaParser.identifier();
    }

    @Test
    public void compilationUnit() {
        String javaClassContent = "public class Test { public void doSomething(int i,) {} }";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        JavaParser javaParser = new JavaParser(tokens);
        javaParser.setTrace(true);
        javaParser.setErrorHandler(new CustomErrorStrategy());
        javaParser.compilationUnit();

    }

    @Test
    public void method_name_starts_with_number_causes_error() {
        String javaClassContent = "public class SampleClass { void doSomething(){ int t = 9; int ;ab;;h;9;; int n = 1; } }";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        JavaParser javaParser = new JavaParser(tokens);
        javaParser.setTrace(true);
        //this.parseAndPrintParseTree(javaParser);
        //TokenStreamListener listener = new TokenStreamListener(tokens);
        //ParseTreeWalker.DEFAULT.walk(listener, javaParser.compilationUnit());
        ParseTree parseTree = javaParser.compilationUnit();
        //logger.info("Replaced code: " + listener.getReplacedCode());
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

    @Test
    public void test_with_visitor() {
        String javaClassContent = "public class SampleClass { void 1doSomething(){} }";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        JavaParser javaParser = new JavaParser(tokens);
        javaParser.setTrace(true);
        CustomVisitor customVisitor = new CustomVisitor();
        customVisitor.visit(javaParser.compilationUnit());
    }

    @Test
    public void test_parseTreePattern() {
        String javaClassContent = "public class SampleClass { void 1doSomething(){} }";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        JavaParser javaParser = new JavaParser(tokens);
        javaParser.setTrace(true);
        this.parseAndPrintParseTree(javaParser);

    }

    private void parseAndPrintParseTree(JavaParser parser) {
        parser.setBuildParseTree(true);
        RuleContext tree = parser.compilationUnit();
        List<String> ruleNamesList = List.of(parser.getRuleNames());
        String prettyTree = TreeUtils.toPrettyTree(tree, ruleNamesList);
        logger.info(prettyTree);
    }
}