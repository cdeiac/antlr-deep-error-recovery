package antlr.extensions;

import antlr.JavaParser;
import antlr.JavaParserBaseListener;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.TokenStreamRewriter;

public class TokenStreamListener extends JavaParserBaseListener {

    private final TokenStreamRewriter rewriter;


    public TokenStreamListener(CommonTokenStream tokens) {
        rewriter = new TokenStreamRewriter(tokens);
    }


    @Override
    public void enterIdentifier(JavaParser.IdentifierContext ctx) {

        // do replacement here after rewriter.replace
        rewriter.replace(ctx.start, ctx.start, "CLASS");//new EvaluateExpressionVisitor().visit(ctx));
    }

    public String getReplacedCode() {
        return rewriter.getText();
    }
}
