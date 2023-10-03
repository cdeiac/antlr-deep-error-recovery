package extensions;

import com.antlr.grammars.java.JavaParser;
import com.antlr.grammars.java.JavaParserBaseListener;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.TokenStreamRewriter;

public class TokenStreamListener extends JavaParserBaseListener {

    private final TokenStreamRewriter rewriter;


    public TokenStreamListener(CommonTokenStream tokens) {
        rewriter = new TokenStreamRewriter(tokens);
    }


    @Override
    public void enterIdentifier(JavaParser.IdentifierContext ctx) {


        rewriter.replace(ctx.start, ctx.stop, "test");//new EvaluateExpressionVisitor().visit(ctx));
    }

    public String getReplacedCode() {
        return rewriter.getText();
    }
}
