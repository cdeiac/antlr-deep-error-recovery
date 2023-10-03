package extensions;

import com.antlr.grammars.java.JavaParser;
import com.antlr.grammars.java.JavaParserBaseVisitor;

public class EvaluateExpressionVisitor extends JavaParserBaseVisitor<Integer> {

    @Override
    public Integer visitIdentifier(JavaParser.IdentifierContext ctx) {

        return visitChildren(ctx);
    }



}
