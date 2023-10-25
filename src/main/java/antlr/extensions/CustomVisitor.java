package antlr.extensions;

import antlr.JavaParser;
import antlr.JavaParserBaseVisitor;
import org.antlr.v4.runtime.tree.ErrorNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CustomVisitor extends JavaParserBaseVisitor<Integer> {

    private static final Logger logger = LoggerFactory.getLogger(CustomErrorListener.class);


    @Override
    public Integer visitIdentifier(JavaParser.IdentifierContext ctx) {

        logger.info("Visit identifier: " + ctx.getText());
        return visitChildren(ctx);
    }

    @Override
    public Integer visitErrorNode(ErrorNode node) {
        logger.info("Visit error node: " + node.getText());
        return this.defaultResult();
    }
}
