package antlr.validator;

import antlr.JavaParserBaseVisitor;
import antlr.extensions.CustomErrorListener;
import org.antlr.v4.runtime.tree.ErrorNode;
import org.antlr.v4.runtime.tree.TerminalNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ErrorReporterVisitor extends JavaParserBaseVisitor<Integer> {

    private static final Logger logger = LoggerFactory.getLogger(CustomErrorListener.class);


    @Override
    public Integer visitTerminal(TerminalNode terminal) {

        logger.info("Visit Terminal: " + terminal.getText());
        return super.visitTerminal(terminal);
    }

    @Override
    public Integer visitErrorNode(ErrorNode node) {
        logger.info("Visit error node: " + node.getText());
        return super.visitTerminal(node);
    }
}
