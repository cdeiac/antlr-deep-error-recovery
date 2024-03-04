package antlr.extensions;

import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.tree.ErrorNode;
import org.antlr.v4.runtime.tree.ParseTreeListener;
import org.antlr.v4.runtime.tree.TerminalNode;

public class CustomParseTreeListener implements ParseTreeListener {
    @Override
    public void visitTerminal(TerminalNode node) {
        System.out.println("ParseTreeListener visitTerminal: " + node.getSymbol().getText());
    }

    @Override
    public void visitErrorNode(ErrorNode node) {
        System.out.println("ParseTreeListener visitErrorNode: " + node.getSymbol().getText());
    }

    @Override
    public void enterEveryRule(ParserRuleContext ctx) {

    }

    @Override
    public void exitEveryRule(ParserRuleContext ctx) {

    }
}
