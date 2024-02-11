package antlr.evaluation;

import antlr.JavaParserBaseVisitor;
import org.antlr.v4.runtime.tree.ErrorNode;
import org.antlr.v4.runtime.tree.TerminalNode;

import java.util.ArrayList;
import java.util.List;

public class ErrorReporterVisitor extends JavaParserBaseVisitor<Integer> {
    private int visitedNodes = 0;
    public List<ErrorNode> errorNodes = new ArrayList<>();


    @Override
    public Integer visitTerminal(TerminalNode terminal) {
        visitedNodes += 1;
        //System.out.println("visitTerminal: " + terminal.getSymbol().getText());
        return super.visitTerminal(terminal);
    }

    @Override
    public Integer visitErrorNode(ErrorNode node) {
        //System.out.println("visitErrorNode: " + node.getSymbol().getText());
        errorNodes.add(node);
        if (!node.getSymbol().getText().contains("missing")) {
            visitedNodes += 1;
        }
        return super.visitTerminal(node);
    }


    public int getVisitedNodes() {
        return visitedNodes;
    }

    public int getErrorNodes() {
        return errorNodes.stream()
                .map(err -> err.getSymbol().getCharPositionInLine()) // cannot take index due to modifications!
                .distinct()
                .toList()
                .size();
    }
}
