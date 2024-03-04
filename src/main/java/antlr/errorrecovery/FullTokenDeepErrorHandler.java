package antlr.errorrecovery;

import antlr.evaluation.CompilationError;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.tree.ErrorNode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.stream.IntStream;

public class FullTokenDeepErrorHandler {

    private List<Token> tokenStream;
    private List<Token> modelStream;

    private List<CompilationError> compilationErrorList = new ArrayList<>();
    private HashMap<Integer, Token> replacements = new HashMap<>();
    private HashMap<Integer, Token> additions = new HashMap<>();
    private HashMap<Integer, Token> deletions = new HashMap<>();



    public FullTokenDeepErrorHandler(List<Token> tokenStream, List<Token> modelStream) {
        this.tokenStream = tokenStream;
        this.modelStream = modelStream;
    }



    public List<Token> reconcileErrorNodes(List<ErrorNode> errorNodes) {
        for (ErrorNode errorNode : errorNodes) {
            this.tokenStream = IntStream.range(0, tokenStream.size())
                    .mapToObj(index -> {
                        Token token = tokenStream.get(index);
                        if (errorNode.getSymbol().getTokenIndex() == -1) {
                            // TODO: Maybe add lookahead?
                            if (token.getCharPositionInLine() == errorNode.getSymbol().getCharPositionInLine()) {
                                try {
                                    //System.out.println("Replaced: "+ token + " with: " + modelStream.get(index) + " at Index: " + index);
                                    return modelStream.get(index);

                                } catch (Exception e) {
                                    //e.printStackTrace();
                                    // ignore, model has no answer for it (e.g., its output is shorter)
                                }
                            }
                        }
                        if (token.getTokenIndex() == errorNode.getSymbol().getTokenIndex()) {
                            try {
                                //System.out.println("Replaced: "+ token + " with: " + modelStream.get(index) + " at Index: " + index);
                                return modelStream.get(index);

                            } catch (Exception e) {
                                //e.printStackTrace();
                                // ignore, model has no answer for it (e.g., its output is shorter)
                            }
                        }
                        return token;
                    }).toList();
        }
        return tokenStream;
    }
}
