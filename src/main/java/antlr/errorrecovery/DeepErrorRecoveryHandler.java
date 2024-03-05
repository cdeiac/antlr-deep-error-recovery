package antlr.errorrecovery;

import antlr.evaluation.CompilationError;
import antlr.evaluation.RecoveryOperation;
import org.antlr.v4.runtime.CommonToken;
import org.antlr.v4.runtime.Token;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static antlr.converters.ANTLRDataConverter.ANTLR_INDEX_2_TOKEN;

public class DeepErrorRecoveryHandler {

    private List<Token> tokenList;
    private List<Token> reconstructedList;
    private List<Token> modelOutput;
    private int[] original;
    private int indexCorrection = 0;
    private List<CompilationError> compilationErrorList = new ArrayList<>();
    private List<RecoveryOperation> operations = new ArrayList<>();
    private HashMap<Integer, Token> replacements = new HashMap<>();
    private HashMap<Integer, Token> additions = new HashMap<>();
    private HashMap<Integer, Token> deletions = new HashMap<>();


    public DeepErrorRecoveryHandler(List<Token> tokenStream,
                                    List<Token> antlrEncodedModelOutput,
                                    int[] original) {
        this.tokenList = new ArrayList<>(tokenStream).stream()
                .filter(t -> t.getType() != 125 && t.getType() != -1)
                .toList();
        this.reconstructedList = new ArrayList<>(tokenList);
        this.modelOutput = antlrEncodedModelOutput.stream()
                .filter(t -> t.getType() != 125 && t.getType() != -1)
                .toList();
        this.original = original;
    }

    public void reconcileCompilationErrors(List<CompilationError> compilationErrors) {

        for (CompilationError compilationError : compilationErrors) {
            int errorPosition = compilationError.getPosition();
            CompilationError adjustedError = new CompilationError(compilationError.getType(), compilationError.getPosition(), getAdjustedPosition(errorPosition));
            this.compilationErrorList.add(adjustedError);
            if (errorPosition == -1) {
                if (tokenList.size() < modelOutput.size()) {
                    addTokenFromModelOutput(tokenList.size());
                }

            }
            else if (errorPosition < tokenList.size()) {
                Token errorToken = tokenList.get(errorPosition);
                reconcile(errorToken, compilationError);
            }

        }
        //return applyOperations();
    }

    private void reconcile(Token token, CompilationError error) {
        int actualTokenIndex = this.getErrorTokenPositionOfTokenStream(token);

        if (token.getTokenIndex() < 0) {
            // missing symbol
            Optional<Token> matchedToken = tokenList.stream()
                    .filter(t -> t.getCharPositionInLine() == token.getCharPositionInLine())
                    .findFirst();
            if (matchedToken.isPresent()) {
                actualTokenIndex = matchedToken.get().getTokenIndex();
            }
        }
        if (actualTokenIndex != -1 && actualTokenIndex < modelOutput.size()-1) {
            int modelCurrent = modelOutput.get(actualTokenIndex + indexCorrection).getType();
            int modelLA = modelOutput.get(actualTokenIndex + 1 + indexCorrection).getType();
            int seqCurrent = reconstructedList.get(actualTokenIndex).getType();
            int seqLA = actualTokenIndex < reconstructedList.size()-1 ?  reconstructedList.get(actualTokenIndex + 1).getType() : -1;
            // insert
            if (modelCurrent != seqCurrent && modelLA == seqLA) {
                // do nothing
                replaceTokenFromModelOutput(actualTokenIndex);
            }
            else if (seqCurrent == modelLA) {
                addTokenFromModelOutput(actualTokenIndex);
            } else if (seqLA == modelCurrent) {
                removeTokenFromModelOutput(actualTokenIndex);
            }
        }
    }

    private void removeTokenFromModelOutput(int actualTokenIndex) {
        //reconstructedList.remove(actualTokenIndex);
        this.deletions.put(actualTokenIndex, null);
    }

    private void replaceTokenFromModelOutput(int actualTokenIndex) {
        CommonToken newToken = new CommonToken(modelOutput.get(actualTokenIndex));
        newToken.setTokenIndex(actualTokenIndex);
        this.replacements.put(actualTokenIndex, newToken); //reconstructedList.set(actualTokenIndex, newToken);
        //this.modelOutput = replaceElement(modelOutput, actualTokenIndex, newToken.getType());
    }

    private void addTokenFromModelOutput(int actualTokenIndex) {
        CommonToken newToken = new CommonToken(modelOutput.get(actualTokenIndex));
        newToken.setTokenIndex(tokenList.size()); // +addedTokens
        this.additions.put(actualTokenIndex, newToken);
    }

    public int getErrorTokenPositionOfTokenStream(Token currentToken) {
        return IntStream.range(0, reconstructedList.size()+1)
                .filter(i -> reconstructedList.get(i).getTokenIndex() == (currentToken.getTokenIndex()))
                .findFirst() // Find the first matching index
                .orElse(-1);
    }

    public List<CompilationError> getCompilationErrorList() {
        return new ArrayList<>(this.compilationErrorList.stream()
                .collect(Collectors.toMap(
                        CompilationError::getPosition, // Key mapper
                        Function.identity(), // Value mapper (identity function)
                        (existing, replacement) -> existing, // Merge function to keep existing objects
                        HashMap::new)) // Use LinkedHashMap to preserve the order
                .values());
    }

    // Apply all operations to the list
    public int[] applyOperations() {
        for (Integer index : replacements.keySet()) {
            Token tokenBefore = reconstructedList.get(index);
            reconstructedList.set(index, replacements.get(index));
            int adjustedPosition = getAdjustedPosition(index);
            operations.add(
                    new RecoveryOperation(
                            "MOD",
                            index,
                            adjustedPosition,
                            ANTLR_INDEX_2_TOKEN.get(tokenBefore.getType()),
                            ANTLR_INDEX_2_TOKEN.get(replacements.get(index).getType()),
                            adjustedPosition < original.length ? ANTLR_INDEX_2_TOKEN.get(original[adjustedPosition]) : "EOF"
                    )
            );
        }
        for (Integer index : deletions.keySet()) {
            Token tokenBefore = reconstructedList.get(index);
            reconstructedList.set(index, deletions.get(index));
            int adjustedPosition = getAdjustedPosition(index);
            operations.add(
                    new RecoveryOperation(
                            "DEL",
                            index,
                            adjustedPosition,
                            ANTLR_INDEX_2_TOKEN.get(tokenBefore.getType()),
                            index+1 < reconstructedList.size() && reconstructedList.get(index+1) != null ? ANTLR_INDEX_2_TOKEN.get(reconstructedList.get(index+1).getType()) : "EOF",
                            adjustedPosition < original.length ? ANTLR_INDEX_2_TOKEN.get(original[adjustedPosition]) : "EOF"
                    )
            );
        }
        TreeMap<Integer, Token> reverseAdditions = new TreeMap<>(Collections.reverseOrder());
        reverseAdditions.putAll(additions);

        for (Integer index : reverseAdditions.keySet()) {
            Token tokenBefore = index < reconstructedList.size() ? reconstructedList.get(index) : null;
            reconstructedList.add(index, reverseAdditions.get(index));
            int adjustedPosition = getAdjustedPosition(index);
            operations.add(
                    new RecoveryOperation(
                            "INS",
                            index,
                            adjustedPosition,
                            ANTLR_INDEX_2_TOKEN.get(tokenBefore != null ? tokenBefore.getType() : "EOF"),
                            ANTLR_INDEX_2_TOKEN.get(additions.get(index).getType()),
                            adjustedPosition < original.length ? ANTLR_INDEX_2_TOKEN.get(original[adjustedPosition]) : "EOF"
                    )
            );
        }
        return reconstructedList.stream().filter(Objects::nonNull).mapToInt(Token::getType).toArray();
    }

    private int getAdjustedPosition(int position) {
        int deletionsCount = 0;
        int additionsCount = 0;
        for (Integer key : deletions.keySet()) {
            if (key < position) {
                deletionsCount++;
            }
        }
        for (Integer key : additions.keySet()) {
            if (key < position) {
                additionsCount++;
            }
        }
        return position - deletionsCount + additionsCount;
    }

    public List<RecoveryOperation> getOperations() {
        return operations;
    }
}
