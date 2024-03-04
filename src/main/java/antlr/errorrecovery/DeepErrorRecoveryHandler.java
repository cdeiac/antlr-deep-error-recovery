package antlr.errorrecovery;

import antlr.evaluation.CompilationError;
import org.antlr.v4.runtime.CommonToken;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.tree.ErrorNode;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DeepErrorRecoveryHandler {

    private List<Token> tokenList;
    private List<Token> reconstructedList;
    private int[] modelOutput;
    private int indexCorrection = 0;
    private List<CompilationError> compilationErrorList = new ArrayList<>();
    private HashMap<Integer, Token> replacements = new HashMap<>();
    private HashMap<Integer, Token> additions = new HashMap<>();
    private HashMap<Integer, Token> deletions = new HashMap<>();
    private final boolean bailOnError;
    private boolean bailed = false;



    public DeepErrorRecoveryHandler(List<Token> tokenStream,
                                    int[] antlrEncodedModelOutput,
                                    boolean bailOnError) {
        this.tokenList = tokenStream;//.stream()
                // filter out WS && EOF
                //.filter(t -> t.getType() != 125 && t.getType() != -1)
                //.toList();
        this.reconstructedList = new ArrayList<>(tokenList);
        this.modelOutput = antlrEncodedModelOutput;
        this.bailOnError = bailOnError;
    }


    public int[] reconcileTokens(List<Token> tokens) {

        for (Token token : tokens) {
            if (bailOnError && bailed) {
                break;
            }
            reconcile(token);
            bailed = true;
        /*    List<Token> adaptedTokenStream = IntStream.range(0, tokenList.size())
                    .mapToObj(index -> {
                        Token token = tokenList.get(index);
                        if (token.equals(errorNode.getSymbol())) {
                            try {
                                CommonToken newToken = new CommonToken(token);
                                newToken.setType(modelOutput[index]);
                                //System.out.println("Replaced: "+ token.getType() + " with: " + newToken.getType() + " at Index: " + index);
                                return newToken;
                            } catch (Exception e) {
                                e.printStackTrace();
                                // ignore, model has no answer for it (e.g., its output is shorter)
                            }
                        }
                        return token;
                    }).toList();
            this.tokenList = adaptedTokenStream;
            */

        }
        return applyOperations();
    }

    public int[] reconcileCompilationErrors(List<CompilationError> compilationErrors) {

        for (CompilationError compilationError : compilationErrors) {
            if (bailOnError && bailed) {
                break;
            }
            Token errorToken = tokenList.get(compilationError.getPosition());
            reconcile(errorToken);
            bailed = true;
        /*    List<Token> adaptedTokenStream = IntStream.range(0, tokenList.size())
                    .mapToObj(index -> {
                        Token token = tokenList.get(index);
                        if (token.equals(errorNode.getSymbol())) {
                            try {
                                CommonToken newToken = new CommonToken(token);
                                newToken.setType(modelOutput[index]);
                                //System.out.println("Replaced: "+ token.getType() + " with: " + newToken.getType() + " at Index: " + index);
                                return newToken;
                            } catch (Exception e) {
                                e.printStackTrace();
                                // ignore, model has no answer for it (e.g., its output is shorter)
                            }
                        }
                        return token;
                    }).toList();
            this.tokenList = adaptedTokenStream;
            */

        }
        return applyOperations();
    }

    public int[] reconcileErrorNodes(List<ErrorNode> errorNodes) {

        for (ErrorNode errorNode : errorNodes) {
            if (bailOnError && bailed) {
                break;
            }
            reconcile(errorNode.getSymbol());
            bailed = true;
        /*    List<Token> adaptedTokenStream = IntStream.range(0, tokenList.size())
                    .mapToObj(index -> {
                        Token token = tokenList.get(index);
                        if (token.equals(errorNode.getSymbol())) {
                            try {
                                CommonToken newToken = new CommonToken(token);
                                newToken.setType(modelOutput[index]);
                                //System.out.println("Replaced: "+ token.getType() + " with: " + newToken.getType() + " at Index: " + index);
                                return newToken;
                            } catch (Exception e) {
                                e.printStackTrace();
                                // ignore, model has no answer for it (e.g., its output is shorter)
                            }
                        }
                        return token;
                    }).toList();
            this.tokenList = adaptedTokenStream;
            */

        }
        return applyOperations();
    }

    private void reconcile(Token token) {
        int actualTokenIndex = this.getErrorTokenPositionOfTokenStream(token);

        if (token.getTokenIndex() < 0) {
            // missing symbol
            Optional<Token> matchedToken = tokenList.stream()
                    .filter(t -> t.getCharPositionInLine() == token.getCharPositionInLine())
                    .findFirst();
            if (matchedToken.isPresent()) {
                actualTokenIndex = getErrorTokenPositionOfTokenStream(matchedToken.get());
            }
        }
        if (actualTokenIndex != -1 && actualTokenIndex < modelOutput.length-1) {
            //System.out.println(msg);
            //System.out.println("Noisy: [" + reconstructedList.get(actualTokenIndex).getType() + ", " + reconstructedList.get(actualTokenIndex+1).getType() + " , " +  "] , Model: [" + modelOutput[actualTokenIndex]+ ", " + modelOutput[actualTokenIndex+1] + "]");
            int modelCurrent = modelOutput[actualTokenIndex + indexCorrection];
            int modelLA = modelOutput[actualTokenIndex + 1 + indexCorrection];
            int seqCurrent = reconstructedList.get(actualTokenIndex).getType();
            int seqLA = actualTokenIndex < reconstructedList.size()-1 ?  reconstructedList.get(actualTokenIndex + 1).getType() : -1;
            // insert
            //System.out.println("rec: [ " + reconstructedList.get(actualTokenIndex).getType() + ", " + reconstructedList.get(actualTokenIndex + 1).getType() + "] | model: [ " + modelOutput[actualTokenIndex] + ", " + modelOutput[actualTokenIndex + 1] + "]");
            if (modelCurrent == seqCurrent && modelLA == seqLA) {
                // do nothing
            } //else if (modelLA == seqLA) {
                // TODO: Set index to next position?
            //}
            else if (seqCurrent == modelLA) {
                //System.out.println("INSERT");
                addTokenFromModelOutput(actualTokenIndex);
                //indexCorrection+=1;
            } else if (seqLA == modelCurrent) {
                //System.out.println("DELETE");
                removeTokenFromModelOutput(actualTokenIndex);
                //indexCorrection-=1;
            } else {
                //System.out.println("MODIFY");
                replaceTokenFromModelOutput(actualTokenIndex);
            }
        }
        else if (modelOutput.length > reconstructedList.size()) {
            // if model has more tokens, we add them
            { // correctionIndex < modelOutput.length &&
                int tokensToAdd = modelOutput.length - reconstructedList.size();
                for (int i = tokensToAdd; i > 0; i--) {
                    addTokenFromModelOutput(modelOutput.length-1-i);
                }
            }
        }
    }

    private void removeTokenFromModelOutput(int actualTokenIndex) {
        //reconstructedList.remove(actualTokenIndex);
        this.deletions.put(actualTokenIndex, null);
    }

    private void replaceTokenFromModelOutput(int actualTokenIndex) {
        CommonToken newToken = new CommonToken(modelOutput[actualTokenIndex]);
        newToken.setTokenIndex(actualTokenIndex);
        this.replacements.put(actualTokenIndex, newToken); //reconstructedList.set(actualTokenIndex, newToken);
        //this.modelOutput = replaceElement(modelOutput, actualTokenIndex, newToken.getType());
    }

    private void addTokenFromModelOutput(int actualTokenIndex) {
        // TODO: index is wrong, should be initial token list size?
        CommonToken newToken = new CommonToken(modelOutput[actualTokenIndex]);
        newToken.setTokenIndex(tokenList.size()); // +addedTokens
        //this.replacements.put(actualTokenIndex, newToken); //reconstructedList.add(newToken);
        //modelOutput = addElement(modelOutput, newToken.getTokenIndex(), newToken.getType());
        this.additions.put(actualTokenIndex, newToken);
    }

    public int getErrorTokenPositionOfTokenStream(Token currentToken) {
        //Optional<Token> matchedToken =  reconstructedList.stream().filter(t -> t.getTokenIndex() == currentToken.getTokenIndex()+netModifications).findFirst();
        //return matchedToken.isPresent() ? reconstructedList.indexOf((Token) currentToken) : -1;
        //reconstructedList.indexOf(currentToken);
        return IntStream.range(0, reconstructedList.size())
                .filter(i -> reconstructedList.get(i).getTokenIndex() == (currentToken.getTokenIndex()))
                .findFirst() // Find the first matching index
                .orElse(-1);
    }

    public int getTotalNumberOfTokens() {
        return this.tokenList.size();
    }

    public List<CompilationError> getCompilationErrorList() {
        return new ArrayList<>(this.compilationErrorList.stream().collect(Collectors.toMap(
                        CompilationError::getPosition, // Key mapper
                        Function.identity(), // Value mapper (identity function)
                        (existing, replacement) -> existing, // Merge function to keep existing objects
                        HashMap::new // Use LinkedHashMap to preserve the order
                ))
                .values());
    }


    public int[] getReconstructedSequence() {

        applyOperations();
        return reconstructedList.stream()
                .map(Token::getType)
                .mapToInt(Integer::intValue)
                .toArray();
    }

    // Apply all operations to the list
    public int[] applyOperations() {
        /*
        try {
            // Apply modifications
            for (Map.Entry<Integer, Token> entry : replacements.entrySet()) {
                int index = entry.getKey();
                Token newElement = entry.getValue();
                reconstructedList.set(index, newElement);
            }
            // Apply deletions first to avoid index shifting
            TreeMap<Integer, Token> sortedMap = new TreeMap<>(Comparator.reverseOrder());
            sortedMap.putAll(deletions);
            for (Integer index : sortedMap.keySet()) {
                reconstructedList.remove((int) index);
            }
            // Apply insertions
            for (Map.Entry<Integer, Token> entry : additions.entrySet()) {
                int index = entry.getKey();
                Token element = entry.getValue();

                reconstructedList.add(index, element);
            }
        } catch (Exception e) {
            e.printStackTrace();;
        }
        // Clear the operations
        //additions.clear();
        //deletions.clear();
        //replacements.clear();
        return reconstructedList.stream()
                .map(Token::getType)
                .mapToInt(Integer::intValue)
                .toArray();
        */
        List<Integer> newList = new ArrayList<>();
        for (Token item : tokenList) {
            int actualTokenIndex = getErrorTokenPositionOfTokenStream(item);
            if (deletions.containsKey(actualTokenIndex)) {
                newList.add(null);
            } else if (replacements.containsKey(actualTokenIndex)) {
                newList.add(replacements.get(actualTokenIndex).getType()); // Replace value from replaceMap
            }
            else if (additions.containsKey(actualTokenIndex)) {
                // first add the new token, then the current one
                newList.add(additions.get(actualTokenIndex).getType());
                newList.add(item.getType());
            } //else newList.add(replacements.getOrDefault(id, item).getType());
            else {
                // just add it
                newList.add(item.getType());
            }
        }
        return newList.stream().filter(Objects::nonNull).mapToInt(n -> n).toArray();
    }

    private static int[] deleteElement(int[] array, int index) {
        // Check if the index is within bounds
        if (index < 0 || index >= array.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for length " + array.length);
        }

        // Create a new array with size one less than the original array
        int[] newArray = new int[array.length - 1];

        // Copy elements from the original array to the new array, skipping the element at the specified index
        for (int i = 0, j = 0; i < array.length; i++) {
            if (i != index) {
                newArray[j++] = array[i];
            }
        }

        return newArray;
    }

    private static int[] replaceElement(int[] array, int index, int newValue) {
        // Check if the index is within bounds
        if (index < 0 || index >= array.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for length " + array.length);
        }

        // Replace the element at the specified index
        array[index] = newValue;
        return array;
    }

    private static int[] addElement(int[] array, int index, int newValue) {
        // Check if the index is within bounds
        if (index < 0 || index > array.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for length " + array.length);
        }

        // Create a new array with size one greater than the original array
        int[] newArray = new int[array.length + 1];

        // Copy elements from the original array to the new array, adding the new element at the specified index
        for (int i = 0, j = 0; i < newArray.length; i++) {
            if (i == index) {
                newArray[i] = newValue;
            } else {
                newArray[i] = array[j++];
            }
        }

        return newArray;
    }

    public int[] applyReplacements() {
        for (Map.Entry<Integer, Token> entry : replacements.entrySet()) {
            reconstructedList.set(entry.getKey(), entry.getValue());
            // Do something with key and value
        }
        return reconstructedList.stream()
                .map(Token::getType)
                .mapToInt(Integer::intValue)
                .toArray();
    }
}
