package antlr.utils;

import org.antlr.v4.runtime.Token;

import java.util.ArrayList;
import java.util.List;

public class FileDiff {

    public static List<Integer> reconstructOperations(List<Token> noisy, List<Token> model) {

        List<Integer> originalList = noisy.stream()
                .filter(t -> t.getType() != 125 && t.getType() != -1)
                .map(Token::getType)
                .mapToInt(Integer::intValue)
                .boxed()
                .toList();
        List<Integer> modifiedList = model.stream()
                .filter(t -> t.getType() != 125 && t.getType() != -1)
                .map(Token::getType)
                .mapToInt(Integer::intValue)
                .boxed()
                .toList();

        int netChange = 0;
        List<Integer> netChanges = new ArrayList<>();
        int i = 0, j = 0;
        while (i < originalList.size()-1 && j < modifiedList.size()-1) {
            if (originalList.get(i).equals(modifiedList.get(j))) {
                // No operation needed, elements are the same
                i++;
                j++;
                netChanges.add(netChange);
            } else {
                if (originalList.get(i+1).equals(modifiedList.get(j+1))) {
                    // Element was modified
                    i++;
                    j++;
                    netChanges.add(netChange);
                }
                else if (!originalList.get(i).equals(modifiedList.get(j + 1))) {
                    // Element was inserted
                    j++;
                    netChange += 1;
                    netChanges.add(netChange);
                } else { //if (!originalList.get(i + 1).equals(modifiedList.get(j))) {
                    // Element was deleted
                    i++;
                    netChange -= 1;
                    netChanges.add(netChange);
                }
            }
        }
        netChanges.add(netChange); // last element
        return netChanges;
    }

    private static int netDiffUntilPosition(List<Integer> diffs, int position) {
        return diffs.get(position);
    }

    public static int matchOperation(List<Token> noisy, List<Token> model, int position) {
        List<Integer> originalList = noisy.stream()
                .filter(t -> t.getType() != 125 && t.getType() != -1)
                .map(Token::getType)
                .mapToInt(Integer::intValue)
                .boxed()
                .toList();
        List<Integer> modifiedList = model.stream()
                .filter(t -> t.getType() != 125 && t.getType() != -1)
                .map(Token::getType)
                .mapToInt(Integer::intValue)
                .boxed()
                .toList();
        int modelPosition = position;
        if (originalList.size() != modifiedList.size()) {
            List<Integer> diffList = reconstructOperations(noisy, model);

            //if (position == originalList.size()) {
            //    return new InsertAllOperation(-1); // EOF --> add what we have
            //}
            modelPosition += netDiffUntilPosition(diffList, position);
            if (modelPosition == originalList.size()) {
                return -1; // EOF
            }
        }
        return modelPosition;
        /*
        if (position < originalList.size() && modelPosition < modifiedList.size() && originalList.get(position).equals(modifiedList.get(modelPosition))) {
            // No operation needed, elements are the same
            return new NoOperation(-1);
        } else {
            if (modelPosition < modifiedList.size() && (position == originalList.size() || !originalList.get(position).equals(modifiedList.get(modelPosition + 1)))) {
                return new InsertOperation(modifiedList.get(modelPosition));
            } else if (modelPosition == modifiedList.size() || !originalList.get(position + 1).equals(modifiedList.get(modelPosition))) {
                // Element was deleted
                return new DeleteOperation(-1);
            } else {
                return new UpdateOperation(modelPosition);
            }
        }*/
    }
}