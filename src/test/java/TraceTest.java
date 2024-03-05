import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class TraceTest {

    // Find the corresponding element in the original sequence for a given index in the modified sequence
    public static Integer findCorrespondingElementInOriginal(Integer[] original, Integer[] modified, int modifiedIndex) {
        // Check for invalid index
        if (modifiedIndex < 0 || modifiedIndex >= modified.length) {
            return null;
        }

        int[][] dp = lcsTable(original, modified);

        // Trace the LCS from the end to the start, collecting the path
        List<Integer> path = traceLCSPath(dp, original, modified);

        // Check if the modifiedIndex is within the path, and find its corresponding index in the original sequence
        if (path.contains(modifiedIndex)) {
            int pathIndex = path.indexOf(modifiedIndex);
            return original[pathIndex];
        }

        return null; // No corresponding element if the index is not part of the LCS path
    }

    // Construct the LCS table
    private static int[][] lcsTable(Integer[] a, Integer[] b) {
        int m = a.length, n = b.length;
        int[][] dp = new int[m + 1][n + 1];

        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i == 0 || j == 0) {
                    dp[i][j] = 0;
                } else if (a[i - 1].equals(b[j - 1])) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        return dp;
    }

    // Trace the LCS path and collect the indices of the modified sequence that are part of the LCS
    private static List<Integer> traceLCSPath(int[][] dp, Integer[] original, Integer[] modified) {
        int i = original.length, j = modified.length;
        List<Integer> path = new ArrayList<>();

        while (i > 0 && j > 0) {
            if (original[i - 1].equals(modified[j - 1])) {
                // When elements match, this index is part of the LCS path
                path.add(0, j - 1); // Prepend the index since we're tracing backwards
                i--;
                j--;
            } else if (dp[i - 1][j] > dp[i][j - 1]) {
                i--;
            } else {
                j--;
            }
        }

        return path;
    }

    @Test
    public void test() {
        Integer[] original = {9, 128, 80, 35, 128, 128, 78, 128, 128, 79, 80, 128, 128, 87, 31, 128, 78, 79, 84, 27, 128, 87, 67, 84, 128, 89, 128, 88, 128, 87, 31, 128, 78, 79, 84, 128, 89, 128, 88, 128, 87, 31, 128, 78, 79, 84, 21, 78, 27, 128, 87, 67, 84, 128, 89, 128, 86, 128, 78, 79, 84, 128, 100, 79, 80, 8, 128, 87, 128, 86, 128, 78, 128, 79, 84, 22, 78, 74, 94, 128, 79, 80, 128, 86, 128, 78, 128, 79, 84, 128, 86, 128, 78, 128, 86, 128, 78, 79, 79, 84, 128, 87, 67, 84, 128, 87, 31, 128, 78, 79, 84, 81, 15, 22, 78, 74, 94, 128, 79, 80, 128, 128, 87, 31, 128, 78, 79, 84, 27, 128, 87, 128, 86, 128, 78, 79, 84, 21, 78, 27, 128, 87, 67, 84, 128, 89, 128, 84, 128, 100, 79, 80, 128, 86, 128, 78, 128, 79, 84, 81, 128, 87, 31, 128, 78, 128, 86, 128, 78, 79, 102, 128, 79, 84, 81, 15, 22, 78, 128, 96, 74, 98, 128, 95, 74, 79, 80, 128, 87, 128, 104, 67, 102, 78, 128, 103, 74, 79, 84, 81, 15, 80, 128, 86, 128, 78, 128, 79, 84, 81, 81, 36, 128, 86, 128, 78, 79, 84, 81, 81};
        Integer[] modified = {9, 128, 80, 35, 128, 128, 78, 128, 128, 79, 80, 128, 128, 87, 31, 128, 78, 79, 84, 27, 128, 87, 67, 84, 128, 89, 128, 88, 128, 87, 31, 128, 78, 79, 84, 128, 89, 128, 88, 128, 87, 31, 128, 78, 79, 84, 21, 78, 27, 128, 87, 67, 84, 128, 89, 128, 86, 128, 78, 79, 84, 128, 100, 79, 80, 8, 128, 87, 128, 86, 128, 78, 128, 79, 84, 22, 78, 74, 94, 128, 79, 80, 128, 86, 128, 78, 128, 79, 84, 128, 86, 128, 78, 128, 86, 128, 78, 79, 79, 84, 128, 87, 67, 84, 128, 87, 31, 128, 78, 79, 84, 81, 15, 22, 78, 74, 94, 128, 79, 80, 128, 128, 87, 31, 128, 78, 79, 84, 27, 128, 87, 128, 86, 128, 78, 79, 84, 21, 78, 27, 128, 87, 67, 84, 128, 89, 128, 84, 128, 100, 79, 80, 128, 86, 128, 78, 128, 79, 84, 81, 128, 87, 31, 128, 63, 78, 128, 86, 128, 78, 79, 102, 128, 79, 84, 81, 15, 22, 78, 128, 96, 74, 98, 128, 95, 79, 80, 128, 87, 128, 104, 67, 102, 78, 128, 103, 74, 79, 84, 81, 15, 80, 128, 86, 128, 78, 128, 79, 84, 81, 81, 36, 128, 86, 128, 78, 79, 84, 81, 81};
        int modifiedIndex = 3; // For example, index of '5' in the modified sequence
        for (int i = 0; i < original.length ; i++) {
            if (!Objects.equals(original[i], modified[i])) {
                modifiedIndex = i;
                break;
            }
        }
        System.out.println(modifiedIndex);
        Integer correspondingElement = findCorrespondingElementInOriginal(original, modified, modifiedIndex);
        if (correspondingElement != null) {
            System.out.println("The corresponding element in the original sequence is: " + correspondingElement);
        } else {
            System.out.println("No corresponding element found.");
        }
    }
}
