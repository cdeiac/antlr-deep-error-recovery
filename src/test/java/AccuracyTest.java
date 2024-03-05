import org.junit.jupiter.api.Test;

import java.util.List;

public class AccuracyTest {
/*
    // Method to compute the accuracy considering specific indexes and using LCS for alignment
    public static double computeLcsBasedAccuracy(Integer[] original, Integer[] modified, List<Integer> indexes) {
        int[][] dp = lcsTable(original, modified);
        List<Integer> alignedIndexes = findAlignedIndexes(dp, original, modified, indexes);

        // Compute matches for the specified indexes after alignment
        int matches = 0;
        for (int index : alignedIndexes) {
            if (indexes.contains(index)) {
                matches++;
            }
        }

        // Calculate accuracy based on matches and the total number of specified indexes
        return !indexes.isEmpty() ? (double) matches / indexes.size() : 0;
    }

    // Helper method to construct the LCS table
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

    // Helper method to find aligned indexes according to LCS
    private static List<Integer> findAlignedIndexes(int[][] dp, Integer[] original, Integer[] modified, List<Integer> indexes) {
        List<Integer> alignedIndexes = new ArrayList<>();
        int i = original.length, j = modified.length;
        while (i > 0 && j > 0) {
            if (original[i - 1].equals(modified[j - 1])) {
                if (indexes.contains(i - 1)) { // Check if this index was one of the specified indexes
                    alignedIndexes.add(i - 1);
                }
                i--;
                j--;
            } else if (dp[i - 1][j] > dp[i][j - 1]) {
                i--;
            } else {
                j--;
            }
        }
        return alignedIndexes;
    }

 */

    // Method to compute the element-wise accuracy
    public static double computeAccuracy(Integer[] original, Integer[] modified) {
        int[][] dp = lcsTable(original, modified);

        // Reconstruct the LCS to find the alignment
        int i = original.length, j = modified.length;
        int commonElements = 0;
        while (i > 0 && j > 0) {
            if (original[i - 1].equals(modified[j - 1])) {
                commonElements++; // Found a common element
                i--;
                j--;
            } else if (dp[i - 1][j] > dp[i][j - 1]) {
                i--; // Deletion or modification in modified list
            } else {
                j--; // Insertion in modified list
            }
        }

        // Calculate accuracy as the ratio of common elements to the length of the original list
        return (double) commonElements / original.length;
    }

    // Helper method to construct the LCS table
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


    @Test
    public void test() {
        Integer[] original = {9, 128, 80, 35, 128, 128, 78, 128, 128, 79, 80, 128, 128, 87, 31, 128, 78, 79, 84, 27, 128, 87, 67, 84, 128, 89, 128, 88, 128, 87, 31, 128, 78, 79, 84, 128, 89, 128, 88, 128, 87, 31, 128, 78, 79, 84, 21, 78, 27, 128, 87, 67, 84, 128, 89, 128, 86, 128, 78, 79, 84, 128, 100, 79, 80, 8, 128, 87, 128, 86, 128, 78, 128, 79, 84, 22, 78, 74, 94, 128, 79, 80, 128, 86, 128, 78, 128, 79, 84, 128, 86, 128, 78, 128, 86, 128, 78, 79, 79, 84, 128, 87, 67, 84, 128, 87, 31, 128, 78, 79, 84, 81, 15, 22, 78, 74, 94, 128, 79, 80, 128, 128, 87, 31, 128, 78, 79, 84, 27, 128, 87, 128, 86, 128, 78, 79, 84, 21, 78, 27, 128, 87, 67, 84, 128, 89, 128, 84, 128, 100, 79, 80, 128, 86, 128, 78, 128, 79, 84, 81, 128, 87, 31, 128, 78, 128, 86, 128, 78, 79, 102, 128, 79, 84, 81, 15, 22, 78, 128, 96, 74, 98, 128, 95, 74, 79, 80, 128, 87, 128, 104, 67, 102, 78, 128, 103, 74, 79, 84, 81, 15, 80, 128, 86, 128, 78, 128, 79, 84, 81, 81, 36, 128, 86, 128, 78, 79, 84, 81, 81};
        Integer[] modified = {9, 128, 80, 35, 128, 128, 78, 128, 128, 79, 80, 128, 128, 87, 31, 128, 78, 79, 84, 27, 128, 87, 67, 84, 128, 89, 128, 88, 128, 87, 31, 128, 78, 79, 84, 128, 89, 128, 88, 128, 87, 31, 128, 78, 79, 84, 21, 78, 27, 128, 87, 67, 84, 128, 89, 128, 86, 128, 78, 79, 84, 128, 100, 79, 80, 8, 128, 87, 128, 86, 128, 78, 128, 79, 84, 22, 78, 74, 94, 128, 79, 80, 128, 86, 128, 78, 128, 79, 84, 128, 86, 128, 78, 128, 86, 128, 78, 79, 79, 84, 128, 87, 67, 84, 128, 87, 31, 128, 78, 79, 84, 81, 15, 22, 78, 74, 94, 128, 79, 80, 128, 128, 87, 31, 128, 78, 79, 84, 27, 128, 87, 128, 86, 128, 78, 79, 84, 21, 78, 27, 128, 87, 67, 84, 128, 89, 128, 84, 128, 100, 79, 80, 128, 86, 128, 78, 128, 79, 84, 81, 128, 87, 31, 128, 63, 78, 128, 86, 128, 78, 79, 102, 128, 79, 84, 81, 15, 22, 78, 128, 96, 74, 98, 128, 95, 79, 80, 128, 87, 128, 104, 67, 102, 78, 128, 103, 74, 79, 84, 81, 15, 80, 128, 86, 128, 78, 128, 79, 84, 81, 81, 36, 128, 86, 128, 78, 79, 84, 81, 81};
        List<Integer> indexes = List.of(0, 1, 2, 3, 4); // Specify the indexes to consider for accuracy

        double accuracy = computeAccuracy(original, modified);
        System.out.printf("LCS-based accuracy for specified indexes: %.4f", accuracy);
    }
}
