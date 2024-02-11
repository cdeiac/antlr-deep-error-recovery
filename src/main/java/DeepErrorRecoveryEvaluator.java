import antlr.evaluation.JavaProgramEvaluator;

import java.util.List;

public class DeepErrorRecoveryEvaluator {

    public static void main(String[] args) throws Exception {
        List<String> cvDirs = List.of("00_1", "00_2", "00_4", "00_8", "01_6", "03_2", "06_4", "12_8");
        int numOfFolds = 3;
        cvDirs.forEach(cvDir -> {
            for (int i = 0; i < numOfFolds; i++) {
                var evaluator = new JavaProgramEvaluator(cvDir, i);
                evaluator.parseAndEvaluate();
                System.out.println("Done with: " + cvDir + " Iteration: " + i);
            }
        });
    }
}
