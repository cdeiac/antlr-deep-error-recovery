package antlr.evaluation;

import antlr.converters.ANTLRModelConverter;
import antlr.converters.ModelDataConverter;
import antlr.utils.FileUtils;
import antlr.utils.TestData;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class JavaOracle {

    private static final Path CV_DIR_PATH = Paths.get("src/main/python/data/generated/cv");
    private static final String JSON_FILE_EXTENSION = ".json";
    private final antlr.errorrecovery.JavaOracle predictor;


    public JavaOracle(String modelPath) {
        this.predictor = new antlr.errorrecovery.JavaOracle(modelPath);
    }


    public void parseAndEvaluate(String cvDir) {
        // load data
        List<TestData> testDataList = FileUtils.loadAllTestData(cvDir);
        // predict
        List<int[]> antlrTokenIdList = this.predictAll(testDataList);
        //feedbackLoop(testDataList);
        // parse
        List<ParseStatistics> parseStatistics = this.parseAll(antlrTokenIdList);
        HashMap<String, Double> successfullyCompiledAverages = new HashMap<>();
        // aggregate results
        /*
        for (Map.Entry<String, List<ParseResult>> entry : parseStatistics.entrySet()) {
            List<Double> parseSuccessList = entry.getValue().stream()
                    .map(ParseStatistics::getSuccessfulParsePercentage)
                    .toList();
            successfullyCompiledAverages.put(entry.getKey(), computeAverageSuccessfullyCompiled(parseSuccessList));
        }
        FileUtils.saveParseStatistics(parseStatistics);*/
    }

    private double computeAverageSuccessfullyCompiled(List<Double> listOfSuccessfullyCompiled) {
        double total = 0.0;
        for (Double val : listOfSuccessfullyCompiled) {
            total += val;
        }
        return (double) listOfSuccessfullyCompiled.size() / total;
    }

    private List<int[]> predictAll(List<TestData> testDataList) {
        List<int[]> resultList = new ArrayList<>();
        // predict
        testDataList.forEach(testData -> {
            // obtain result in model IDs & map result to ANTLR IDs
            int[] antlrEncodedIds = this.predictor.predictMostProbableTokens(testData.getNoisyData());
            resultList.add(antlrEncodedIds);
        });
        return resultList;
    }

    private List<ParseStatistics> parseAll(List<int[]> antlrTokenIdList) {
        // create placeholder ANTLR Token Objects
        List<ParseStatistics> parseStatistics = new ArrayList<>();
        // parse with listener
        antlrTokenIdList.forEach(idArray -> parseStatistics.add(ParserEvaluator.parseWithErrorRecovery(idArray)));
        return parseStatistics;
    }

    private List<ParseStatistics> parseAll(List<int[]> antlrTokenIdList) {
        // create placeholder ANTLR Token Objects
        List<ParseStatistics> parseStatistics = new ArrayList<>();
        // parse with listener
        antlrTokenIdList.forEach(idArray -> parseStatistics.add(ParserEvaluator.parseWithErrorRecovery(idArray)));
        return parseStatistics;
    }

    private List<Integer> feedbackLoop(List<TestData> testDataList) {
        List<Integer> resultList = new ArrayList<>();
        // predict
        testDataList.forEach(
                testData -> {
                    // obtain result in model IDs & map result to ANTLR IDs
                    int parseAttempts = feedbackLoop(ModelDataConverter.encodeSequence(testData.getNoisyData()));
                    resultList.add(parseAttempts);
                });
        return resultList;
    }

    private int feedbackLoop(int[] antlrEncodedIds) {
        boolean doneParsing = false;
        int parsingAttempts = 1;
        int parsingBudget = 3 * antlrEncodedIds.length;
        int[] currentInput = ANTLRModelConverter.encodeSequence(antlrEncodedIds);

        ParseStatistics currentParseStatistics = ParserEvaluator.parseWithErrorRecovery(currentInput);
        if (currentParseStatistics.getSuccessfulParsePercentage() == 100.0) {
            return parsingAttempts;
        }
        while (!doneParsing || parsingAttempts < parsingBudget) {

            int[] output = this.predictor.predictSecondMostProbableToken(
                    currentInput,
                    currentParseStatistics.getCompilationErrors().get(0).getTokenIndex()
            );
            currentInput = output;
            ParserEvaluator.parseWithErrorRecovery(currentInput);
            if (currentParseStatistics.getSuccessfulParsePercentage() == 100.0) {
                doneParsing = true;
            }
            parsingAttempts += 1;

        }
        return parsingAttempts;
    }

    private int[] replaceErroneousToken(int[] input, int position) {
        return null;
    }

}