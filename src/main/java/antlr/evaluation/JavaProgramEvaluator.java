package antlr.evaluation;

import antlr.JavaLexer;
import antlr.JavaParser;
import antlr.converters.ANTLRDataConverter;
import antlr.converters.ANTLRModelConverter;
import antlr.converters.ANTLRPlaceholderToken;
import antlr.converters.ModelDataConverter;
import antlr.errorrecovery.DeepErrorRecoveryHandler2;
import antlr.errorrecovery.JavaOracle;
import antlr.extensions.CustomErrorStrategy;
import antlr.utils.FileUtils;
import antlr.utils.TestData;
import antlr.utils.TokenUtils;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class JavaProgramEvaluator {

    private final String cvDir;
    private final int cvIter;
    private final JavaOracle oracle;


    public JavaProgramEvaluator(String cvDir, int cvIter) {
        this.cvDir = cvDir;
        this.cvIter = cvIter;
        this.oracle = new JavaOracle(cvDir, cvIter);
    }


    public void parseAndEvaluate() {
        // load data
        List<TestData> testDataList = FileUtils.loadAllTestData(cvDir, cvIter);
        List<String> noisyData = testDataList.stream().map(TestData::getNoisyData).toList();
        List<String> originalData = testDataList.stream().map(TestData::getOriginalData).toList();
        this.parseAndEvaluateAllOn(noisyData, originalData);
        this.oracle.closePredictor();
    }

    private void parseAndEvaluateAllOn(List<String> noisySequences,
                                      List<String> originalSequences) {

        List<SequenceStatistics> sequenceStatistics = new ArrayList<>();
        List<ParseStatistics> parseStatistics = new ArrayList<>();
        for (int i = 0; i < noisySequences.size(); i++) {
            String original = originalSequences.get(i);
            String noisy = noisySequences.get(i);
            // parse
            ANTLRStatistics antlrStats = this.parseWithAntlrEr(noisy, original);
            DERStatistics derStats = this.parseWithDeepErrorRecovery(noisy, original, antlrStats);
            DERStatistics onFileStats = this.parseWithOnFileDeepErrorRecovery(noisy, original, antlrStats);
            // aggregate
            if (antlrStats != null && derStats != null && onFileStats != null) {
                ParseStatistics finalStats = ParseStatistics.finalizeStatistics(
                        antlrStats,
                        derStats,
                        onFileStats);
                parseStatistics.add(finalStats);
                sequenceStatistics.add(SequenceStatistics.finalizeSequenceStatistics(
                        antlrStats,
                        derStats,
                        onFileStats
                ));
            }

        }
        System.out.println("Base Comp. Errors: " + parseStatistics.stream().mapToInt(s -> s.getBaseCompilationErrors().size()).average().orElse(-1));
        System.out.println("Base Reconstruction: " + parseStatistics.stream().map(ParseStatistics::getBaseReconstruction).toList().stream().mapToDouble(Double::doubleValue).average().orElse(-1));
        System.out.println("ANTLR ER On-Error Avg. Comp. Errors: " + parseStatistics.stream().mapToInt(s -> s.getAntlrCompilationErrors().size()).average().orElse(-1));
        System.out.println("ANTLR ER On-Error Avg. Reconstruction: " + parseStatistics.stream().map(ParseStatistics::getAntlrReconstruction).toList().stream().mapToDouble(Double::doubleValue).average().orElse(-1));
        System.out.println("DER Avg. Comp. Errors: " + parseStatistics.stream().mapToInt(s -> s.getDerOnErrorCompilationErrors().size()).average().orElse(-1));
        System.out.println("DER Avg. Reconstruction: " + parseStatistics.stream().map(ParseStatistics::getDerOnErrorReconstruction).toList().stream().mapToDouble(Double::doubleValue).average().orElse(-1));
        System.out.println("DER On-File Avg. Comp. Errors: " + parseStatistics.stream().mapToInt(s -> s.getDerOnFileCompilationErrors().size()).average().orElse(-1));
        System.out.println("DER On-File Avg. Reconstruction: " + parseStatistics.stream().map(ParseStatistics::getDerOnFileReconstruction).toList().stream().mapToDouble(Double::doubleValue).average().orElse(-1));
        FileUtils.saveParseStatistics(cvDir, cvIter, parseStatistics);
        FileUtils.saveSequenceStatistics(cvDir, cvIter, sequenceStatistics);
    }


    private ANTLRStatistics parseWithAntlrEr(String noisySequence, String originalSequence) {
        String[] original = originalSequence.split(" ");
        String[] noisy = noisySequence.split(" ");
        // decode sequence
        int[] encodedInput = ANTLRDataConverter.mapTokenToIds(noisySequence);
        String decodedInput = ANTLRPlaceholderToken.replaceSourceWithDummyTokens(encodedInput);
        int[] encodedTarget = ANTLRDataConverter.mapTokenToIds(originalSequence);
        // statistics
        double parseSuccessBase;
        double reconstructionBase;
        List<CompilationError> compilationErrorsBase;
        List<RecoveryOperation> recoveryOperations;
        double parseSuccess;
        double reconstruction;
        // parse noisy
        JavaLexer baseLexer = new JavaLexer(CharStreams.fromString(decodedInput));
        CommonTokenStream tokens = new CommonTokenStream(baseLexer);
        tokens.fill();
        CustomErrorStrategy baseStrategy = new CustomErrorStrategy();
        JavaParser baseParser = new JavaParser(tokens);
        ErrorReporterVisitor baseVisitor = new ErrorReporterVisitor();
        baseParser.removeErrorListeners();
        baseParser.setErrorHandler(baseStrategy);
        JavaErrorListener baseListener = new JavaErrorListener(tokens.getTokens(), encodedTarget);
        baseParser.addErrorListener(baseListener);
        baseVisitor.visit(baseParser.compilationUnit());
        int[] adaptedSequence = baseListener.getReconstructedSequence();
        String adaptedParseableSequence = ANTLRPlaceholderToken.replaceSourceWithDummyTokens(adaptedSequence);
        // evaluation
        recoveryOperations = baseListener.getOperations();
        parseSuccessBase = TokenUtils.computeSuccessfulCompilationPercentage(
                baseVisitor.getErrorNodes(),
                baseVisitor.getVisitedNodes()
        );
        reconstructionBase = TokenUtils.computeSequenceEquality(
                encodedInput,
                encodedTarget
        );
        compilationErrorsBase = baseListener.getCompilationErrorList();
        reconstruction = TokenUtils.computeSequenceEquality(
                adaptedSequence,
                encodedTarget
        );
        // parse updated sequence
        if (compilationErrorsBase.isEmpty()) {
            // early return if no error nodes
            return null;
        }
        JavaLexer lexer = new JavaLexer(CharStreams.fromString(adaptedParseableSequence));
        tokens = new CommonTokenStream(lexer);
        tokens.fill();
        CustomErrorStrategy errorStrategy = new CustomErrorStrategy();
        JavaParser parser = new JavaParser(tokens);
        parser.removeErrorListeners();
        parser.setErrorHandler(errorStrategy);
        JavaErrorListener errorListener = new JavaErrorListener(tokens.getTokens(), encodedTarget);
        parser.addErrorListener(errorListener);
        ErrorReporterVisitor visitor = new ErrorReporterVisitor();
        visitor.visit(parser.compilationUnit());
        // evaluation
        parseSuccess = TokenUtils.computeSuccessfulCompilationPercentage(
                errorListener.getCompilationErrorList().size(),
                visitor.getVisitedNodes()
        );
        List<CompilationError> compilationErrors = errorListener.getCompilationErrorList();
        return new ANTLRStatistics(
                parseSuccessBase,
                reconstructionBase,
                compilationErrorsBase,
                parseSuccess,
                reconstruction,
                compilationErrors,
                recoveryOperations,
                original,
                noisy,
                ANTLRDataConverter.mapIdsToToken(adaptedSequence).split(" ")
        );
    }

    private DERStatistics parseWithOnFileDeepErrorRecovery(String noisySequence,
                                                           String originalSequence,
                                                           ANTLRStatistics baseStatistics) {

        String[] noisy = noisySequence.split(" ");

        if (baseStatistics == null) {
            // there is nothing to recover from
            return null;
        }
        // keeping track of parsing statistics
        double successfulParsePercentage = 0.0000;
        double reconstruction = 0.0000;
        List<CompilationError> compilationErrors;
        // encode ANTLR tokens to model tokens
        int[] noisyEncodedInput = ModelDataConverter.encodeSequence(noisySequence);
        int[] currentInput = ModelDataConverter.encodeSequence(noisySequence);
        // encode ANTLR tokens to IDs
        int[] originalInput = ModelDataConverter.encodeSequence(originalSequence);
        int[] antlrEncodedOriginalInput = ANTLRDataConverter.mapTokenToIds(originalSequence);
        int[] modelOutput = this.oracle.predictMostProbableTokens(getAllButLast(currentInput));
        // encode input sequence from model encoding to ANTLR encoding
        int[] antlrEncodedModelOutput = ANTLRModelConverter.encodeSequence(modelOutput);
        String parseableModelOutput = ANTLRPlaceholderToken.replaceSourceWithDummyTokens(ANTLRModelConverter.encodeSequence(modelOutput));
        // parse with DER
        JavaLexer lexer = new JavaLexer(CharStreams.fromString(parseableModelOutput));
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        tokens.fill();
        CustomErrorStrategy errorStrategy = new CustomErrorStrategy();
        JavaParser parser = new JavaParser(tokens);
        ErrorReporterVisitor visitor = new ErrorReporterVisitor();
        parser.removeErrorListeners();
        parser.setErrorHandler(errorStrategy);
        BaseJavaErrorListener errorListener = new BaseJavaErrorListener(tokens.getTokens());
        parser.addErrorListener(errorListener);
        // parse
        parser.compilationUnit();
        // evaluate
        successfulParsePercentage = TokenUtils.computeSuccessfulCompilationPercentage(
                visitor.getErrorNodes(),
                errorListener.getTotalNumberOfTokens()
        );
        reconstruction = TokenUtils.computeSequenceEquality(antlrEncodedModelOutput, antlrEncodedOriginalInput);
        compilationErrors = errorListener.getCompilationErrorList();
        return new DERStatistics(
                successfulParsePercentage,
                reconstruction,
                compilationErrors,
                new ArrayList<>(),
                ANTLRDataConverter.mapIdsToToken(antlrEncodedModelOutput).split(" ")
        );
    }

    private DERStatistics parseWithDeepErrorRecovery(String noisySequence,
                                                     String originalSequence,
                                                     ANTLRStatistics baseStatistics) {

        String[] noisy = noisySequence.split(" ");

        if (baseStatistics == null) {
            // there is nothing to recover from
            return null;
        }
        // keeping track of parsing statistics
        double successfulParsePercentage = 0.0000;
        double reconstruction = 0.0000;
        List<CompilationError> compilationErrors;
        List<RecoveryOperation>recoveryOperations;
        // encode ANTLR tokens to model tokens
        int[] antlrEncodedNoisy = ANTLRDataConverter.mapTokenToIds(noisySequence);
        int[] noisyEncodedInput = ModelDataConverter.encodeSequence(noisySequence);
        int[] currentInput = ModelDataConverter.encodeSequence(noisySequence);
        // encode ANTLR tokens to IDs
        int[] originalInput = ModelDataConverter.encodeSequence(originalSequence);
        int[] antlrEncodedOriginalInput = ANTLRDataConverter.mapTokenToIds(originalSequence);
        int[] modelOutput = this.oracle.predictMostProbableTokens(getAllButLast(currentInput));
        // encode input sequence from model encoding to ANTLR encoding
        int[] antlrEncodedModelOutput = ANTLRModelConverter.encodeSequence(modelOutput);
        String parseableModelOutput = ANTLRPlaceholderToken.replaceSourceWithDummyTokens(antlrEncodedModelOutput);
        String parseableInput = ANTLRPlaceholderToken.replaceSourceWithDummyTokens(
                ANTLRDataConverter.mapTokenToIds(noisySequence));
        // parse with DER
        JavaLexer lexer = new JavaLexer(CharStreams.fromString(parseableInput));
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        tokens.fill();
        JavaLexer modelLexer = new JavaLexer(CharStreams.fromString(parseableModelOutput));
        CommonTokenStream modelTokens = new CommonTokenStream(modelLexer);
        modelTokens.fill();
        // TODO: Experimental
        //FullTokenDeepErrorHandler handler = new FullTokenDeepErrorHandler(tokens.getTokens(), baseTokenstream.getTokens());//new DeepErrorRecoveryHandler(tokens.getTokens(), antlrEncodedModelOutput);
        //List<Token> onErrorStream = handler.reconcileErrorNodes(visitor.errorNodes);//handler.reconcileErrorNodes(visitor.errorNodes);
        DeepErrorRecoveryHandler2 handler = new DeepErrorRecoveryHandler2(tokens.getTokens(), modelTokens.getTokens(), antlrEncodedOriginalInput);
        //  int[] onErrorOutput = handler.reconcileErrorNodes(visitor.errorNodes);
        handler.reconcileCompilationErrors(baseStatistics.getBaseCompilationErrors());

        int[] onErrorOutput = Arrays.stream(handler.applyOperations())
        //int[] onErrorOutput = onErrorStream.stream()
        // filter out WS && EOF
                .filter(t -> t!= 125 && t != -1).toArray();
               //.toList().stream().mapToInt(Token::getTokenIndex).toArray();
         //take over partial model prediction at point of error

        // parse adapted sequence
        String modifiedParsableInput = ANTLRPlaceholderToken.replaceSourceWithDummyTokens(onErrorOutput);
        JavaLexer baseLexer = new JavaLexer(CharStreams.fromString(modifiedParsableInput));
        CommonTokenStream tokensWithFeedback = new CommonTokenStream(baseLexer);
        tokensWithFeedback.fill();
        CustomErrorStrategy baseErrorStrategy = new CustomErrorStrategy();
        JavaParser baseParser = new JavaParser(tokensWithFeedback);
        ErrorReporterVisitor baseVisitor = new ErrorReporterVisitor();
        baseParser.removeErrorListeners();
        baseParser.setErrorHandler(baseErrorStrategy);
        BaseJavaErrorListener baseErrorListener= new BaseJavaErrorListener(tokens.getTokens());
        baseParser.addErrorListener(baseErrorListener);
        baseVisitor.visit(baseParser.compilationUnit());
        // evaluate
        recoveryOperations = handler.getOperations();
        successfulParsePercentage = TokenUtils.computeSuccessfulCompilationPercentage(
                baseVisitor.getErrorNodes(),
                baseErrorListener.getTotalNumberOfTokens()
        );
        reconstruction = TokenUtils.computeSequenceEquality(onErrorOutput, antlrEncodedOriginalInput);
        compilationErrors = baseErrorListener.getCompilationErrorList();
        return new DERStatistics(
                successfulParsePercentage,
                reconstruction,
                compilationErrors,
                recoveryOperations,
                ANTLRDataConverter.mapIdsToToken(onErrorOutput).split(" ")
        );
    }

    public static int[] getAllButLast(int[] array) {
        if (array == null || array.length <= 1) {
            return new int[0]; // Return an empty array if the input is null or has only one element
        }

        int[] result = new int[array.length - 1];
        System.arraycopy(array, 0, result, 0, array.length - 1);

        return result;
    }

    public static int[] getAllButFirst(int[] array) {
        if (array == null || array.length <= 1) {
            return new int[0]; // Return an empty array if the input is null or has only one element
        }

        int[] result = new int[array.length - 1];
        System.arraycopy(array, 1, result, 0, array.length - 1);
        return result;
    }
}