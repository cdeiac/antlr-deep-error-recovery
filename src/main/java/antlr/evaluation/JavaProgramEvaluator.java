package antlr.evaluation;

import antlr.JavaLexer;
import antlr.JavaParser;
import antlr.converters.ANTLRDataConverter;
import antlr.converters.ANTLRModelConverter;
import antlr.converters.ANTLRPlaceholderToken;
import antlr.converters.ModelDataConverter;
import antlr.errorrecovery.DeepErrorRecoveryHandler;
import antlr.errorrecovery.JavaOracle;
import antlr.extensions.CustomErrorStrategy;
import antlr.utils.FileUtils;
import antlr.utils.TestData;
import antlr.utils.TokenUtils;
import org.antlr.v4.runtime.BailErrorStrategy;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.misc.ParseCancellationException;

import java.util.ArrayList;
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
        this.parseAndEvaluateAllOn(cvDir, noisyData, originalData);
        this.oracle.closePredictor();
    }
    // TODO: private
    public void parseAndEvaluateAllOn(String cvDir,
                                       List<String> noisySequences,
                                       List<String> originalSequences) {

        List<ParseStatistics> parseStatistics = new ArrayList<>();
        for (int i = 0; i < noisySequences.size(); i++) {
            String original = originalSequences.get(i);
            String noisy = noisySequences.get(i);
            // parse
            ANTLRStatistics antlrBailStats = this.parseWithAntlrBail(noisy, original); // ON_ERROR only
            ANTLRStatistics antlrStats = this.parseWithAntlrEr(noisy, original); // ON_ERROR only
            DERStatistics derBailStats = this.parseWithDeepErrorRecoveryWithBail(noisy, original, antlrBailStats);
            DERStatistics derStats = this.parseWithDeepErrorRecovery(noisy, original, antlrStats);
            // aggregate
            ParseStatistics finalStats = ParseStatistics.finalizeStatistics(
                    antlrBailStats,
                    antlrStats,
                    derBailStats,
                    derStats);
            parseStatistics.add(finalStats);
        }
        System.out.println("Base Bail Comp. Errors: " + parseStatistics.stream().mapToInt(s -> s.getBaseCompilationErrorsWithBail().size()).average().orElse(-1));
        System.out.println("Base Bail Reconstruction: " + parseStatistics.stream().map(ParseStatistics::getBaseReconstructionWithBail).toList().stream().mapToDouble(Double::doubleValue).average().orElse(-1));
        System.out.println("Base Comp. Errors: " + parseStatistics.stream().mapToInt(s -> s.getBaseCompilationErrors().size()).average().orElse(-1));
        System.out.println("Base Reconstruction: " + parseStatistics.stream().map(ParseStatistics::getBaseReconstruction).toList().stream().mapToDouble(Double::doubleValue).average().orElse(-1));
        System.out.println("ANTLR Bail On-Error Avg. Comp. Errors: " + parseStatistics.stream().mapToInt(s -> s.getAntlrCompilationErrorsWithBail().size()).average().orElse(-1));
        System.out.println("ANTLR Bail On-Error Avg. Reconstruction: " + parseStatistics.stream().map(ParseStatistics::getAntlrReconstructionWithBail).toList().stream().mapToDouble(Double::doubleValue).average().orElse(-1));
        System.out.println("ANTLR ER On-Error Avg. Comp. Errors: " + parseStatistics.stream().mapToInt(s -> s.getAntlrCompilationErrors().size()).average().orElse(-1));
        System.out.println("ANTLR ER On-Error Avg. Reconstruction: " + parseStatistics.stream().map(ParseStatistics::getAntlrReconstruction).toList().stream().mapToDouble(Double::doubleValue).average().orElse(-1));
        System.out.println("DER Bail Avg. Comp. Errors: " + parseStatistics.stream().mapToInt(s -> s.getDerOnErrorCompilationErrorsWithBail().size()).average().orElse(-1));
        System.out.println("DER Bail Avg. Reconstruction: " + parseStatistics.stream().map(ParseStatistics::getDerOnErrorReconstructionWithBail).toList().stream().mapToDouble(Double::doubleValue).average().orElse(-1));
        System.out.println("DER Avg. Comp. Errors: " + parseStatistics.stream().mapToInt(s -> s.getDerOnErrorCompilationErrors().size()).average().orElse(-1));
        System.out.println("DER Avg. Reconstruction: " + parseStatistics.stream().map(ParseStatistics::getDerOnErrorReconstruction).toList().stream().mapToDouble(Double::doubleValue).average().orElse(-1));
        FileUtils.saveParseStatistics(cvDir, cvIter, parseStatistics);
    }

    private ANTLRStatistics parseWithAntlrBail(String noisySequence, String originalSequence) {
        // decode sequence
        int[] encodedInput = ANTLRDataConverter.mapTokenToIds(noisySequence);
        String decodedInput = ANTLRPlaceholderToken.replaceSourceWithDummyTokens(encodedInput);
        int[] encodedTarget = ANTLRDataConverter.mapTokenToIds(originalSequence);
        // statistics
        double baseParseSuccess = 1.0000;
        double baseReconstruction = 1.0000;
        double parseSuccess = 1.0000;
        double reconstruction = 1.0000;
        // prepare parser without ER
        JavaLexer bailLexer = new JavaLexer(CharStreams.fromString(decodedInput));
        CommonTokenStream bailTokens = new CommonTokenStream(bailLexer);
        bailTokens.fill();
        JavaParser bailParser = new JavaParser(bailTokens);
        bailParser.removeErrorListeners();
        BailErrorStrategy bailErrorStrategy = new BailErrorStrategy();
        bailParser.setErrorHandler(bailErrorStrategy);
        JavaBailErrorListener bailListener = new JavaBailErrorListener(bailTokens.getTokens(), true);
        bailParser.addErrorListener(bailListener);
        ErrorReporterVisitor bailVisitor = new ErrorReporterVisitor();
        try {
            bailVisitor.visit(bailParser.compilationUnit());
        } catch (ParseCancellationException e) {
            baseParseSuccess = TokenUtils.computeSuccessfulCompilationPercentageWithBail(
                    //bailVisitor.errorNodes.isEmpty() ? 0 : (bailVisitor.getVisitedNodes() - bailListener.getBailedPosition(bailVisitor.errorNodes.get(0))),
                    bailListener.getBailedPosition(bailParser.getCurrentToken()),
                    bailListener.getReconstructedSequence().length
            );
        }
        // evaluation
        int[] adaptedSequence = bailListener.getReconstructedSequence();
        baseReconstruction = TokenUtils.computeSequenceEquality(
                encodedInput,
                encodedTarget
        );
        reconstruction = TokenUtils.computeSequenceEquality(
                adaptedSequence,
                encodedTarget
        );
        if (bailListener.getCompilationErrorList().isEmpty()) {
            // no exception was caught: early return
            return new ANTLRStatistics(
                    baseParseSuccess,
                    baseReconstruction,
                    bailListener.getCompilationErrorList(),
                    baseParseSuccess,
                    reconstruction,
                    bailListener.getCompilationErrorList()
            );
        }
        String adaptedParseableSequence = ANTLRPlaceholderToken.replaceSourceWithDummyTokens(adaptedSequence);
        JavaLexer lexer = new JavaLexer(CharStreams.fromString(adaptedParseableSequence));
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        tokens.fill();
        JavaParser parser = new JavaParser(tokens);
        BailErrorStrategy customErrorStrategy = new BailErrorStrategy();
        ErrorReporterVisitor visitor = new ErrorReporterVisitor();
        parser.removeErrorListeners();
        parser.setErrorHandler(customErrorStrategy);
        JavaBailErrorListener listener = new JavaBailErrorListener(tokens.getTokens(), true);
        parser.addErrorListener(listener);
        // parse
        try {
            visitor.visit(parser.compilationUnit());
        } catch (ParseCancellationException e) {
            parseSuccess = TokenUtils.computeSuccessfulCompilationPercentageWithBail(
                    listener.getBailedPosition(bailParser.getCurrentToken()),
                    listener.getReconstructedSequence().length
            );
        }
        return new ANTLRStatistics(
                baseParseSuccess,
                baseReconstruction,
                bailListener.getCompilationErrorList(),
                parseSuccess,
                reconstruction,
                listener.getCompilationErrorList()
        );
    }

    private DERStatistics parseWithDeepErrorRecovery(String noisySequence,
                                                     String originalSequence,
                                                     ANTLRStatistics baseStatistics) {

        if (baseStatistics.getBaseCompilationErrors().isEmpty()) {
            // there is nothing to recover from
            return new DERStatistics(
                    baseStatistics.getBaseParsePercentage(),
                    baseStatistics.getBaseReconstruction(),
                    baseStatistics.getBaseCompilationErrors()
            );
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
        String parseableInput = ANTLRPlaceholderToken.replaceSourceWithDummyTokens(
                ANTLRDataConverter.mapTokenToIds(noisySequence));
        // parse with DER
        JavaLexer lexer = new JavaLexer(CharStreams.fromString(parseableInput));
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        tokens.fill();
        CustomErrorStrategy errorStrategy = new CustomErrorStrategy();
        JavaParser parser = new JavaParser(tokens);
        ErrorReporterVisitor visitor = new ErrorReporterVisitor();
        parser.removeErrorListeners();
        parser.setErrorHandler(errorStrategy);
        DeepJavaErrorListener errorListener = new DeepJavaErrorListener(tokens.getTokens(), antlrEncodedModelOutput);
        parser.addErrorListener(errorListener);
        // parse
        visitor.visit(parser.compilationUnit());
        // adapt input for feedback
        //JavaLexer baseLexer = new JavaLexer(CharStreams.fromString(parseableModelOutput));
        //CommonTokenStream baseTokenstream = new CommonTokenStream(baseLexer);
        //baseTokenstream.fill();
        // TODO: Experimental
        //FullTokenDeepErrorHandler handler = new FullTokenDeepErrorHandler(tokens.getTokens(), baseTokenstream.getTokens());//new DeepErrorRecoveryHandler(tokens.getTokens(), antlrEncodedModelOutput);
        //List<Token> onErrorStream = handler.reconcileErrorNodes(visitor.errorNodes);//handler.reconcileErrorNodes(visitor.errorNodes);
        DeepErrorRecoveryHandler handler = new DeepErrorRecoveryHandler(tokens.getTokens(), antlrEncodedModelOutput, false);
        int[] onErrorOutput = handler.reconcileErrorNodes(visitor.errorNodes);
        //int[] onErrorOutput = onErrorStream.stream()
        // filter out WS && EOF
        //        .filter(t -> t.getType() != 125 && t.getType() != -1)
        //       .toList().stream().mapToInt(Token::getTokenIndex).toArray();
        // take over partial model prediction at point of error

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
        successfulParsePercentage = TokenUtils.computeSuccessfulCompilationPercentage(
                baseVisitor.getErrorNodes(),
                baseErrorListener.getTotalNumberOfTokens()
        );
        reconstruction = TokenUtils.computeSequenceEquality(onErrorOutput, antlrEncodedOriginalInput);
        compilationErrors = errorListener.getCompilationErrorList();
        return new DERStatistics(successfulParsePercentage, reconstruction, compilationErrors);
    }

    private DERStatistics parseWithDeepErrorRecoveryWithBail(String noisySequence,
                                                             String originalSequence,
                                                             ANTLRStatistics baseStatistics) {

        if (baseStatistics.getBaseCompilationErrors().isEmpty()) {
            // there is nothing to recover from
            return new DERStatistics(
                    baseStatistics.getBaseParsePercentage(),
                    baseStatistics.getBaseReconstruction(),
                    baseStatistics.getBaseCompilationErrors()
            );
        }
        // keeping track of parsing statistics
        double successfulParsePercentage = 0.0000;
        double reconstruction = 0.0000;
        List<CompilationError> compilationErrors = new ArrayList<>();
        // encode ANTLR tokens to model tokens
        int[] noisyEncodedInput = ModelDataConverter.encodeSequence(noisySequence);
        int[] currentInput = ModelDataConverter.encodeSequence(noisySequence);
        // encode ANTLR tokens to IDs
        int[] originalInput = ModelDataConverter.encodeSequence(originalSequence);
        int[] antlrEncodedOriginalInput = ANTLRDataConverter.mapTokenToIds(originalSequence);
        int[] modelOutput = this.oracle.predictMostProbableTokens(getAllButLast(currentInput)); // currentInput
        // encode input sequence from model encoding to ANTLR encoding
        int[] antlrEncodedModelOutput = ANTLRModelConverter.encodeSequence(modelOutput);
        String parseableModelOutput = ANTLRPlaceholderToken.replaceSourceWithDummyTokens(ANTLRModelConverter.encodeSequence(modelOutput));
        String parseableInput = ANTLRPlaceholderToken.replaceSourceWithDummyTokens(
                ANTLRDataConverter.mapTokenToIds(noisySequence));
        // prepare parser with Bail
        JavaLexer lexer = new JavaLexer(CharStreams.fromString(parseableInput));
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        tokens.fill();
        //CustomErrorStrategy errorStrategy = new CustomErrorStrategy();
        BailErrorStrategy errorStrategy = new BailErrorStrategy();
        JavaParser parser = new JavaParser(tokens);
        ErrorReporterVisitor bailVisitor = new ErrorReporterVisitor();
        parser.removeErrorListeners();
        parser.setErrorHandler(errorStrategy);
        DeepJavaErrorListener errorListener = new DeepJavaErrorListener(tokens.getTokens(), antlrEncodedModelOutput);
        parser.addErrorListener(errorListener);
        int[] onErrorOutput = new int[]{};
        // parse
        try {
            bailVisitor.visit(parser.compilationUnit());
        } catch (ParseCancellationException e) {
            DeepErrorRecoveryHandler handler = new DeepErrorRecoveryHandler(tokens.getTokens(), antlrEncodedModelOutput, true);
            onErrorOutput = handler.reconcileTokens(List.of(parser.getCurrentToken()));
        }
        // we already know that the input has compilation errors
        if (onErrorOutput.length == 0) {
            throw new RuntimeException("On-Error Output is empty!");
        }
        // adapt input for feedback

        //JavaLexer baseLexer = new JavaLexer(CharStreams.fromString(parseableModelOutput));
        //CommonTokenStream baseTokenstream = new CommonTokenStream(baseLexer);
        //baseTokenstream.fill();

        // TODO: Experimental
        //DeepErrorRecoveryHandler handler = new DeepErrorRecoveryHandler(tokens.getTokens(), antlrEncodedModelOutput, true);
        //int[] onErrorOutput = handler.reconcileErrorNodes(List.of(bailVisitor.errorNodes.get(0)));


        //FullTokenDeepErrorHandler handler = new FullTokenDeepErrorHandler(tokens.getTokens(), baseTokenstream.getTokens());//new DeepErrorRecoveryHandler(tokens.getTokens(), antlrEncodedModelOutput);
        //List<Token> onErrorStream = handler.reconcileErrorNodes(visitor.errorNodes);//handler.reconcileErrorNodes(visitor.errorNodes);

        //int[] onErrorOutput = onErrorStream.stream()
        // filter out WS && EOF
        //        .filter(t -> t.getType() != 125 && t.getType() != -1)
        //       .toList().stream().mapToInt(Token::getTokenIndex).toArray();
        // take over partial model prediction at point of error

        // parse adapted sequence
        String modifiedParsableInput = ANTLRPlaceholderToken.replaceSourceWithDummyTokens(onErrorOutput);
        // bail
        JavaLexer bailLexer = new JavaLexer(CharStreams.fromString(modifiedParsableInput));
        CommonTokenStream bailTokens = new CommonTokenStream(bailLexer);
        bailTokens.fill();
        JavaParser bailParser = new JavaParser(bailTokens);
        bailParser.removeErrorListeners();
        bailLexer.removeErrorListeners();
        BailErrorStrategy bailErrorStrategy = new BailErrorStrategy();
        bailParser.setErrorHandler(bailErrorStrategy);
        JavaBailErrorListener bailErrorListener = new JavaBailErrorListener(tokens.getTokens(), false);
        bailParser.addErrorListener(bailErrorListener);
        ErrorReporterVisitor visitor = new ErrorReporterVisitor();
        try {
            visitor.visit(bailParser.compilationUnit());
        } catch (ParseCancellationException e) {
            successfulParsePercentage = TokenUtils.computeSuccessfulCompilationPercentageWithBail(
                    bailErrorListener.getBailedPosition(bailParser.getCurrentToken()),
                    bailErrorListener.getReconstructedSequence().length
            );
            reconstruction = TokenUtils.computeSequenceEquality(
                    onErrorOutput,
                    antlrEncodedOriginalInput
            );
            compilationErrors = bailErrorListener.getCompilationErrorList();
        }


        //int errorIndex = bailErrorListener.getErrorTokenPositionOfTokenStream(bailParser.getCurrentToken());
        //if (errorIndex < 0) {
        //    errorIndex = bailErrorListener.findPreviousIndex(bailParser.getCurrentToken().getTokenIndex());
        //}
        /*
        successfulParsePercentage = TokenUtils.computeSuccessfulCompilationPercentage(
                errorIndex,
                bailErrorListener.getTotalNumberOfTokens()
        );
        reconstruction = TokenUtils.computeSequenceEquality(
                errorIndex,
                currentInput,
                antlrEncodedOriginalInput
        );*/
        return new DERStatistics(
                successfulParsePercentage, reconstruction, compilationErrors
        );
    }

    private ANTLRStatistics parseWithAntlrEr(String noisySequence, String originalSequence) {
        // decode sequence
        int[] encodedInput = ANTLRDataConverter.mapTokenToIds(noisySequence);
        String decodedInput = ANTLRPlaceholderToken.replaceSourceWithDummyTokens(encodedInput);
        int[] encodedTarget = ANTLRDataConverter.mapTokenToIds(originalSequence);
        // statistics
        double parseSuccessBase;
        double reconstructionBase;
        List<CompilationError> compilationErrorsBase;
        double parseSuccess = 0.0000;
        double reconstruction = 0.0000;
        // parse noisy
        JavaLexer baseLexer = new JavaLexer(CharStreams.fromString(decodedInput));
        CommonTokenStream tokens = new CommonTokenStream(baseLexer);
        tokens.fill();
        CustomErrorStrategy baseStrategy = new CustomErrorStrategy();
        JavaParser baseParser = new JavaParser(tokens);
        ErrorReporterVisitor baseVisitor = new ErrorReporterVisitor();
        baseParser.removeErrorListeners();
        baseParser.setErrorHandler(baseStrategy);
        JavaErrorListener baseListener = new JavaErrorListener(tokens.getTokens(), true);
        baseParser.addErrorListener(baseListener);
        baseVisitor.visit(baseParser.compilationUnit());
        int[] adaptedSequence = baseListener.getReconstructedSequence();
        String adaptedParseableSequence = ANTLRPlaceholderToken.replaceSourceWithDummyTokens(adaptedSequence);
        // evaluation
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
            return new ANTLRStatistics(
                    parseSuccessBase,
                    reconstructionBase,
                    compilationErrorsBase,
                    parseSuccessBase,
                    reconstruction,
                    compilationErrorsBase
            );
        }
        JavaLexer lexer = new JavaLexer(CharStreams.fromString(adaptedParseableSequence));
        tokens = new CommonTokenStream(lexer);
        tokens.fill();
        CustomErrorStrategy errorStrategy = new CustomErrorStrategy();
        JavaParser parser = new JavaParser(tokens);
        parser.removeErrorListeners();
        parser.setErrorHandler(errorStrategy);
        JavaErrorListener errorListener = new JavaErrorListener(tokens.getTokens(), false);
        parser.addErrorListener(errorListener);
        ErrorReporterVisitor visitor = new ErrorReporterVisitor();
        visitor.visit(parser.compilationUnit());
        // evaluation
        parseSuccess = TokenUtils.computeSuccessfulCompilationPercentage(
                visitor.getErrorNodes(),
                visitor.getVisitedNodes()
        );
        List<CompilationError> compilationErrors = errorListener.getCompilationErrorList();
        return new ANTLRStatistics(
                parseSuccessBase,
                reconstructionBase,
                compilationErrorsBase,
                parseSuccess,
                reconstruction,
                compilationErrors
        );
    }

    private ANTLRStatistics parseBaseline(String noisySequence, String originalSequence) {
        // decode sequence
        int[] encodedInput = ANTLRDataConverter.mapTokenToIds(noisySequence);
        String decodedInput = ANTLRPlaceholderToken.replaceSourceWithDummyTokens(encodedInput);
        int[] encodedTarget = ANTLRDataConverter.mapTokenToIds(originalSequence);
        // statistics
        double reconstruction = 0.0000;
        double reconstructionWithBail = -1.0000;
        double parseSuccess = 0.0000;
        double parseSuccessWithBail = -1.0000;
        // prepare parser without ER
        JavaLexer lexer = new JavaLexer(CharStreams.fromString(decodedInput));
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        tokens.fill();
        JavaParser parser = new JavaParser(tokens);
        parser.removeErrorListeners();
        parser.setErrorHandler(new BailErrorStrategy());
        BaseJavaErrorListener bailListener = new BaseJavaErrorListener(tokens.getTokens());
        parser.addErrorListener(bailListener);
        // parse
        try {
            parser.compilationUnit();
        } catch (ParseCancellationException e) {
            parseSuccessWithBail = TokenUtils.computeSuccessfulCompilationPercentageWithBail(
                    bailListener.getErrorTokenPositionOfTokenStream(parser.getCurrentToken()),
                    bailListener.getTotalNumberOfTokens()
            );
            reconstructionWithBail = TokenUtils.computeSequenceEqualityWithBail(
                    bailListener.getErrorTokenPositionOfTokenStream(parser.getCurrentToken()),
                    encodedInput,
                    encodedTarget
            );
        }
        // prepare parser with ER
        lexer = new JavaLexer(CharStreams.fromString(decodedInput));
        tokens = new CommonTokenStream(lexer);
        tokens.fill();
        CustomErrorStrategy errorStrategy = new CustomErrorStrategy();
        parser = new JavaParser(tokens);
        parser.removeErrorListeners();
        parser.setErrorHandler(errorStrategy);
        BaseJavaErrorListener errorListener = new BaseJavaErrorListener(tokens.getTokens());
        parser.addErrorListener(errorListener);
        parser.compilationUnit();
        // evaluation
        parseSuccess = TokenUtils.computeSuccessfulCompilationPercentage(
                errorListener.getCompilationErrorList().size(),
                errorListener.getTotalNumberOfTokens());
        reconstruction = TokenUtils.computeSequenceEquality(encodedInput, encodedTarget);
        if (parseSuccessWithBail < 0.0) {
            // no exception occurred
            parseSuccessWithBail = parseSuccess;
            reconstructionWithBail = reconstruction;
        } // TODO: Fix
        return null; //ParseStatistics.buildBaseStatisticsWithBail(
                //parseSuccess,
                //parseSuccessWithBail,
                //reconstruction,
                //reconstructionWithBail,
                //errorListener.getCompilationErrorList()
        //);
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

    /*
        private ParseStatistics parseWithANTLR(String noisySequence, String originalSequence) {
        // decode sequence
        int[] encodedInput = ANTLRDataConverter.mapTokenToIds(noisySequence);
        String decodedInput = ANTLRPlaceholderToken.replaceSourceWithDummyTokens(encodedInput);
        int[] encodedTarget = ANTLRDataConverter.mapTokenToIds(originalSequence);
        // statistics
        double parseSuccessWithBail = 1.0000;
        double reconstructionWithBail = 1.0000;
        // prepare parser without ER
        JavaLexer lexer = new JavaLexer(CharStreams.fromString(decodedInput));
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        tokens.fill();
        JavaParser parser = new JavaParser(tokens);
        parser.removeErrorListeners();
        parser.setErrorHandler(new BailErrorStrategy());
        BaseJavaErrorListener bailListener = new BaseJavaErrorListener(tokens.getTokens());
        parser.addErrorListener(bailListener);
        // parse
        try {
            parser.compilationUnit();
        } catch (ParseCancellationException e) {
            parseSuccessWithBail = TokenUtils.computeSuccessfulCompilationPercentageWithBail(
                    bailListener.getErrorTokenPositionOfTokenStream(parser.getCurrentToken()),
                    bailListener.getTotalNumberOfTokens()
            );
            //reconstructionWithBail = TokenUtils.computeSequenceEqualityWithBail(bailListener.getErrorTokenPositionOfTokenStream(parser.getCurrentToken()), encodedInput, encodedTarget);
            reconstructionWithBail = TokenUtils.computeSequenceEqualityWithBail(
                    bailListener.getErrorTokenPositionOfTokenStream(parser.getCurrentToken()),
                    bailListener.getReconstructedSequence(),
                    encodedTarget
            );
        }
        // prepare parser with ER
        lexer = new JavaLexer(CharStreams.fromString(decodedInput));
        tokens = new CommonTokenStream(lexer);
        tokens.fill();
        CustomErrorStrategy errorStrategy = new CustomErrorStrategy();
        parser = new JavaParser(tokens);
        parser.removeErrorListeners();
        parser.setErrorHandler(errorStrategy);
        BaseJavaErrorListener errorListener = new BaseJavaErrorListener(tokens.getTokens());
        parser.addErrorListener(errorListener);
        // parse
        parser.compilationUnit();
        return buildBaseParseStatistics(
                TokenUtils.computeSuccessfulCompilationPercentage(
                        errorListener.getCompilationErrorList().size(),
                        errorListener.getTotalNumberOfTokens()),
                parseSuccessWithBail,
                errorListener.getCompilationErrorList(),
                TokenUtils.computeSequenceEquality(errorListener.getReconstructedSequence(), encodedTarget),
                reconstructionWithBail
        );
    }

     */

}