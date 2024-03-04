package antlr.evaluation;

import java.util.List;

public class ParseStatistics {

    private double baseSuccessfulParsePercentage;
    private List<CompilationError> baseCompilationErrors;
    private double baseReconstruction;

    private double antlrParsePercentage;
    private double antlrReconstruction;
    private List<CompilationError> antlrCompilationErrors;
    private List<RecoveryOperation> antlrRecoveryOperations;

    private double derOnFileSuccessfulParsePercentage;
    private double derOnErrorSuccessfulParsePercentage;
    private List<RecoveryOperation> derOnErrorRecoveryOperations;
    private double derOnFileReconstruction;
    private double derOnErrorReconstruction;
    private List<CompilationError> derOnErrorCompilationErrors;
    private List<CompilationError> derOnFileCompilationErrors;


    public static ParseStatistics finalizeStatistics(ANTLRStatistics antlrStats,
                                                     DERStatistics onErrorStats,
                                                     DERStatistics experimentalStats) {
        ParseStatistics ps = new ParseStatistics();
        // baseline
        ps.setBaseSuccessfulParsePercentage(antlrStats.getBaseParsePercentage());
        ps.setBaseCompilationErrors(antlrStats.getBaseCompilationErrors());
        ps.setBaseReconstruction(antlrStats.getBaseReconstruction());
        // ANTLR ER on-error
        ps.setAntlrParsePercentage(antlrStats.getParsePercentage());
        ps.setAntlrReconstruction(antlrStats.getReconstruction());
        ps.setAntlrCompilationErrors(antlrStats.getCompilationErrors());
        ps.setAntlrRecoveryOperations(antlrStats.getRecoveryOperations());
        // Deep ER on-error with ER
        ps.setDerOnErrorReconstruction(onErrorStats.getReconstruction());
        ps.setDerOnErrorSuccessfulParsePercentage(onErrorStats.getParsePercentage());
        ps.setDerOnErrorCompilationErrors(onErrorStats.getCompilationErrors());
        ps.setDerOnErrorRecoveryOperations(onErrorStats.getRecoveryOperations());
        // Deep ER on-file with ER
        ps.setDerOnFileReconstruction(experimentalStats.getReconstruction());
        ps.setDerOnFileSuccessfulParsePercentage(experimentalStats.getParsePercentage());
        ps.setDerOnFileCompilationErrors(experimentalStats.getCompilationErrors());
        return ps;
    }


    public double getBaseSuccessfulParsePercentage() {
        return baseSuccessfulParsePercentage;
    }

    public void setBaseSuccessfulParsePercentage(double baseSuccessfulParsePercentage) {
        this.baseSuccessfulParsePercentage = baseSuccessfulParsePercentage;
    }

    public List<CompilationError> getBaseCompilationErrors() {
        return baseCompilationErrors;
    }

    public void setBaseCompilationErrors(List<CompilationError> baseCompilationErrors) {
        this.baseCompilationErrors = baseCompilationErrors;
    }

    public double getBaseReconstruction() {
        return baseReconstruction;
    }

    public void setBaseReconstruction(double baseReconstruction) {
        this.baseReconstruction = baseReconstruction;
    }

    public double getAntlrParsePercentage() {
        return antlrParsePercentage;
    }

    public void setAntlrParsePercentage(double antlrParsePercentage) {
        this.antlrParsePercentage = antlrParsePercentage;
    }

    public double getAntlrReconstruction() {
        return antlrReconstruction;
    }

    public void setAntlrReconstruction(double antlrReconstruction) {
        this.antlrReconstruction = antlrReconstruction;
    }

    public double getDerOnFileSuccessfulParsePercentage() {
        return derOnFileSuccessfulParsePercentage;
    }

    public void setDerOnFileSuccessfulParsePercentage(double derOnFileSuccessfulParsePercentage) {
        this.derOnFileSuccessfulParsePercentage = derOnFileSuccessfulParsePercentage;
    }

    public double getDerOnErrorSuccessfulParsePercentage() {
        return derOnErrorSuccessfulParsePercentage;
    }

    public void setDerOnErrorSuccessfulParsePercentage(double derOnErrorSuccessfulParsePercentage) {
        this.derOnErrorSuccessfulParsePercentage = derOnErrorSuccessfulParsePercentage;
    }
    public double getDerOnFileReconstruction() {
        return derOnFileReconstruction;
    }

    public void setDerOnFileReconstruction(double derOnFileReconstruction) {
        this.derOnFileReconstruction = derOnFileReconstruction;
    }

    public double getDerOnErrorReconstruction() {
        return derOnErrorReconstruction;
    }

    public void setDerOnErrorReconstruction(double derOnErrorReconstruction) {
        this.derOnErrorReconstruction = derOnErrorReconstruction;
    }

    public List<CompilationError> getDerOnErrorCompilationErrors() {
        return derOnErrorCompilationErrors;
    }

    public void setDerOnErrorCompilationErrors(List<CompilationError> derOnErrorCompilationErrors) {
        this.derOnErrorCompilationErrors = derOnErrorCompilationErrors;
    }

    public List<CompilationError> getDerOnFileCompilationErrors() {
        return derOnFileCompilationErrors;
    }

    public void setDerOnFileCompilationErrors(List<CompilationError> derOnFileCompilationErrors) {
        this.derOnFileCompilationErrors = derOnFileCompilationErrors;
    }

    public List<CompilationError> getAntlrCompilationErrors() {
        return antlrCompilationErrors;
    }

    public void setAntlrCompilationErrors(List<CompilationError> antlrCompilationErrors) {
        this.antlrCompilationErrors = antlrCompilationErrors;
    }

    public List<RecoveryOperation> getAntlrRecoveryOperations() {
        return antlrRecoveryOperations;
    }

    public void setAntlrRecoveryOperations(List<RecoveryOperation> antlrRecoveryOperations) {
        this.antlrRecoveryOperations = antlrRecoveryOperations;
    }

    public List<RecoveryOperation> getDerOnErrorRecoveryOperations() {
        return derOnErrorRecoveryOperations;
    }

    public void setDerOnErrorRecoveryOperations(List<RecoveryOperation> derOnErrorRecoveryOperations) {
        this.derOnErrorRecoveryOperations = derOnErrorRecoveryOperations;
    }
}
