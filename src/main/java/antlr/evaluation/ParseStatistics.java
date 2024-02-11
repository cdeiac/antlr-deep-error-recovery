package antlr.evaluation;

import java.util.List;

public class ParseStatistics {

    private double baseSuccessfulParsePercentage;
    private double baseSuccessfulParsePercentageWithBail;
    private List<CompilationError> baseCompilationErrors;
    private List<CompilationError> baseCompilationErrorsWithBail;
    private double baseReconstruction;
    private double baseReconstructionWithBail;

    private double antlrParsePercentage;
    private double antlrParsePercentageWithBail;
    private double antlrReconstruction;
    private double antlrReconstructionWithBail;
    private List<CompilationError> antlrCompilationErrors;
    private List<CompilationError> antlrCompilationErrorsWithBail;

    private double derOnFileSuccessfulParsePercentage;
    private double derOnErrorSuccessfulParsePercentage;
    private double derOnFileSuccessfulParsePercentageWithBail;
    private double derOnErrorSuccessfulParsePercentageWithBail;
    private double derOnFileReconstruction;
    private double derOnFileReconstructionWithBail;
    private double derOnErrorReconstruction;
    private double derOnErrorReconstructionWithBail;
    private List<CompilationError> derOnErrorCompilationErrors;
    private List<CompilationError> derOnFileCompilationErrors;
    private List<CompilationError> derOnErrorCompilationErrorsWithBail;
    private List<CompilationError> derOnFileCompilationErrorsWithBail;


    public static ParseStatistics buildBailOnErrorParseStatistics(double successfulParsePercentage,
                                                                  double reconstruction) {

        ParseStatistics ps = new ParseStatistics();
        ps.setDerOnErrorSuccessfulParsePercentageWithBail(successfulParsePercentage);
        ps.setDerOnErrorReconstructionWithBail(reconstruction);
        // skip recovery attempts & compilation errors
        return ps;
    }

    public static ParseStatistics finalizeStatistics(ANTLRStatistics antlrBailStats,
                                                     ANTLRStatistics antlrStats,
                                                     DERStatistics onErrorStatsWithBail,
                                                     DERStatistics onErrorStats) {
        ParseStatistics ps = new ParseStatistics();
        // baseline
        ps.setBaseSuccessfulParsePercentage(antlrStats.getBaseParsePercentage());
        ps.setBaseSuccessfulParsePercentageWithBail(antlrBailStats.getBaseParsePercentage());
        ps.setBaseCompilationErrors(antlrStats.getBaseCompilationErrors());
        ps.setBaseCompilationErrorsWithBail(antlrBailStats.getBaseCompilationErrors());
        ps.setBaseReconstruction(antlrStats.getReconstruction());
        ps.setBaseReconstructionWithBail(antlrBailStats.getReconstruction());
        // ANTLR bail on-error
        ps.setAntlrParsePercentageWithBail(antlrBailStats.getReconstruction());
        ps.setAntlrReconstructionWithBail(antlrBailStats.getReconstruction());
        ps.setAntlrCompilationErrorsWithBail(antlrBailStats.getCompilationErrors());
        // ANTLR ER on-error
        ps.setAntlrParsePercentage(antlrStats.getParsePercentage());
        ps.setAntlrReconstruction(antlrStats.getReconstruction());
        ps.setAntlrCompilationErrors(antlrStats.getCompilationErrors());
        // Deep ER on-error with bail
        ps.setDerOnErrorReconstructionWithBail(onErrorStatsWithBail.getReconstruction());
        ps.setDerOnErrorSuccessfulParsePercentageWithBail(onErrorStatsWithBail.getParsePercentage());
        ps.setDerOnErrorCompilationErrorsWithBail(onErrorStatsWithBail.getCompilationErrors());
        // Deep ER on-error with ER
        ps.setDerOnErrorReconstruction(onErrorStats.getReconstruction());
        ps.setDerOnErrorSuccessfulParsePercentage(onErrorStats.getParsePercentage());
        ps.setDerOnErrorCompilationErrors(onErrorStats.getCompilationErrors());
        return ps;
    }


    public double getBaseSuccessfulParsePercentage() {
        return baseSuccessfulParsePercentage;
    }

    public void setBaseSuccessfulParsePercentage(double baseSuccessfulParsePercentage) {
        this.baseSuccessfulParsePercentage = baseSuccessfulParsePercentage;
    }

    public double getBaseSuccessfulParsePercentageWithBail() {
        return baseSuccessfulParsePercentageWithBail;
    }

    public void setBaseSuccessfulParsePercentageWithBail(double baseSuccessfulParsePercentageWithBail) {
        this.baseSuccessfulParsePercentageWithBail = baseSuccessfulParsePercentageWithBail;
    }

    public List<CompilationError> getBaseCompilationErrors() {
        return baseCompilationErrors;
    }

    public void setBaseCompilationErrors(List<CompilationError> baseCompilationErrors) {
        this.baseCompilationErrors = baseCompilationErrors;
    }

    public List<CompilationError> getBaseCompilationErrorsWithBail() {
        return baseCompilationErrorsWithBail;
    }

    public void setBaseCompilationErrorsWithBail(List<CompilationError> baseCompilationErrorsWithBail) {
        this.baseCompilationErrorsWithBail = baseCompilationErrorsWithBail;
    }

    public double getBaseReconstruction() {
        return baseReconstruction;
    }

    public void setBaseReconstruction(double baseReconstruction) {
        this.baseReconstruction = baseReconstruction;
    }

    public double getBaseReconstructionWithBail() {
        return baseReconstructionWithBail;
    }

    public void setBaseReconstructionWithBail(double baseReconstructionWithBail) {
        this.baseReconstructionWithBail = baseReconstructionWithBail;
    }

    public double getAntlrParsePercentage() {
        return antlrParsePercentage;
    }

    public void setAntlrParsePercentage(double antlrParsePercentage) {
        this.antlrParsePercentage = antlrParsePercentage;
    }

    public double getAntlrParsePercentageWithBail() {
        return antlrParsePercentageWithBail;
    }

    public void setAntlrParsePercentageWithBail(double antlrParsePercentageWithBail) {
        this.antlrParsePercentageWithBail = antlrParsePercentageWithBail;
    }

    public double getAntlrReconstruction() {
        return antlrReconstruction;
    }

    public void setAntlrReconstruction(double antlrReconstruction) {
        this.antlrReconstruction = antlrReconstruction;
    }

    public double getAntlrReconstructionWithBail() {
        return antlrReconstructionWithBail;
    }

    public void setAntlrReconstructionWithBail(double antlrReconstructionWithBail) {
        this.antlrReconstructionWithBail = antlrReconstructionWithBail;
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

    public double getDerOnFileSuccessfulParsePercentageWithBail() {
        return derOnFileSuccessfulParsePercentageWithBail;
    }

    public void setDerOnFileSuccessfulParsePercentageWithBail(double derOnFileSuccessfulParsePercentageWithBail) {
        this.derOnFileSuccessfulParsePercentageWithBail = derOnFileSuccessfulParsePercentageWithBail;
    }

    public double getDerOnErrorSuccessfulParsePercentageWithBail() {
        return derOnErrorSuccessfulParsePercentageWithBail;
    }

    public void setDerOnErrorSuccessfulParsePercentageWithBail(double derOnErrorSuccessfulParsePercentageWithBail) {
        this.derOnErrorSuccessfulParsePercentageWithBail = derOnErrorSuccessfulParsePercentageWithBail;
    }

    public double getDerOnFileReconstruction() {
        return derOnFileReconstruction;
    }

    public void setDerOnFileReconstruction(double derOnFileReconstruction) {
        this.derOnFileReconstruction = derOnFileReconstruction;
    }

    public double getDerOnFileReconstructionWithBail() {
        return derOnFileReconstructionWithBail;
    }

    public void setDerOnFileReconstructionWithBail(double derOnFileReconstructionWithBail) {
        this.derOnFileReconstructionWithBail = derOnFileReconstructionWithBail;
    }

    public double getDerOnErrorReconstruction() {
        return derOnErrorReconstruction;
    }

    public void setDerOnErrorReconstruction(double derOnErrorReconstruction) {
        this.derOnErrorReconstruction = derOnErrorReconstruction;
    }

    public double getDerOnErrorReconstructionWithBail() {
        return derOnErrorReconstructionWithBail;
    }

    public void setDerOnErrorReconstructionWithBail(double derOnErrorReconstructionWithBail) {
        this.derOnErrorReconstructionWithBail = derOnErrorReconstructionWithBail;
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

    public List<CompilationError> getDerOnErrorCompilationErrorsWithBail() {
        return derOnErrorCompilationErrorsWithBail;
    }

    public void setDerOnErrorCompilationErrorsWithBail(List<CompilationError> derOnErrorCompilationErrorsWithBail) {
        this.derOnErrorCompilationErrorsWithBail = derOnErrorCompilationErrorsWithBail;
    }

    public List<CompilationError> getDerOnFileCompilationErrorsWithBail() {
        return derOnFileCompilationErrorsWithBail;
    }

    public void setDerOnFileCompilationErrorsWithBail(List<CompilationError> derOnFileCompilationErrorsWithBail) {
        this.derOnFileCompilationErrorsWithBail = derOnFileCompilationErrorsWithBail;
    }

    public List<CompilationError> getAntlrCompilationErrors() {
        return antlrCompilationErrors;
    }

    public void setAntlrCompilationErrors(List<CompilationError> antlrCompilationErrors) {
        this.antlrCompilationErrors = antlrCompilationErrors;
    }

    public List<CompilationError> getAntlrCompilationErrorsWithBail() {
        return antlrCompilationErrorsWithBail;
    }

    public void setAntlrCompilationErrorsWithBail(List<CompilationError> antlrCompilationErrorsWithBail) {
        this.antlrCompilationErrorsWithBail = antlrCompilationErrorsWithBail;
    }
}
