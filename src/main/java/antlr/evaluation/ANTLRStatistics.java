package antlr.evaluation;

import java.util.List;

public class ANTLRStatistics {

    private double baseParsePercentage;
    private double baseReconstruction;
    private List<CompilationError> baseCompilationErrors;

    private double parsePercentage;
    private double reconstruction;
    private List<CompilationError> compilationErrors;
    private List<RecoveryOperation> recoveryOperations;
    private String[] original;
    private String[] noisy;
    private String[] prediction;


    public ANTLRStatistics(double baseParsePercentage,
                           double baseReconstruction,
                           List<CompilationError> baseCompilationErrors,
                           double parsePercentage,
                           double reconstruction,
                           List<CompilationError> compilationErrors,
                           List<RecoveryOperation> recoveryOperations,
                           String[] original,
                           String[] noisy,
                           String[] prediction) {

        this.baseParsePercentage = baseParsePercentage;
        this.baseReconstruction = baseReconstruction;
        this.baseCompilationErrors = baseCompilationErrors;
        this.parsePercentage = parsePercentage;
        this.reconstruction = reconstruction;
        this.compilationErrors = compilationErrors;
        this.recoveryOperations = recoveryOperations;
        this.original = original;
        this.noisy = noisy;
        this.prediction = prediction;
    }


    public double getBaseParsePercentage() {
        return baseParsePercentage;
    }

    public void setBaseParsePercentage(double baseParsePercentage) {
        this.baseParsePercentage = baseParsePercentage;
    }

    public double getBaseReconstruction() {
        return baseReconstruction;
    }

    public List<CompilationError> getBaseCompilationErrors() {
        return baseCompilationErrors;
    }

    public double getParsePercentage() {
        return parsePercentage;
    }

    public void setParsePercentage(double parsePercentage) {
        this.parsePercentage = parsePercentage;
    }

    public double getReconstruction() {
        return reconstruction;
    }

    public void setReconstruction(double reconstruction) {
        this.reconstruction = reconstruction;
    }

    public List<CompilationError> getCompilationErrors() {
        return compilationErrors;
    }

    public void setCompilationErrors(List<CompilationError> compilationErrors) {
        this.compilationErrors = compilationErrors;
    }

    public void setBaseReconstruction(double baseReconstruction) {
        this.baseReconstruction = baseReconstruction;
    }

    public void setBaseCompilationErrors(List<CompilationError> baseCompilationErrors) {
        this.baseCompilationErrors = baseCompilationErrors;
    }

    public String[] getOriginal() {
        return original;
    }

    public void setOriginal(String[] original) {
        this.original = original;
    }

    public String[] getNoisy() {
        return noisy;
    }

    public void setNoisy(String[] noisy) {
        this.noisy = noisy;
    }

    public String[] getPrediction() {
        return prediction;
    }

    public void setPrediction(String[] prediction) {
        this.prediction = prediction;
    }

    public List<RecoveryOperation> getRecoveryOperations() {
        return recoveryOperations;
    }

    public void setRecoveryOperations(List<RecoveryOperation> recoveryOperations) {
        this.recoveryOperations = recoveryOperations;
    }
}
