package antlr.evaluation;

import java.util.List;

public class DERStatistics {

    private double parsePercentage;
    private double reconstruction;
    private List<CompilationError> compilationErrors;
    private List<RecoveryOperation> recoveryOperations;
    private String[] prediction;


    public DERStatistics(double parsePercentage,
                         double reconstruction,
                         List<CompilationError> compilationErrors,
                         List<RecoveryOperation> recoveryOperations,
                         String[] prediction) {
        this.parsePercentage = parsePercentage;
        this.reconstruction = reconstruction;
        this.compilationErrors = compilationErrors;
        this.recoveryOperations = recoveryOperations;
        this.prediction = prediction;
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
