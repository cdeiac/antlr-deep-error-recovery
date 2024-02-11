package antlr.evaluation;

import java.util.List;

public class DERStatistics {

    private double parsePercentage;
    private double reconstruction;
    private List<CompilationError> compilationErrors;


    public DERStatistics() {}

    public DERStatistics(double parsePercentage, double reconstruction, List<CompilationError> compilationErrors) {
        this.parsePercentage = parsePercentage;
        this.reconstruction = reconstruction;
        this.compilationErrors = compilationErrors;
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
}
