package antlr.evaluation;

import java.util.List;

public class ANTLRStatistics {

    private double baseParsePercentage;
    private double baseReconstruction;
    private List<CompilationError> baseCompilationErrors;

    private double parsePercentage;
    private double reconstruction;
    private List<CompilationError> compilationErrors;


    public ANTLRStatistics(double baseParsePercentage,
                           double baseReconstruction,
                           List<CompilationError> baseCompilationErrors,
                           double parsePercentage,
                           double reconstruction,
                           List<CompilationError> compilationErrors) {

        this.baseParsePercentage = baseParsePercentage;
        this.baseReconstruction = baseReconstruction;
        this.baseCompilationErrors = baseCompilationErrors;
        this.parsePercentage = parsePercentage;
        this.reconstruction = reconstruction;
        this.compilationErrors = compilationErrors;
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
}
