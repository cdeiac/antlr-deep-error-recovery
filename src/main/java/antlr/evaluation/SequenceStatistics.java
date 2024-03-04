package antlr.evaluation;

public class SequenceStatistics {

    private String[] original;
    private String[] noisy;

    private String[] antlrBailPrediction;
    private String[] antlrErPrediction;
    private String[] derBailOnErrorPrediction;
    private String[] derOnErrorPrediction;
    private String[] derOnFilePrediction;


    public SequenceStatistics(String[] original, String[] noisy, String[] antlrBailPrediction, String[] antlrErPrediction, String[] derBailOnErrorPrediction, String[] derOnErrorPrediction, String[] derOnFilePrediction) {
        this.original = original;
        this.noisy = noisy;
        this.antlrBailPrediction = antlrBailPrediction;
        this.antlrErPrediction = antlrErPrediction;
        this.derBailOnErrorPrediction = derBailOnErrorPrediction;
        this.derOnErrorPrediction = derOnErrorPrediction;
        this.derOnFilePrediction = derOnFilePrediction;
    }

    public SequenceStatistics(String[] original, String[] noisy, String[] antlrErPrediction, String[] derOnErrorPrediction, String[] derOnFilePrediction) {
        this.original = original;
        this.noisy = noisy;
        this.antlrErPrediction = antlrErPrediction;
        this.derOnErrorPrediction = derOnErrorPrediction;
        this.derOnFilePrediction = derOnFilePrediction;
    }


    public static SequenceStatistics finalizeSequenceStatistics(ANTLRStatistics antlrBailStats,
                                                                ANTLRStatistics antlrStats,
                                                                DERStatistics derBailStats,
                                                                DERStatistics derStats,
                                                                DERStatistics onFileStats) {
       return new SequenceStatistics(
                antlrStats.getOriginal(),
                antlrStats.getNoisy(),
                antlrBailStats.getPrediction(),
                antlrStats.getPrediction(),
                derBailStats.getPrediction(),
                derStats.getPrediction(),
                onFileStats.getPrediction()
        );
    }

    public static SequenceStatistics finalizeSequenceStatistics(ANTLRStatistics antlrStats,
                                                                DERStatistics derStats,
                                                                DERStatistics onFileStats) {
        return new SequenceStatistics(
                antlrStats.getOriginal(),
                antlrStats.getNoisy(),
                antlrStats.getPrediction(),
                derStats.getPrediction(),
                onFileStats.getPrediction()
        );
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

    public String[] getAntlrBailPrediction() {
        return antlrBailPrediction;
    }

    public void setAntlrBailPrediction(String[] antlrBailPrediction) {
        this.antlrBailPrediction = antlrBailPrediction;
    }

    public String[] getAntlrErPrediction() {
        return antlrErPrediction;
    }

    public void setAntlrErPrediction(String[] antlrErPrediction) {
        this.antlrErPrediction = antlrErPrediction;
    }

    public String[] getDerBailOnErrorPrediction() {
        return derBailOnErrorPrediction;
    }

    public void setDerBailOnErrorPrediction(String[] derBailOnErrorPrediction) {
        this.derBailOnErrorPrediction = derBailOnErrorPrediction;
    }

    public String[] getDerOnErrorPrediction() {
        return derOnErrorPrediction;
    }

    public void setDerOnErrorPrediction(String[] derOnErrorPrediction) {
        this.derOnErrorPrediction = derOnErrorPrediction;
    }

    public String[] getDerOnFilePrediction() {
        return derOnFilePrediction;
    }

    public void setDerOnFilePrediction(String[] derOnFilePrediction) {
        this.derOnFilePrediction = derOnFilePrediction;
    }
}
