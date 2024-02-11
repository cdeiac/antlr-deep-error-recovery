package antlr.evaluation;

public class BaselineStatistics {

    private double parseSuccessAntlrBail;
    private double parseSuccessAntlrWithER;


    public BaselineStatistics(double parseSuccessAntlrBail, double parseSuccessAntlrWithER) {
        this.parseSuccessAntlrBail = parseSuccessAntlrBail;
        this.parseSuccessAntlrWithER = parseSuccessAntlrWithER;
    }


    public double getParseSuccessAntlrBail() {
        return parseSuccessAntlrBail;
    }

    public double getParseSuccessAntlrWithER() {
        return parseSuccessAntlrWithER;
    }
}
