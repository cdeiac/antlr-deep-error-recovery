package antlr.errorrecovery;

public class ErrorPosition {

    private double relativeErrorPosition;


    public ErrorPosition(double relativeErrorPosition) {
        this.relativeErrorPosition = relativeErrorPosition;
    }


    public double getRelativeErrorPositions() {
        return relativeErrorPosition;
    }
}
