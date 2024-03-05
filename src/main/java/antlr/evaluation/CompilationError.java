package antlr.evaluation;

public class CompilationError {

    private final String type;
    private final int position;
    private final int adjustedPosition;


    public CompilationError(String message, int position, int adjustedPosition) {
        this.type = this.formatType(message);
        this.adjustedPosition = adjustedPosition;
        this.position = position;
    }


    private String formatType(String message) {
        if (message.contains("missing")) {
            return "MissingError";
        }
        else if (message.contains("mismatched")) {
            return "MismatchedError";
        }
        else return "ExtraneousInputError";
    }


    public String getType() {
        return type;
    }

    public int getPosition() {
        return position;
    }

    public int getAdjustedPosition() {
        return adjustedPosition;
    }


}
