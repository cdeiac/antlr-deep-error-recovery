package antlr.evaluation;

public class CompilationError {

    private final String type;
    private final int position;


    public CompilationError(String message, int position) {
        this.type = this.formatType(message);
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
}
