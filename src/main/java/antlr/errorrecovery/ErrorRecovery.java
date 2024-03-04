package antlr.errorrecovery;

public class ErrorRecovery {

    private String input;
    private String modification;
    private String target;


    public ErrorRecovery(String input, String modification, String target) {
        this.input = input;
        this.modification = modification;
        this.target = target;
    }


    public String getInput() {
        return input;
    }

    public void setInput(String input) {
        this.input = input;
    }

    public String getTarget() {
        return target;
    }

    public void setTarget(String target) {
        this.target = target;
    }

    public String getModification() {
        return modification;
    }

    public void setModification(String modification) {
        this.modification = modification;
    }
}
