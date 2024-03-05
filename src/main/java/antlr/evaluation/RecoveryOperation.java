package antlr.evaluation;

public class RecoveryOperation {

    private String type;
    private int position;
    private int positionAfter;
    private String tokenOriginal;
    private String tokenBefore;
    private String tokenAfter;


    public RecoveryOperation(String type, int position,int positionAfter, String tokenBefore, String tokenAfter, String tokenOriginal) {
        this.type = type;
        this.position = position;
        this.positionAfter = positionAfter;
        this.tokenOriginal = tokenOriginal;
        this.tokenBefore = tokenBefore;
        this.tokenAfter = tokenAfter;
    }


    public String getTokenBefore() {
        return tokenBefore;
    }

    public void setTokenBefore(String tokenBefore) {
        this.tokenBefore = tokenBefore;
    }

    public String getTokenAfter() {
        return tokenAfter;
    }

    public void setTokenAfter(String tokenAfter) {
        this.tokenAfter = tokenAfter;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public int getPosition() {
        return position;
    }

    public void setPosition(int position) {
        this.position = position;
    }

    public String getTokenOriginal() {
        return tokenOriginal;
    }

    public void setTokenOriginal(String tokenOriginal) {
        this.tokenOriginal = tokenOriginal;
    }

    public int getPositionAfter() {
        return positionAfter;
    }

    public void setPositionAfter(int positionAfter) {
        this.positionAfter = positionAfter;
    }
}
