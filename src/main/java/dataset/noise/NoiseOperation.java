package dataset.noise;

public class NoiseOperation {

    private String[] token;

    private int noiseOperation;


    public NoiseOperation(String[] token, int noiseOperation) {
        this.token = token;
        this.noiseOperation = noiseOperation;
    }


    public String[] getToken() {
        return token;
    }

    public void setToken(String[] token) {
        this.token = token;
    }

    public int getNoiseOperation() {
        return noiseOperation;
    }

    public void setNoiseOperation(int noiseOperation) {
        this.noiseOperation = noiseOperation;
    }
}
