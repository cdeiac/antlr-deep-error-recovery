package antlr.utils;

import java.util.List;

public class TestDataContainer {

    private final List<String> noisyData;
    private final List<int[]> noiseOperations;


    public TestDataContainer(List<String> noisyData, List<int[]> noiseOperations) {
        this.noisyData = noisyData;
        this.noiseOperations = noiseOperations;
    }


    public List<String> getNoisyData() {
        return noisyData;
    }

    public List<int[]> getNoiseOperations() {
        return noiseOperations;
    }
}
