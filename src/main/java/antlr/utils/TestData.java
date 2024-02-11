package antlr.utils;

import com.fasterxml.jackson.annotation.JsonProperty;

public class TestData {

    @JsonProperty("original_data")
    private String originalData;

    @JsonProperty("noisy_data")
    private String noisyData;

    @JsonProperty("noise_operations")
    private int[] noiseOperations;

    // Getters and setters
    public String getOriginalData() {
        return originalData;
    }

    public void setOriginalData(String originalData) {
        this.originalData = originalData;
    }

    public String getNoisyData() {
        return noisyData;
    }

    public void setNoisyData(String noisyData) {
        this.noisyData = noisyData;
    }

    public int[] getNoiseOperations() {
        return noiseOperations;
    }

    public void setNoiseOperations(int[] noiseOperations) {
        this.noiseOperations = noiseOperations;
    }
}
