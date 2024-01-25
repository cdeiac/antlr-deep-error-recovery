package dataset.json;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonRootName;

@JsonRootName(value = "source")
public class Entry {

    @JsonProperty
    private File file;
    @JsonProperty("source")
    private String sourceOriginal;

    @JsonProperty("sourceWithNoise")
    private String sourceWithNoise;

    @JsonProperty("noiseOperations")
    private int[] noiseOperations;


    public Entry() {

    }

    public Entry(File file, String sourceOriginal, String sourceWithNoise, int[] noiseOperations) {
        this.file = file;
        this.sourceOriginal = sourceOriginal;
        this.sourceWithNoise = sourceWithNoise;
        this.noiseOperations = noiseOperations;
    }


    public void setFile(File file) {
        this.file = file;
    }

    public void setSourceOriginal(String source) {
        this.sourceOriginal = source;
    }

    public void setSourceWithNoise(String sourceWithNoise) {
        this.sourceWithNoise = sourceWithNoise;
    }

    public void setNoiseOperations(int[] noiseOperations) {
        this.noiseOperations = noiseOperations;
    }


    public String getSourceWithNoise() {
        return sourceWithNoise;
    }

    public File getFile() {
        return file;
    }

    public String getSourceOriginal() {
        return sourceOriginal;
    }

    public int[] getNoiseOperations() {
        return noiseOperations;
    }
}
