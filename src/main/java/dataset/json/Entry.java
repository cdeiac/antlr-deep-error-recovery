package dataset.json;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonRootName;

@JsonRootName(value = "source")
public class Entry {

    @JsonProperty
    private File file;
    @JsonProperty("source")
    //@JsonDeserialize(using = SourceFormatter.class)
    private String sourceOriginal;

    @JsonProperty("sourceWithNoise")
    private String sourceWithNoise;


    public Entry() {

    }

    public Entry(File file, String sourceOriginal, String sourceWithNoise) {
        this.file = file;
        this.sourceOriginal = sourceOriginal;
        this.sourceWithNoise = sourceWithNoise;
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


    public String getSourceWithNoise() {
        return sourceWithNoise;
    }

    public File getFile() {
        return file;
    }

    public String getSourceOriginal() {
        return sourceOriginal;
    }
}
