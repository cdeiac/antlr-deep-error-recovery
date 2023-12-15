package dataset.json;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;

public class DataContainer {

    @JsonProperty
    private Entry source;
    @JsonIgnore
    private String hetas;


    public DataContainer() {}

    public DataContainer(Entry entry, String hetas) {
        this.source = entry;
        this.hetas = hetas;
    }


    public void setSource(Entry entry) {
        this.source = entry;
    }

    public void setHetas(String hetas) {
        this.hetas = hetas;
    }

    public Entry getSource() {
        return source;
    }

    public String getHetas() {
        return hetas;
    }
}
