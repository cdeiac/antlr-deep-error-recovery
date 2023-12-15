package dataset.json;

import com.fasterxml.jackson.annotation.JsonProperty;

public class File {

    @JsonProperty
    private String repo;
    @JsonProperty
    private String path;
    @JsonProperty
    private String url;


    public File() {}


    public void setRepo(String repo) {
        this.repo = repo;
    }

    public void setPath(String path) {
        this.path = path;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public String getRepo() {
        return repo;
    }

    public String getPath() {
        return path;
    }

    public String getUrl() {
        return url;
    }
}
