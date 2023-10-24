package com.antlr.data;

import com.fasterxml.jackson.annotation.JsonProperty;

public class JsonLine {
    
    @JsonProperty
    public String id;

    @JsonProperty
    public String url;

    @JsonProperty
    public String source;
}
