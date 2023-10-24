package com.antlr;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.Recognizer;
import org.antlr.v4.runtime.Token;

import com.antlr.data.Dataset;
import com.antlr.data.JsonLine;
import com.antlr.data.noise.NoiseGenerator;
import com.antlr.grammars.java.JavaLexer;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

public class Runner {
    
    public static void main(String[] args) throws Exception {
        
        String fileName = "data/data-java.jsonl";
        URL resource = Runner.class.getClassLoader().getResource(fileName);
       
        NoiseGenerator noiseGenerator = new NoiseGenerator();
        ObjectMapper mapper = new ObjectMapper();

        try (Stream<String> lines = Files.lines(Paths.get(resource.toURI()))) {
            lines.forEach(line -> {

                try {
                    JsonLine jsonLine = mapper.readValue(line, JsonLine.class);
                    JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(jsonLine.source));
                    List<? extends Token> tokens = javaLexer.getAllTokens();
                    // split line by token
                    //String[] tokens = jsonLine.source.split(" ");
                    StringBuilder newLine = new StringBuilder();

                    Stream.of(tokens).forEach(token -> {

                        //int javaToken = tokenizer.getTokenType(token);
                        //Map<String, Integer> map = tokenizer.getTokenTypeMap();

                        if (token.equals("\\n")) {
                            newLine.append(token);
                        }
                        else {
                            //noiseGenerator.apply(token);
                            newLine.append(token);
                            newLine.append(" ");
                        }
                    
                    });
                } catch (JsonProcessingException e) {
                    e.printStackTrace();
                }
               
                
            });
        } catch(Exception e) {
            e.printStackTrace();
        }
        

            
       
    }
}
