package dataset;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class Dataset {

    private final String fileName;
    private final String outputFileName;
    private final ObjectMapper mapper;
    private BufferedInputStream inputStream;
    private BufferedWriter writer;


    public Dataset(String fileName) {
        this.fileName = fileName;
        this.mapper = new ObjectMapper();
        this.outputFileName = "target" + this.fileName.substring(this.fileName.indexOf("/"));

    }


    public void close() {
        try {
            this.inputStream.close();
            this.writer.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    /**
     *  Loads the dataset from classpath
     *
     */
    public Stream<String> loadFile() {
        try {
            InputStream is = this.getClass().getClassLoader().getResourceAsStream(fileName);
            URL resource = Dataset.class.getClassLoader().getResource(fileName);
            return Files.lines(Paths.get(resource.toURI()));
        } catch (IOException | URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    public JsonLine mapJsonLine(String line) {
        try {
            return mapper.readValue(line, JsonLine.class);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     *  Save modified dataset
     */
    public void writeToFile(StringBuilder stringBuilder) {
        try {
            if (this.writer == null) {
                File newFile = new File(this.outputFileName);
                if (newFile.exists()) {
                    newFile.delete();
                }
                newFile.createNewFile();
                this.writer = new BufferedWriter(new FileWriter(this.outputFileName, true));
            }
            this.writer.write(stringBuilder.toString());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public BufferedInputStream getInputStream() {
        return this.inputStream;
    }
}
