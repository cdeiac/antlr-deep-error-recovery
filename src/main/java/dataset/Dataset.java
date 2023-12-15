package dataset;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import dataset.json.DataContainer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

public class Dataset {

    private final static Logger logger = LoggerFactory.getLogger(Dataset.class);
    private final String fileName;
    private final String originalOutputFileName;
    private final String noisyOutputFileName;
    private final ObjectMapper mapper;
    private BufferedInputStream inputStream;


    public Dataset(String fileName) {
        this.fileName = fileName;
        this.mapper = new ObjectMapper();
        this.originalOutputFileName = this.formatFileName("original");//"src/main/resources/generated" + this.fileName.substring(this.fileName.indexOf("/"));
        this.noisyOutputFileName = this.formatFileName("noisy");//"src/main/resources/generated" + this.fileName.substring(this.fileName.indexOf("/"));
        this.mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
    }


    private String formatFileName(String source) {
        return "src/main/resources/generated/" + source + "_" + this.fileName.substring(this.fileName.indexOf("/")+1);
    }

    public void close() {
        try {
            this.inputStream.close();
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

    public Stream<DataContainer> parseJSON() {
        JsonFactory factory = new JsonFactory();
        List<DataContainer> jsonObjects = new ArrayList<>();
        try {
            JsonParser parser = factory.createParser(this.getClass().getClassLoader().getResourceAsStream(fileName));
            while (parser.nextToken() != null) {
                if (JsonToken.START_OBJECT.equals(parser.getCurrentToken())) {
                    DataContainer dataObject = this.mapper.readValue(parser, DataContainer.class);
                    jsonObjects.add(dataObject);
                }
            }
            parser.close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return jsonObjects.stream();
    }

    public JsonLine mapJsonLine(String line) {
        try {
            return mapper.readValue(line, JsonLine.class);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    public DataContainer mapContainer(String line) {
        try {
            return this.mapper.readValue(line, DataContainer.class);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     *  Save original tokenized dataset
     */
    public void writeToOriginalFile(List<DataContainer> containers) {
        File newFile = new File(this.originalOutputFileName);
        if (newFile.exists()) {
            newFile.delete();
        }
        try {
            newFile.createNewFile();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        try {
            this.mapper.writer().writeValue(new File(this.originalOutputFileName), containers);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } finally {
            logger.debug("Created original file: " + this.originalOutputFileName);
        }
    }

    /**
     *  Save modified dataset
     */
    public void writeToNoisyFile(List<DataContainer> containers) {
        File newFile = new File(this.noisyOutputFileName);
        if (newFile.exists()) {
            newFile.delete();
        }
        try {
            newFile.createNewFile();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        try {
            this.mapper.writer().writeValue(new File(this.noisyOutputFileName), containers);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } finally {
            logger.debug("Created noisy file: " + this.noisyOutputFileName);
        }
    }

    public BufferedInputStream getInputStream() {
        return this.inputStream;
    }
}
