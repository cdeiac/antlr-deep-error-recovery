import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import dataset.Pipeline;
import dataset.json.DataContainer;
import dataset.noise.NoiseGenerator;
import dataset.tokenizer.JavaTokenizer;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class Runner {

    static {
        System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "DEBUG");
    }

    public static List<DataContainer> parse() throws Exception {
        JsonFactory factory = new JsonFactory();
        JsonParser parser = factory.createParser(new File("src/main/resources/original/jhetas_clean.json"));

        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        List<DataContainer> objectsList = new ArrayList<>();

        while (parser.nextToken() != null) {
            if (JsonToken.START_OBJECT.equals(parser.getCurrentToken())) {
                DataContainer dataObject = objectMapper.readValue(parser, DataContainer.class);
                objectsList.add(dataObject);
            }
        }
        return objectsList;

    }


    public static void main(String[] args) throws Exception {

        Pipeline pipeline = new Pipeline(
                //"data/data-java-test.jsonl",
                "original/jhetas_clean.json",
                //"data/jhetas_test.json",
                new double[]{0.1},
                //new double[]{0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8},
                new NoiseGenerator(new JavaTokenizer())
        );
        pipeline.run();
    }


}