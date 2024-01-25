package dataset;

import cli.Config;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import dataset.json.DataContainer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.stream.Stream;

public class Dataset {

    private final static Logger logger = LoggerFactory.getLogger(Dataset.class);
    private final String fileName;
    private final String noisyOutputFileName;
    private final ObjectMapper mapper;


    public Dataset(Config config) {
        this.fileName = config.getDataPath();
        this.mapper = new ObjectMapper();
        this.noisyOutputFileName = this.formatFileName(config.getGeneratedDirectoryName());
        this.mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
    }


    private String formatFileName(String directory) {
        return "src/main/resources/generated/" + directory + "/" + "noisy_" + this.fileName.substring(this.fileName.lastIndexOf("/")+1);
    }

    public Stream<DataContainer> parseJSON() {
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            List<DataContainer> jsonObjects = objectMapper.readValue(new File(this.fileName), new TypeReference<>() {});
            return jsonObjects.stream();
        } catch (Exception e) {
            throw new RuntimeException(e);
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
            this.formatDirectory(newFile.getPath()).mkdirs();
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

    public File formatDirectory(String pathToFile) {
        return new File(pathToFile.substring(0, pathToFile.lastIndexOf("/")));
    }
}
