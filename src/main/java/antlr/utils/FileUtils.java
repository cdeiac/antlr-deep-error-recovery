package antlr.utils;

import antlr.evaluation.ParseStatistics;
import antlr.evaluation.SequenceStatistics;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

public class FileUtils {

    private static final String CV_BASE_DIR = "src/main/python/data/generated/cv";
    private static final ObjectMapper objectMapper = new ObjectMapper();


    private static List<TestData> loadTestData(Path filePath) {
        try {
            // read & deserialize JSON file
            String jsonContent = new String(Files.readAllBytes(filePath));
            // JSON correction
            jsonContent = "[" + jsonContent + "]";
            jsonContent = jsonContent.replace("}{", "},{");
            List<TestData> parsedFiles = objectMapper.readValue(jsonContent, new TypeReference<>() {
            });
            return parsedFiles.stream()
                    // skip "empty" files
                    .filter(file -> file.getNoiseOperations().length > 0)
                    .collect(Collectors.toList());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static List<TestData> loadAllTestData(String cvDir, int cvIter) {
        Path filePath = formatCVFileName(cvDir, cvIter);
        return loadTestData(filePath);

    }

    private static Path formatCVFileName(String cvDir, int cvIter) {
        return Paths.get(CV_BASE_DIR + "/" + cvDir + "/" + "fold_" + cvIter + "_test.json");
    }

    private static Path formatCVDirectoryPath(String cvDir, String fileName) {
        return Paths.get(CV_BASE_DIR + "/" + cvDir + "/" + fileName);
    }

    public static void saveParseStatistics(String cvDir, int cvIter, List<ParseStatistics> parseStatistics) {
        try {
            objectMapper.writeValue(
                    new File(FileUtils.formatCVDirectoryPath(cvDir, "parse_statistics" + cvIter + ".json")
                            .toString()),
                    parseStatistics
            );
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void saveSequenceStatistics(String cvDir, int cvIter, List<SequenceStatistics> sequenceStatistics) {
        try {
            objectMapper.writeValue(
                    new File(FileUtils.formatCVDirectoryPath(cvDir, "sequence_statistics" + cvIter + ".json")
                            .toString()),
                    sequenceStatistics
            );
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
