package antlr.utils;

import antlr.evaluation.ParseStatistics;
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
    private static final String JSON_FILE_EXTENSION = "test.json";
    private static final List<String> TEST_FILE_NAMES = List.of("fold_2_test.json"); //List.of("fold_0_test.json", "fold_1_test.json", "fold_2_test.json");
    private static final List<String> CV_DIRS = List.of("fold_2_test.json");//List.of("fold_0_test.json", "fold_1_test.json", "fold_2_test.json");
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
        String path = CV_BASE_DIR + "/" + cvDir + "/" + "fold_" + cvIter + "_test.json";
        return Paths.get(path);
    }

    private static Path formatCVDirectoryPath(String cvDir, String fileName) {
        String path = CV_BASE_DIR + "/" + cvDir + "/" + fileName;
        return Paths.get(path);
    }
    /*
    public static HashMap<String, TestDataContainer> loadAllTestData() {
        try {
            HashMap<String, TestDataContainer> testDataContainerMap = new HashMap<>();
            Files.walkFileTree(CV_DIR_PATH, new SimpleFileVisitor<>() {
                private TestDataContainer readNoisyJSON(Path filePath) {
                    try {
                        // read & deserialize JSON file
                        String jsonContent = new String(Files.readAllBytes(filePath));
                        // JSON correction
                        jsonContent = "[" + jsonContent + "]";
                        jsonContent = jsonContent.replace("}{", "},{");
                        List<TestData> testDataList = objectMapper.readValue(jsonContent, new TypeReference<>() {});
                        // map data
                        List<String> noisyData = testDataList.stream()
                                .map(TestData::getNoisyData)
                                .toList();
                        List<int[]> noiseOperations = testDataList.stream()
                                .map(TestData::getNoiseOperations)
                                .toList();
                        return new TestDataContainer(noisyData, noiseOperations);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }

                @Override
                public FileVisitResult visitFile(Path filePath, BasicFileAttributes attrs) {
                    // Check if the file is a JSON file
                    if (Files.isRegularFile(filePath) && filePath.toString().endsWith(JSON_FILE_EXTENSION)) {
                        testDataContainerMap.put(filePath.getParent().getFileName().toString(), this.readNoisyJSON(filePath));
                    }
                    return FileVisitResult.CONTINUE;
                }
            });
            return testDataContainerMap;
        } catch (IOException e) {
            throw new RuntimeException("Failed to parse test data!");
        }
    }

     */

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
}
