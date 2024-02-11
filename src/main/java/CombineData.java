import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import dataset.json.DataContainer;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class CombineData {

    public static void main(String[] args) {
        try {
            ObjectMapper mapper = new ObjectMapper();

            // Read JSON files
            byte[] bytes = Files.readAllBytes(Paths.get("src/main/python/data/generated/cv/00_1/fold_2_test.json"));
            String jsonString =  new String(bytes);
            jsonString = "["+jsonString+"]";
            jsonString = jsonString.replace("}{", "},{");
            JsonNode jsonNode1 = mapper.readTree(jsonString);
            JsonNode jsonNode2 = mapper.readTree(new File("src/main/resources/generated/00_1/noisy_jhetas_clean.json"));

            // Combine JSON nodes
            List<Integer> indexList = findMatch(jsonNode1, jsonNode2);
            JsonNode combinedNode = combineJSON(jsonNode1, indexList);


        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static JsonNode combineJSON(JsonNode node1, List<Integer> indexList) {
        // load original sources
        List<DataContainer> jsonObjects;
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            jsonObjects = objectMapper.readValue(new File("src/main/resources/data/jhetas_clean.json"), new TypeReference<>() {});
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        List<String> originalSources = new ArrayList<>();
        indexList.forEach(idx -> originalSources.add(jsonObjects.get(idx).getSource().getSourceOriginal()));
        // Create an array node to hold combined JSON objects
        ObjectMapper mapper = new ObjectMapper();
        ArrayNode combinedArray = mapper.createArrayNode();

        for (int i = 0; i < originalSources.size(); i++) {
            ObjectNode originalData1 = (ObjectNode) node1.get(i);
            originalData1.put("source",  originalSources.get(i));
            combinedArray.add(originalData1);
        }
        return combinedArray;
    }

    private static List<Integer> findMatch(JsonNode node1, JsonNode node2) {
        // Create an array node to hold combined JSON objects
        ObjectMapper mapper = new ObjectMapper();
        ArrayNode combinedArray = mapper.createArrayNode();
        List<Integer> indexList = new ArrayList<>();

        // Iterate through first JSON array
        for (JsonNode obj1 : node1) {
            String originalData1 = obj1.get("original_data").asText();
            int index = 0;
            // Iterate through second JSON array
            for (JsonNode obj2 : node2) {
                String originalData2 = obj2.get("source").get("source").asText();
                // If original_data matches, combine the objects
                if (originalData1.contains(originalData2)) {
                    indexList.add(index);
                    break;
                }
                index +=1;
            }
        }
        return indexList;
    }

    private static JsonNode combineJSON(JsonNode node1, JsonNode node2, JsonNode node3) {
        // Create an array node to hold combined JSON objects
        ObjectMapper mapper = new ObjectMapper();
        ArrayNode combinedArray = mapper.createArrayNode();
        List<Integer> indexList = new ArrayList<>();

        // Iterate through first JSON array
        for (JsonNode obj1 : node1) {
            String originalData1 = obj1.get("original_data").asText();

            // Iterate through second JSON array
            for (JsonNode obj2 : node2) {
                String originalData2 = obj2.get("source").get("source").asText();

                // If original_data matches, combine the objects
                if (originalData1.contains(originalData2)) {
                    ObjectNode combinedObject = mapper.createObjectNode();
                    combinedObject.put("original_data", originalData1);
                    combinedObject.put("noisy_data", obj1.get("noisy_data").asText());
                    combinedObject.put("noise_operations", obj1.get("noise_operations").asText());
                    combinedObject.put("source", obj2.get("source").get("source").asText());
                    combinedObject.put("path", obj2.get("source").get("file").get("path").asText());
                    combinedObject.put("sourceWithNoise", obj2.get("sourceWithNoise").asText());
                    combinedObject.set("noiseOperations", obj2.get("noiseOperations"));
                    combinedArray.add(combinedObject);
                    break; // Break the inner loop as we found the matching object
                }
            }
        }
        ArrayNode finalArray = mapper.createArrayNode();
        // Iterate through first JSON array
        for (JsonNode obj1 : combinedArray) {
            String targetPath = obj1.get("path").asText();

            // Iterate through second JSON array
            for (JsonNode obj3 : node3) {
                String currentPath = obj3.get("source").get("path").asText();

                // If original_data matches, combine the objects
                if (currentPath.equals(targetPath)) {
                    ObjectNode combinedObject = (ObjectNode) obj1;
                    combinedObject.put("original_data", obj3.get("source").get("source"));
                    finalArray.add(combinedObject);
                    break; // Break the inner loop as we found the matching object
                }
            }
        }
        return finalArray;
    }
}
