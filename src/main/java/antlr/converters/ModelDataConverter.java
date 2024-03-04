package antlr.converters;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class ModelDataConverter {

    public static final HashMap<String, Integer> TOKEN_2_INDEX;
    public static final HashMap<Integer, String> INDEX_2_TOKEN;
    // mapping to ANTLR token ID
    private static final ObjectMapper objectMapper = new ObjectMapper();
    private static final String T2I_PATH = "src/main/python/persistent/token2index.json";
    private static final String I2T_PATH = "src/main/python/persistent/index2token.json";


    static {
        try {
            TOKEN_2_INDEX = objectMapper.readValue(new File(T2I_PATH), new TypeReference<>() {});
            INDEX_2_TOKEN = objectMapper.readValue(new File(I2T_PATH), new TypeReference<>() {});
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static int[] encodeSequence(String sequence) {
        List<Integer> modelTokens = new ArrayList<>();
        for (String token : sequence.split("\\s")) {
            if (token.equals("WS")) {
                // skip whitespaces
                continue;
            }
            modelTokens.add(TOKEN_2_INDEX.get(token));
        }
        try {
            return modelTokens.stream()
                    .mapToInt(Integer::intValue)
                    .toArray();
        } catch (NullPointerException e) {
            System.out.println(e.getMessage());
            throw e;
        }
    }
}
