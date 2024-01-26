import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.modality.nlp.preprocess.PunctuationSeparator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.ParameterStore;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;
import antlr.errorrecovery.ERPredictor;
import com.google.gson.reflect.TypeToken;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Type;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class DeepErrorRecovery {

    private static final Logger logger = LoggerFactory.getLogger(DeepErrorRecovery.class);

    private static final int HIDDEN_SIZE = 256;
    private static final int EOS_TOKEN = 1;
    private static final int MAX_LENGTH = 50;

    private static final String TOKEN2INDEX_PATH = "src/main/python/persistence/token2index.json";
    private static final String INDEX2TOKEN_PATH = "src/main/python/persistence/index2token.json";


    public static void main(String[] args) throws Exception {
        var predictor = ERPredictor.load("src/main/python/data/generated/checkpoints/00_1/traced_model.pt");
        int[] input = exampleInput();
        int numPredictions = 16;
        for (int i=0; i <= numPredictions; i++) {
            var result = predictor.predict(input);
            System.out.println(result);
        }
    }

    public static int[] exampleInput() {
        return new int[] {
                0, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,  3,  4,  5,  6,  4,
                21,  4, 26,  4, 10,  4,  4, 12,  5,  4, 21,  4, 26,  4, 13, 14,  4, 21,
                4, 26, 10, 12, 18, 24, 10,  4, 38, 34, 27,  4, 19,  4, 10, 12, 38, 16,
                12,  5, 29,  4, 18, 17,  4,  8,  9,  4, 13,  5, 55, 11, 55, 11, 55, 11,
                55, 11, 55, 11, 55, 11, 55, 11, 55, 11, 55, 11, 55, 17, 18,  4, 10,  4,
                11,  4, 11, 55, 11, 16, 11,  4, 12, 18, 29,  4, 18, 17,  6, 32,  4, 10,
                4, 21,  4, 26,  4, 11,  4,  4, 11,  4,  4, 11,  7,  4, 11,  4,  8,  9,
                4, 12,  5, 24, 10,  4, 38,  4, 19,  4, 10, 12, 12,  5,  4, 19,  4, 10,
                4, 12, 18, 29, 18, 17,  4,  4, 13,  4,  8,  4, 19,  4, 10,  4, 12, 15,
                42,  9, 18, 39, 10,  7,  4, 13, 16, 18,  4, 21,  4, 19,  4, 10, 12, 18,
                4, 40, 12,  5,  4, 10,  4, 11,  4, 11,  4, 22,  4, 19,  4, 10,  4, 12,
                11,  4, 22, 16, 11,  4, 12, 18, 17, 17, 17,  1
        };
    }

    public static int[] generateRandomIntArray(int size) {
        int[] array = new int[size];
        Random random = new Random();
        for (int i = 0; i < size; i++) {
            array[i] = random.nextInt(0, 130);
        }
        return array;
    }
    static Translator<Integer, Float> translator = new Translator<>() {
        @Override
        public NDList processInput(TranslatorContext ctx, Integer input) {
            NDManager manager = ctx.getNDManager();
            NDArray array = manager.create(new int[]{input});
            return new NDList(array);
        }

        @Override
        public Float processOutput(TranslatorContext ctx, NDList list) {
            NDArray temp_arr = list.get(0);
            return temp_arr.getFloat();
        }
    };

    public static void predict() throws ModelException, TranslateException, IOException {
        Path path = Paths.get(TOKEN2INDEX_PATH);
        Map<String, Long> wrd2idx;
        try (InputStream is = Files.newInputStream(path)) {
            String json = Utils.toString(is);
            Type mapType = new TypeToken<Map<String, Long>>() {
            }.getType();
            wrd2idx = JsonUtils.GSON.fromJson(json, mapType);
        }

        path = Paths.get(INDEX2TOKEN_PATH);
        Map<String, String> idx2wrd;
        try (InputStream is = Files.newInputStream(path)) {
            String json = Utils.toString(is);
            Type mapType = new TypeToken<Map<String, String>>() {
            }.getType();
            idx2wrd = JsonUtils.GSON.fromJson(json, mapType);
        }

        Engine engine = Engine.getEngine("PyTorch");
        try (NDManager manager = engine.newBaseManager()) {
            try (ZooModel<NDList, NDList> encoder = getEncoderModel();
                 ZooModel<NDList, NDList> decoder = getDecoderModel()) {

                String french = "trop tard";
                NDList toDecode = predictEncoder(french, encoder, wrd2idx, manager);
                String english = predictDecoder(toDecode, decoder, idx2wrd, manager);

                logger.info("French: {}", french);
                logger.info("English: {}", english);
            }
        }
    }

    public static ZooModel<NDList, NDList> getEncoderModel() throws ModelException, IOException {
        String url =
                "https://resources.djl.ai/demo/pytorch/android/neural_machine_translation/optimized_encoder_150k.zip";

        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(url)
                        .optModelName("optimized_encoder_150k.ptl")
                        .optEngine("PyTorch")
                        .build();
        return criteria.loadModel();
    }

    public static ZooModel<NDList, NDList> getDecoderModel() throws ModelException, IOException {
        String url =
                "https://resources.djl.ai/demo/pytorch/android/neural_machine_translation/optimized_decoder_150k.zip";

        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(url)
                        .optModelName("optimized_decoder_150k.ptl")
                        .optEngine("PyTorch")
                        .build();
        return criteria.loadModel();
    }

    public static NDList predictEncoder(
            String text,
            ZooModel<NDList, NDList> model,
            Map<String, Long> wrd2idx,
            NDManager manager) {
        // maps french input to id's from french file
        List<String> list = Collections.singletonList(text);
        PunctuationSeparator punc = new PunctuationSeparator();
        list = punc.preprocess(list);
        List<Long> inputs = new ArrayList<>();
        for (String word : list) {
            if (word.length() == 1 && !Character.isAlphabetic(word.charAt(0))) {
                continue;
            }
            Long id = wrd2idx.get(word.toLowerCase(Locale.FRENCH));
            if (id == null) {
                throw new IllegalArgumentException("Word \"" + word + "\" not found.");
            }
            inputs.add(id);
        }

        // for forwarding the model
        Shape inputShape = new Shape(1);
        Shape hiddenShape = new Shape(1, 1, 256);
        FloatBuffer fb = FloatBuffer.allocate(256);
        NDArray hiddenTensor = manager.create(fb, hiddenShape);
        long[] outputsShape = {MAX_LENGTH, HIDDEN_SIZE};
        FloatBuffer outputTensorBuffer = FloatBuffer.allocate(MAX_LENGTH * HIDDEN_SIZE);

        // for using the model
        Block block = model.getBlock();
        ParameterStore ps = new ParameterStore();

        // loops through forwarding of each word
        for (long input : inputs) {
            NDArray inputTensor = manager.create(new long[]{input}, inputShape);
            NDList inputTensorList = new NDList(inputTensor, hiddenTensor);
            NDList outputs = block.forward(ps, inputTensorList, false);
            NDArray outputTensor = outputs.get(0);
            outputTensorBuffer.put(outputTensor.toFloatArray());
            hiddenTensor = outputs.get(1);
        }
        outputTensorBuffer.rewind();
        NDArray outputsTensor = manager.create(outputTensorBuffer, new Shape(outputsShape));

        return new NDList(outputsTensor, hiddenTensor);
    }

    public static String predictDecoder(
            NDList toDecode,
            ZooModel<NDList, NDList> model,
            Map<String, String> idx2wrd,
            NDManager manager) {
        // for forwarding the model
        Shape decoderInputShape = new Shape(1, 1);
        NDArray inputTensor = manager.create(new long[]{0}, decoderInputShape);
        ArrayList<Integer> result = new ArrayList<>(MAX_LENGTH);
        NDArray outputsTensor = toDecode.get(0);
        NDArray hiddenTensor = toDecode.get(1);

        // for using the model
        Block block = model.getBlock();
        ParameterStore ps = new ParameterStore();

        // loops through forwarding of each word
        for (int i = 0; i < MAX_LENGTH; i++) {
            NDList inputTensorList = new NDList(inputTensor, hiddenTensor, outputsTensor);
            NDList outputs = block.forward(ps, inputTensorList, false);
            NDArray outputTensor = outputs.get(0);
            hiddenTensor = outputs.get(1);
            float[] buf = outputTensor.toFloatArray();
            int topIdx = 0;
            double topVal = -Double.MAX_VALUE;
            for (int j = 0; j < buf.length; j++) {
                if (buf[j] > topVal) {
                    topVal = buf[j];
                    topIdx = j;
                }
            }

            if (topIdx == EOS_TOKEN) {
                break;
            }

            result.add(topIdx);
            inputTensor = manager.create(new long[]{topIdx}, decoderInputShape);
        }

        StringBuilder sb = new StringBuilder();
        // map english words and create output string
        for (Integer word : result) {
            sb.append(idx2wrd.get(word.toString())).append(' ');
        }
        return sb.toString().trim();
    }
}
