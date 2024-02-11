package antlr.errorrecovery;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslateException;
import antlr.converters.ANTLRModelConverter;
import antlr.converters.ModelDataConverter;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ERPredictor {

    private Predictor<int[], int[]> predictor;
    private Predictor<int[], NDArray> top2Predictor;


    public ERPredictor(String modelPath) {
        this.initPredictor(modelPath);
    }


    private void initPredictor(String modelPath) {
        try {
            // model
            Path modelDir = Paths.get(modelPath.substring(0, modelPath.lastIndexOf("/")+1));
            String modelName = modelPath.substring(modelPath.lastIndexOf("/")+1);
            Model model = Model.newInstance(modelName);
            model.load(modelDir);
            // predictor
            this.predictor = model.newPredictor(new ERModelTop1Translator());
            this.top2Predictor = model.newPredictor(new ERModelTranslator());

        } catch (IOException | MalformedModelException e) {
            throw new RuntimeException(e);
        }
    }

    public int[] predictMostProbableTokens(String sequence) {
        // convert to model tokens
        int[] input = ModelDataConverter.encodeSequence(sequence);
        // predict
        int[] modelEncodedResult = this.predictTop1(input);
        // convert to ANTLR tokens
        return ANTLRModelConverter.encodeSequence(modelEncodedResult);
    }

    public int[] predictSecondMostProbableToken(int[] modelEncodedIds, int position) {
        // predict
        int[] modelEncodedResult = this.predictTop2AtPosition(modelEncodedIds, position);
        return modelEncodedIds;
    }

    private int[] predictTop1(int[] modelEncodedInput) {
        try {
            return this.predictor.predict(modelEncodedInput);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
    }

    public int[] predictTop2AtPosition(int[] modelEncodedInput, int position) {
        try {
            try (NDManager manager = NDManager.newBaseManager()) {
                NDArray output = this.top2Predictor.predict(modelEncodedInput);
                output.attach(manager);
                float[] result = new float[(int) output.getShape().get(0)];
                NDList topK = output.topK(2, 1);
                // Iterate through the elements of the NDArray
                for (int i = 0; i < output.getShape().get(0); i++) {
                    float element;
                    if (i == position) {
                        element = output.getFloat(i, 2-1); // top1 = element 0
                    }
                    else {
                        element = output.getFloat(i, 0);
                    }
                    result[i] = element;
                }
                return manager.create(result).toIntArray();
            }


        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
    }

    private int[] predictTopKAtPosition(int[] modelEncodedInput, int k, int position) {
        try {
            try (NDManager manager = NDManager.newBaseManager()) {
                NDArray output = this.top2Predictor.predict(modelEncodedInput);
                output.attach(manager);
                float[] result = new float[(int) output.getShape().get(0)];
                NDList topK = output.topK(k, 1);
                // Iterate through the elements of the NDArray
                for (int i = 0; i < output.getShape().get(0); i++) {
                    float element;
                    if (i == position) {
                        element = output.getFloat(i, k-1); // top1 = element 0
                    }
                    else {
                        element = output.getFloat(i, 0);
                    }
                    result[i] = element;
                }
                return manager.create(result).toIntArray();
            }


        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
    }

}
