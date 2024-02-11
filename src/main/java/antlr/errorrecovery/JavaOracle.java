package antlr.errorrecovery;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class JavaOracle {

    private Model model;
    private Predictor<int[], int[]> predictor;


    public JavaOracle(String cvDir, int cvIter) {
        this.initPredictor(this.formatModelPath(cvDir, cvIter));
    }


    private void initPredictor(String modelPath) {
        try {
            // model
            Path modelDir = Paths.get(modelPath.substring(0, modelPath.lastIndexOf("/")+1));
            String modelName = modelPath.substring(modelPath.lastIndexOf("/")+1);
            this.model = Model.newInstance(modelName);
            this.model.load(modelDir);
            // predictor
            this.predictor = this.model.newPredictor(new ERModelTop1Translator());
        } catch (IOException | MalformedModelException e) {
            throw new RuntimeException(e);
        }
    }

    public void closePredictor() {
        this.predictor.close();
        this.model.close();
    }

    public int[] predictMostProbableTokens(int[] input) {
        // convert to model tokens
        //int[] input = ModelDataConverter.encodeSequence(sequence);
        // predict
        return this.predictTop1(input);
        // convert to ANTLR tokens
        //return ANTLRModelConverter.encodeSequence(modelEncodedResult);
    }

    private int[] predictTop1(int[] modelEncodedInput) {
        try {
            return this.predictor.predict(modelEncodedInput);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
    }

    private String formatModelPath(String cvDir, int cvIter) {
        return "src/main/python/data/generated/checkpoints/" + cvDir + "/traced_model" + cvIter + ".pt";
    }
}
