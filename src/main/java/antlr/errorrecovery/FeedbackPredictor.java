package antlr.errorrecovery;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;

public class FeedbackPredictor {


    private Predictor<int[], NDArray> predictor;


    public FeedbackPredictor(String modelPath) {
        this.predictor = ERPredictor.load(modelPath);
    }


    public String predict(int feedbackPosition) {
        return null;
    }
}
