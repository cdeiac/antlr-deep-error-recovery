package antlr.errorrecovery;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ERPredictor {

    public static Predictor<int[], NDArray> load(String modelPath) {
        try {
            // model
            Path modelDir = Paths.get(modelPath.substring(0, modelPath.lastIndexOf("/")+1));
            String modelName = modelPath.substring(modelPath.lastIndexOf("/")+1);
            Model model = Model.newInstance(modelName);
            model.load(modelDir);
            // predictor
            return model.newPredictor(new ERModelTranslator());

        } catch (IOException | MalformedModelException e) {
            throw new RuntimeException(e);
        }
    }
}
