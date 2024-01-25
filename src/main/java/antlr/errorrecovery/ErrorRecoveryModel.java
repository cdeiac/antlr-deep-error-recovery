import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslateException;
import ai.djl.util.Utils;
import ai.djl.inference.Predictor;
import ai.djl.inference.Prediction;

public class ModelLoader {
    public static void main(String[] args) throws ModelException, TranslateException {
        try (Model model = ModelZoo.loadModel("ai.djl.pytorch", "resnet18", "1.0", null)) {
            // Load the model
            model.loadCriteria();

            // Perform inference
            try (NDManager manager = NDManager.newBaseManager()) {
                NDArray input = manager.create(new float[]{0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f}).reshape(new Shape(2, 3));

                Predictor<NDList, NDList> predictor = model.newPredictor();
                NDList result = predictor.predict(new NDList(input));
                System.out.println(result.singletonOrThrow());
            }
        }
    }
}
