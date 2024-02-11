package antlr.errorrecovery;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class JavaOracleTranslator implements Translator<int[], NDArray> {

    @Override
    public NDArray processOutput(TranslatorContext translatorContext, NDList ndList) {
        // output dimensions: [seq_len, vocab_size]
        // do not compute argmax since we want to retrieve k-most probable tokens for feedback mechanism
        NDArray array = ndList.singletonOrThrow();
        NDArray transformed = array.argSort(1);
        NDArray res = transformed;
        return res;
    }

    @Override
    public NDList processInput(TranslatorContext translatorContext, int[] input) {
        NDArray array = translatorContext.getNDManager().create(input);
        NDList ndList = new NDList();
        ndList.add(array);
        return ndList;
    }

    @Override
    public Batchifier getBatchifier() {
        // disable batching since our model expects a single input at a time
        return null;
    }
}
