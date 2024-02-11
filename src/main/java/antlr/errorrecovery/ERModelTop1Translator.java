package antlr.errorrecovery;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.util.Arrays;

public class ERModelTop1Translator implements Translator<int[], int[]> {

    @Override
    public int[] processOutput(TranslatorContext translatorContext, NDList ndList) {
        // output dimensions: [seq_len, vocab_size]
        // do not compute argmax since we want to retrieve k-most probable tokens for feedback mechanism
        NDArray output = ndList.singletonOrThrow().argMax(1);
        // transform output
        return Arrays.stream(output.toLongArray())
                .mapToInt(Math::toIntExact)
                .toArray();
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
