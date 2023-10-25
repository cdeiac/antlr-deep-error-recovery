import dataset.Pipeline;
import dataset.noise.NoiseGenerator;
import dataset.tokenizer.JavaTokenizer;

public class Runner {

    static {
        System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "DEBUG");
    }


    public static void main(String[] args) throws Exception {

        Pipeline pipeline = new Pipeline(
                "data/data-java-test.jsonl",//"data/data-java.jsonl",
                new double[]{0.1},// new double[]{0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8},
                new NoiseGenerator(new JavaTokenizer())
        );
        pipeline.run();
    }
}