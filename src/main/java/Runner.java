import cli.Config;
import dataset.Pipeline;
import dataset.noise.NoiseGenerator;
import dataset.tokenizer.JavaTokenizer;

public class Runner {

    static {
        System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "INFO");
    }


    public static void main(String[] args) throws Exception {

        Config config = Config.parseArguments(args);

        Pipeline pipeline = new Pipeline(
                config,
                new NoiseGenerator(new JavaTokenizer())
        );
        pipeline.run();
        System.out.println(config.getGeneratedDirectoryName());
    }
}