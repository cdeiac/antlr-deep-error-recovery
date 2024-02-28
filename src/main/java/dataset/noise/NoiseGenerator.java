package dataset.noise;

import dataset.tokenizer.Tokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Random;

public class NoiseGenerator {

    private final static Logger logger = LoggerFactory.getLogger(NoiseGenerator.class);
    private final List<NoiseStrategy> noiseStrategies;
    private final Random randomStrategy;
    private final Random randomNoise;


    public NoiseGenerator(Tokenizer tokenizer) {
        this.noiseStrategies = List.of(new Deletion(), new Modification(tokenizer), new Insertion(tokenizer));
        this.randomStrategy = new Random();
        this.randomNoise = new Random();
    }


    /**
     *  Generate noise for given token according to the configured probability
     *  @param token given token
     *  @param probability given probability (e.g., 0.1, 1.5, 3.0)
     *  @return generated tokens, or the same token if the probability did not apply
     */
    public NoiseOperation processWithProbability(String token, double probability) {
        logger.debug("Process Token with probability {}", probability);
        double randomNumber = this.randomNoise.nextDouble();
        String[] generatedToken = new String[]{token};
        int noiseOperation = 0;
        if (randomNumber < probability) {
            NoiseStrategy noiseStrategy = this.selectRandomNoiseStrategy();

            if (noiseStrategy instanceof Deletion) {
                noiseOperation = 1;
            }
            else if (noiseStrategy instanceof Insertion) {
                noiseOperation = 2;
            }
            else if (noiseStrategy instanceof Modification) {
                noiseOperation = 3;
            }
            generatedToken = noiseStrategy.apply(token);
        }
        return new NoiseOperation(generatedToken, noiseOperation);
    }

    /**
     *  Selects a noise strategy at random
     *  @return noise strategy
     */
    private NoiseStrategy selectRandomNoiseStrategy() {
        return noiseStrategies.get(randomStrategy.nextInt(noiseStrategies.size()));
    }
}
