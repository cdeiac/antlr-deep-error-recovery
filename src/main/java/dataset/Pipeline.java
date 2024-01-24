package dataset;

import antlr.JavaLexer;
import cli.Config;
import dataset.json.DataContainer;
import dataset.json.Entry;
import dataset.noise.NoiseGenerator;
import dataset.noise.NoiseOperation;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.Token;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Pipeline {

    private final Dataset dataset;
    private final double probability;
    private final NoiseGenerator noiseGenerator;


    public Pipeline(Config config, NoiseGenerator noiseGenerator) {
        this.dataset = new Dataset(config);
        this.probability = config.getNoiseProbability();
        this.noiseGenerator = noiseGenerator;
    }


    public void run() {
        //this.generateNoise();
        List<DataContainer> originalContainers = new ArrayList<>();
        List<DataContainer> newContainers = new ArrayList<>();

        try(Stream<DataContainer> containers = dataset.parseJSON()) {
            containers.forEach(container -> {
                Lexer javaLexer = this.newLexerInstanceForLanguage(container.getSource().getSourceOriginal());
                @SuppressWarnings("unchecked")
                List<Token> tokenList = javaLexer.getAllTokens()
                        // filter out whitespace tokens
                        .stream()
                        .filter(token -> token.getType() != 125) // 125 = WS
                        .collect(Collectors.toList());
                // add original source
                List<String> originalTokens = new ArrayList<>();
                tokenList.forEach(token -> originalTokens.add(javaLexer.getVocabulary().getSymbolicName(token.getType())));
                String tokenizedFile  = String.join(" ", originalTokens);
                DataContainer tokenizedContainer = new DataContainer(
                        new Entry(container.getSource().getFile(), tokenizedFile, null, new int[]{}),
                        null
                );
                originalContainers.add(tokenizedContainer);
                // add noisy sources
                List<Integer> noiseOperations = new ArrayList<>();
                List<String> noisySource = new ArrayList<>();
                // split line by token
                tokenList.forEach(token -> {
                    NoiseOperation noiseOperation = noiseGenerator.processWithProbability(
                            javaLexer.getVocabulary().getSymbolicName(token.getType()),
                            probability
                    );
                    noisySource.addAll(Arrays.asList(noiseOperation.getToken()));
                    noiseOperations.add(noiseOperation.getNoiseOperation());
                });
                String noisyFile  = String.join(" ", noisySource);
                newContainers.add(new DataContainer(
                        new Entry(
                                tokenizedContainer.getSource().getFile(),
                                tokenizedFile,
                                noisyFile,
                                noiseOperations.stream().mapToInt(i->i).toArray()
                        ),
                        null
                ));
            });
        }
        // write original (tokenized) and noisy files
        //dataset.writeToOriginalFile(originalContainers);
        dataset.writeToNoisyFile(newContainers);

    }

    private Lexer newLexerInstanceForLanguage(String source) {
        // TODO: Support remaining languages! (from file name)
        return new JavaLexer(CharStreams.fromString(source));
    }
}
