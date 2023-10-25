package dataset;

import antlr.JavaLexer;
import dataset.noise.NoiseGenerator;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.Token;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class Pipeline {

    private final Dataset dataset;
    private final double[] probabilities;
    private final NoiseGenerator noiseGenerator;
    private static final String SYSTEM_LINE_SEPARATOR = System.getProperty("line.separator");


    public Pipeline(String fileName, double[] probabilities, NoiseGenerator noiseGenerator) {
        this.dataset = new Dataset(fileName);
        this.probabilities = probabilities;
        this.noiseGenerator = noiseGenerator;
    }


    public void run() {
        this.generateNoise();
    }

    /**
     *  Parse a file in the jsonl format and generate random noise according to the given probabilities
     */
    private void generateNoise() {
        Arrays.stream(probabilities).forEach(probability -> {
            try(Stream<String> lines = dataset.loadFile()) {
                lines.forEach(line -> {
                    JsonLine jsonLine = dataset.mapJsonLine(line);
                    StringBuilder newLine = new StringBuilder();

                    Lexer javaLexer = this.newLexerInstanceForLanguage(jsonLine.source);
                    @SuppressWarnings("unchecked")
                    List<Token> tokenList = (List<Token>) javaLexer.getAllTokens().stream()
                            .toList();
                    // split line by token
                    tokenList.forEach(token -> {
                        if (token.getText().contains(SYSTEM_LINE_SEPARATOR)) {
                            newLine.append(" ");
                        }
                        else if (" ".equals(token.getText())) {
                            newLine.append(token.getText());
                        }
                        else {
                            String[] noisyTokens = noiseGenerator.processWithProbability(token.getText(), probability);
                            Arrays.stream(noisyTokens).forEach(newLine::append);
                        }
                    });
                    dataset.writeToFile(newLine);
                });
            }
        });
    }

    private Lexer newLexerInstanceForLanguage(String source) {
        // TODO: Support remaining langauges! (from file name)
        return new JavaLexer(CharStreams.fromString(source));
    }
}
