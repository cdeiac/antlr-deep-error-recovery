package dataset;

import antlr.JavaLexer;
import dataset.json.DataContainer;
import dataset.json.Entry;
import dataset.noise.NoiseGenerator;
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
    private final double[] probabilities;
    private final NoiseGenerator noiseGenerator;
    private static final String SYSTEM_LINE_SEPARATOR = System.getProperty("line.separator");


    public Pipeline(String fileName, double[] probabilities, NoiseGenerator noiseGenerator) {
        this.dataset = new Dataset(fileName);
        this.probabilities = probabilities;
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
                        new Entry(container.getSource().getFile(), tokenizedFile, null),
                        null
                );
                originalContainers.add(tokenizedContainer);
                // add noisy sources
                Arrays.stream(this.probabilities).forEach(probability -> {
                    List<String> noisySource = new ArrayList<>();
                    // split line by token
                    tokenList.forEach(token -> {
                        String[] noisyTokens = noiseGenerator.processWithProbability(
                                javaLexer.getVocabulary().getSymbolicName(token.getType()),
                                probability
                        );
                        noisySource.addAll(Arrays.asList(noisyTokens));
                    });
                    String noisyFile  = String.join(" ", noisySource);
                    // only add if we generated noise for the given source file
                    if (!tokenizedFile.equals(noisyFile)) {
                        newContainers.add(new DataContainer(
                                new Entry(tokenizedContainer.getSource().getFile(), tokenizedFile, noisyFile),
                                null
                        ));
                    }
                });
            });
        }
        // write original (tokenized) and noisy files
        dataset.writeToOriginalFile(originalContainers);
        dataset.writeToNoisyFile(newContainers);

    }

    /**
     *  Parse a file in the jsonl format and generate random noise according to the given probabilities
     */
    private void generateNoise() {
        Arrays.stream(probabilities).forEach(probability -> {
            try(Stream<DataContainer> lines = dataset.parseJSON()) {
                lines.forEach(container -> {
                    //JsonLine jsonLine = dataset.mapJsonLine(line.getSource().getSourceOriginal());
                    StringBuilder newLine = new StringBuilder();

                    Lexer javaLexer = this.newLexerInstanceForLanguage(container.getSource().getSourceOriginal());
                    @SuppressWarnings("unchecked")
                    List<Token> tokenList = (List<Token>) javaLexer.getAllTokens().stream()
                            .toList();
                    // split line by token
                    tokenList.forEach(token -> {
                        /*
                        if (token.getText().contains(SYSTEM_LINE_SEPARATOR)) {
                            newLine.append(" ");
                        }
                        else if (" ".equals(token)) {
                            newLine.append(token);
                        }
                        else {*/
                        if (token.getText().strip().equalsIgnoreCase("")) {
                            return;
                        }
                        if (token.getText().equalsIgnoreCase(" ")) {
                            newLine.append(" ");
                        }

                        else {
                            newLine.append(javaLexer.getVocabulary().getSymbolicName(token.getType()));
                            newLine.append(" ");

                            String[] noisyTokens = noiseGenerator.processWithProbability(
                                    javaLexer.getVocabulary().getSymbolicName(token.getType()),
                                    probability
                            );
                            Arrays.stream(noisyTokens).forEach(nt -> {
                                if (!(token.getText().equals(nt))) {
                                    newLine.append(nt);
                                    newLine.append(" ");
                                }
                            });
                        }
                    });
                    //newContent.add(newLine.toString());
                });
            }
        });
        //dataset.writeToFile(newContent);
    }

    private Lexer newLexerInstanceForLanguage(String source) {
        // TODO: Support remaining languages! (from file name)
        return new JavaLexer(CharStreams.fromString(source));
    }
}
