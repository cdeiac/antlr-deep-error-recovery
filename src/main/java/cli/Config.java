package cli;

import org.apache.commons.cli.*;

import java.text.DecimalFormat;

public class Config {

    private final String dataPath;
    private final String generatedDirectoryName;
    private final double noiseProbability;


    public Config(String dataPath, double noiseProbability) {
        this.dataPath = dataPath;
        this.noiseProbability = noiseProbability;
        this.generatedDirectoryName = new DecimalFormat("00.0")
                .format(noiseProbability)
                .replace(".", "_");
    }


    public String getDataPath() {
        return dataPath;
    }

    public double getNoiseProbability() {
        return noiseProbability;
    }

    public String getGeneratedDirectoryName() {
        return generatedDirectoryName;
    }


    public static Config parseArguments(String[] args) {

        Options options = new Options();

        Option dataPathOption = Option.builder("d")
                .longOpt("data_path")
                .argName("data_path")
                .hasArg()
                .required(true)
                .desc("Path to the original data directory")
                .type(String.class)
                .build();
        options.addOption(dataPathOption);

        Option noiseProbabilityOption = Option.builder("p")
                .longOpt("noise_prob")
                .argName("noise_prob")
                .hasArg()
                .required(true)
                .desc("Noise probability")
                .type(Double.class)
                .build();
        options.addOption(noiseProbabilityOption);

        // define parser
        CommandLine cmd;
        CommandLineParser parser = new DefaultParser();
        HelpFormatter helper = new HelpFormatter();

        try {
            cmd = parser.parse(options, args);
            return new Config(cmd.getOptionValue("d"), Double.parseDouble(cmd.getOptionValue("p")));

        } catch (ParseException e) {
            helper.printHelp("Usage:", options);
            throw new RuntimeException();
        }
    }
}
