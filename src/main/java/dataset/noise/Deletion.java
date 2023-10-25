package dataset.noise;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Deletion implements NoiseStrategy {

    private final static Logger logger = LoggerFactory.getLogger(Insertion.class);


    @Override
    public String[] apply(String token) {
        logger.debug("Delete Token: {}", token);
        return new String[0];
    }
}