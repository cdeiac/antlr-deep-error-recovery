package dataset.noise;

public interface NoiseStrategy {

    /**
     *  Apply noise strategy to the given token
     *  @param token a valid token of any programming language
     *  @return token replacement or null (deletion)
     */
    String[] apply(String token);
}