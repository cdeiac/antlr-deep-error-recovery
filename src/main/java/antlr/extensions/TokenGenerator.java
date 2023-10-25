package antlr.extensions;

import java.util.Random;

public class TokenGenerator {

    private static final Random RANDOM = new Random();
    private static final int LEFT_LIMIT = 97; // numeral 'a'
    private static final int RIGHT_LIMIT = 122; // letter 'z'


    public static String getRandomString(int stringLength) {
       return RANDOM.ints(LEFT_LIMIT, RIGHT_LIMIT + 1)
               .filter(i -> (i <= 57 || i >= 65) && (i <= 90 || i >= 97))
               .limit(stringLength)
               .collect(StringBuilder::new, StringBuilder::appendCodePoint, StringBuilder::append)
               .toString();
    }
}
