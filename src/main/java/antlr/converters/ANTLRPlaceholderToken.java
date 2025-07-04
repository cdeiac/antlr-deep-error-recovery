package antlr.converters;

import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class ANTLRPlaceholderToken {

    public static HashMap<Integer, String> placeholders = new HashMap<>() {{
        put(1, "abstract");
        put(2, "assert");
        put(3, "boolean");
        put(4, "break");
        put(5, "byte");
        put(6, "case");
        put(7, "catch");
        put(8, "char");
        put(9, "class");
        put(10, "const");
        put(11, "continue");
        put(12, "default");
        put(13, "do");
        put(14, "double");
        put(15, "else");
        put(16, "enum");
        put(17, "extends");
        put(18, "final");
        put(19, "finally");
        put(20, "float");
        put(21, "for");
        put(22, "if");
        put(23, "goto");
        put(24, "implements");
        put(25, "import");
        put(26, "instanceof");
        put(27, "int");
        put(28, "interface");
        put(29, "long");
        put(30, "native");
        put(31, "new");
        put(32, "package");
        put(33, "private");
        put(34, "protected");
        put(35, "public");
        put(36, "return");
        put(37, "short");
        put(38, "static");
        put(39, "strictfp");
        put(40, "super");
        put(41, "switch");
        put(42, "synchronized");
        put(43, "this");
        put(44, "throw");
        put(45, "throws");
        put(46, "transient");
        put(47, "try");
        put(48, "void");
        put(49, "volatile");
        put(50, "while");
        put(51, "module");
        put(52, "open");
        put(53, "requires");
        put(54, "exports");
        put(55, "opens");
        put(56, "to");
        put(57, "uses");
        put(58, "provides");
        put(59, "with");
        put(60, "transitive");
        put(61, "var");
        put(62, "yield");
        put(63, "record");
        put(64, "sealed");
        put(65, "permits");
        put(66, "non-sealed");
        put(67, "12345");
        put(68, "0xABCD");
        put(69, "0756");
        put(70, "0b101010");
        put(71, "3.14f");
        put(72, "0x1.8p0f");
        put(73, "true");
        put(74, "\"A\"");
        put(75, "\"abc\"");
        put(76, "\"\"\" multiline text block \"\"\"\n");
        put(77, "null");
        put(78, "(");
        put(79, ")");
        put(80, "{");
        put(81, "}");
        put(82, "[");
        put(83, "]");
        put(84, ";");
        put(85, ",");
        put(86, ".");
        put(87, "=");
        put(88, ">");
        put(89, "<");
        put(90, "!");
        put(91, "~");
        put(92, "?");
        put(93, ":");
        put(94, "==");
        put(95, "<=");
        put(96, ">=");
        put(97, "!=");
        put(98, "&&");
        put(99, "||");
        put(100, "++");
        put(101, "--");
        put(102, "+");
        put(103, "-");
        put(104, "*");
        put(105, "/");
        put(106, "&");
        put(107, "|");
        put(108, "^");
        put(109, "%");
        put(110, "+=");
        put(111, "-=");
        put(112, "*=");
        put(113, "/=");
        put(114, "&=");
        put(115, "|=");
        put(116, "^=");
        put(117, "%=");
        put(118, "<<=");
        put(119, ">>=");
        put(120, ">>>=");
        put(121, "->");
        put(122, "::");
        put(123, "@");
        put(124, "...");
        put(125, " ");
        put(126, "/* comment */\n");
        put(127, "// line comment\n");
        put(128, "dummy");
    }};

    public static HashMap<String, Integer> reversePlaceholders = new HashMap<>() {{
        put("abstract", 1);
        put("assert", 2);
        put("boolean", 3);
        put("break", 4);
        put("byte", 5);
        put("case", 6);
        put("catch", 7);
        put("char", 8);
        put("class", 9);
        put( "const", 10);
        put( "continue", 11);
        put( "default", 12);
        put( "do", 13);
        put( "double", 14);
        put( "else", 15);
        put( "enum", 16);
        put( "extends", 17);
        put( "final", 18);
        put( "finally", 19);
        put( "float", 20);
        put( "for", 21);
        put( "if", 22);
        put( "goto", 23);
        put( "implements", 24);
        put( "import", 25);
        put( "instanceof", 26);
        put( "int", 27);
        put( "interface", 28);
        put( "long", 29);
        put( "native", 30);
        put( "new", 31);
        put( "package", 32);
        put( "private", 33);
        put( "protected", 34);
        put( "public", 35);
        put( "return", 36);
        put( "short", 37);
        put( "static", 38);
        put( "strictfp", 39);
        put( "super", 40);
        put( "switch", 41);
        put( "synchronized", 42);
        put( "this", 43);
        put( "throw", 44);
        put( "throws", 45);
        put( "transient", 46);
        put( "try", 47);
        put( "void", 48);
        put( "volatile", 49);
        put( "while", 50);
        put( "module", 51);
        put( "open", 52);
        put( "requires", 53);
        put( "exports", 54);
        put( "opens", 55);
        put( "to", 56);
        put( "uses", 57);
        put( "provides", 58);
        put( "with", 59);
        put( "transitive", 60);
        put( "var", 61);
        put( "yield", 62);
        put( "record", 63);
        put( "sealed", 64);
        put( "permits", 65);
        put( "non-sealed", 66);
        put( "12345", 67);
        put( "0xABCD", 68);
        put( "0756", 69);
        put( "0b101010", 70);
        put( "3.14f", 71);
        put( "0x1.8p0f", 72);
        put( "true", 73);
        put( "\"A\"", 74);
        put( "\"abc\"", 75);
        put( "\"\"\" This is a multiline text block.\"\"\"\n", 76);
        put( "null", 77);
        put( "(", 78);
        put( ")", 79);
        put( "{", 80);
        put( "}", 81);
        put( "[", 82);
        put( "]", 83);
        put( ";", 84);
        put( ",", 85);
        put( ".", 86);
        put( "=", 87);
        put( ">", 88);
        put( "<", 89);
        put( "!", 90);
        put( "~", 91);
        put( "?", 92);
        put( ":", 93);
        put( "==", 94);
        put( "<=>", 95);
        put( ">=", 96);
        put( "!=", 97);
        put( "&&", 98);
        put( "||", 99);
        put("++", 100);
        put("--", 101);
        put("+", 102);
        put("-", 103);
        put("*", 104);
        put("/", 105);
        put("&", 106);
        put("|", 107);
        put("^", 108);
        put("%", 109);
        put("+=", 110);
        put("-=", 111);
        put("*=", 112);
        put("/=", 113);
        put("&=", 114);
        put("|=", 115);
        put("^=", 116);
        put("%=", 117);
        put("<<=", 118);
        put(">>=", 119);
        put(">>>=", 120);
        put("->", 121);
        put(";;", 122);
        put("@", 123);
        put("...", 124);
        put(" ", 125);
        put("/* comment */\n", 126);
        put("// line comment\n", 127);
        put("dummy", 128);
    }};

    public static String replaceSourceWithDummyTokens(int[] encodedSourceCode) {
        List<String> dummySource = new ArrayList<>();
        for (int i = 0; i < encodedSourceCode.length; i++) {
            String token = placeholders.get(encodedSourceCode[i]);
            if (token == null) {
                // EOF
                continue;
            }
            if (encodedSourceCode[i] == 128) { // identifier
                token = generateRandomString(8);
            }
            // capitalize identifier if class token precedes it
            if (i > 0 && encodedSourceCode[i-1] == 9) { // class
                dummySource.add(token.toUpperCase());
            }
            else {
                dummySource.add(token);
            }
        }
        return String.join(" ", dummySource);
    }

    private static String generateRandomString(int length) {
        StringBuilder stringBuilder = new StringBuilder(length);
        SecureRandom random = new SecureRandom();
        String allowedCharacters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        for (int i = 0; i < length; i++) {
            int randomIndex = random.nextInt(allowedCharacters.length());
            char randomChar = allowedCharacters.charAt(randomIndex);
            stringBuilder.append(randomChar);
        }
        return stringBuilder.toString();
    }
}
