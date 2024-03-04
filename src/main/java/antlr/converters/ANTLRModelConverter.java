package antlr.converters;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class ANTLRModelConverter {
    private static HashMap<Integer, Integer> MODEL_2_ANTLR_IDS = new HashMap<>() {{
        put(3, 9);
        put(4, 39);
        put(5, 28);
        put(6, 46);
        put(7, 70);
        put(8, 21);
        put(9, 77);
        put(10, 17);
        put(11, 81);
        put(12, 79);
        put(13, 37);
        put(14, 43);
        put(15, 83);
        put(16, 71);
        put(17, 1);
        put(18, 2);
        put(19, 65);
        put(20, 75);
        put(21, 84);
        put(22, 100);
        put(23, 85);
        put(24, 42);
        put(25, 115);
        put(26, 60);
        put(27, 126);
        put(28, 56);
        put(29, 31);
        put(30, 22);
        put(31, 23);
        put(32, 114);
        put(33, 54);
        put(34, 50);
        put(35, 74);
        put(36, 27);
        put(37, 107);
        put(38, 119);
        put(39, 48);
        put(40, 45);
        put(41, 18);
        put(42, 82);
        put(43, 102);
        put(44, 62);
        put(45, 26);
        put(46, 15);
        put(47, 92);
        put(48, 96);
        put(49, 36);
        put(50, 118);
        put(51, 4);
        put(52, 34);
        put(53, 72);
        put(54, 8);
        put(55, 91);
        put(56, 11);
        put(57, 67);
        put(58, 101);
        put(59, 109);
        put(60, 120);
        put(61, 73);
        put(62, 106);
        put(63, 25);
        put(64, 105);
        put(65, 69);
        put(66, 76);
        put(67, 32);
        put(68, 13);
        put(69, 59);
        put(70, 53);
        put(71, 40);
        put(72, 110);
        put(73, 51);
        put(74, 68);
        put(75, 112);
        put(76, 88);
        put(77, 57);
        put(78, 55);
        put(79, 41);
        put(80, 87);
        put(81, 6);
        put(82, 24);
        put(83, 52);
        put(84, 108);
        put(85, 122);
        put(86, 99);
        put(87, 33);
        put(88, 35);
        put(89, 113);
        put(90, 44);
        put(91, 116);
        put(92, 61);
        put(93, 30);
        put(94, 89);
        put(95, 80);
        put(96, 38);
        put(97, 12);
        put(98, 90);
        put(99, 123);
        put(100, 20);
        put(101, 19);
        put(102, 127);
        put(103, 47);
        put(104, 10);
        put(105, 7);
        put(106, 63);
        put(107, 104);
        put(108, 111);
        put(109, 93);
        put(110, 49);
        put(111, 97);
        put(112, 86);
        put(113, 66);
        put(114, 78);
        put(115, 14);
        put(116, 3);
        put(117, 98);
        put(118, 64);
        put(119, 5);
        put(120, 117);
        put(121, 58);
        put(122, 29);
        put(123, 94);
        put(124, 16);
        put(125, 124);
        put(126, 103);
        put(127, 121);
        put(128, 95);
        put(129, 128);
    }};
    private static HashMap<Integer, Integer> ANTLR_TO_MODEL_IDS = new HashMap<>() {{
        put(9, 3);
        put(39, 4);
        put(28, 5);
        put(46, 6);
        put(70, 7);
        put(21, 8);
        put(77, 9);
        put(17, 10);
        put(81, 11);
        put(79, 12);
        put(37, 13);
        put(43, 14);
        put(83, 15);
        put(71, 16);
        put(1, 17);
        put(2, 18);
        put(65, 19);
        put(75, 20);
        put(84, 21);
        put(100, 22);
        put(85, 23);
        put(42, 24);
        put(115, 25);
        put(60, 26);
        put(126, 27);
        put(56, 28);
        put(31, 29);
        put(22, 30);
        put(23, 31);
        put(114, 32);
        put(54, 33);
        put(50, 34);
        put(74, 35);
        put(27, 36);
        put(107, 37);
        put(119, 38);
        put(48, 39);
        put(45, 40);
        put(18, 41);
        put(82, 42);
        put(102, 43);
        put(62, 44);
        put(26, 45);
        put(15, 46);
        put(92, 47);
        put(96, 48);
        put(36, 49);
        put(118, 50);
        put(4, 51);
        put(34, 52);
        put(72, 53);
        put(8, 54);
        put(91, 55);
        put(11, 56);
        put(67, 57);
        put(101, 58);
        put(109, 59);
        put(120, 60);
        put(73, 61);
        put(106, 62);
        put(25, 63);
        put(105, 64);
        put(69, 65);
        put(76, 66);
        put(32, 67);
        put(13, 68);
        put(59, 69);
        put(53, 70);
        put(40, 71);
        put(110, 72);
        put(51, 73);
        put(68, 74);
        put(112, 75);
        put(88, 76);
        put(57, 77);
        put(55, 78);
        put(41, 79);
        put(87, 80);
        put(6, 81);
        put(24, 82);
        put(52, 83);
        put(108, 84);
        put(122, 85);
        put(99, 86);
        put(33, 87);
        put(35, 88);
        put(113, 89);
        put(44, 90);
        put(116, 91);
        put(61, 92);
        put(30, 93);
        put(89, 94);
        put(80, 95);
        put(38, 96);
        put(12, 97);
        put(90, 98);
        put(123, 99);
        put(20, 100);
        put(19, 101);
        put(127, 102);
        put(47, 103);
        put(10, 104);
        put(7, 105);
        put(63, 106);
        put(104, 107);
        put(111, 108);
        put(93, 109);
        put(49, 110);
        put(97, 111);
        put(86, 112);
        put(66, 113);
        put(78, 114);
        put(14, 115);
        put(3, 116);
        put(98, 117);
        put(64, 118);
        put(5, 119);
        put(117, 120);
        put(58, 121);
        put(29, 122);
        put(94, 123);
        put(16, 124);
        put(124, 125);
        put(103, 126);
        put(121, 127);
        put(95, 128);
        put(128, 129);
    }};


    public static int[] encodeSequence(int[] sequence) {
        int lookup = 0;
        List<Integer> encodedSequence = new ArrayList<>();
        for (int i : sequence) {
            lookup+=1;
            if (i == 0) {
                // SKIP SOS
                continue;
            }
            if (i == 1) {
                // stop at EOS
                if (lookup == sequence.length) {
                    break;
                }
                encodedSequence.add(77); // null fallback
            }
            else {
                encodedSequence.add(MODEL_2_ANTLR_IDS.get(i));
            }

        }
        return encodedSequence.stream()
                .mapToInt(Integer::intValue)
                .toArray();
    }

    public static int[] decodeSequence(int[] sequence) {
        List<Integer> encodedSequence = new ArrayList<>();
        for (int i : sequence) {
            encodedSequence.add(ANTLR_TO_MODEL_IDS.get(i));
        }
        return encodedSequence.stream()
                .mapToInt(Integer::intValue)
                .toArray();
    }
}
