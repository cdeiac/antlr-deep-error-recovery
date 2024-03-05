import antlr.JavaLexer;
import antlr.JavaParser;
import antlr.evaluation.ErrorReporterVisitor;
import antlr.extensions.CustomErrorStrategy;
import antlr.extensions.CustomVisitor;
import antlr.extensions.DeepErrorStrategy;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.DefaultErrorStrategy;
import org.antlr.v4.runtime.RuleContext;
import org.antlr.v4.runtime.tree.ParseTree;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.logging.Logger;

@Disabled
public class JavaParserTest {

    private static final Logger logger = Logger.getLogger(JavaParserTest.class.getName());


    @Test
    public void identifier() {
        String javaClassContent = "int int i;ab;c";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        JavaParser javaParser = new JavaParser(tokens);
        javaParser.setTrace(true);
        javaParser.identifier();
    }

    @Test
    public void compilationUnit() {
        String javaClassContent = "class FjWDAEXx { public int [ ] quWBwEtG ( int [ ] CIoJDQSJ , int gevDEizi ) { int [ ] oDigUbJx = new int [ ] { - 12345 , - 12345 } ; int VAoIhEsX = 12345 ; int FvmAyREr = AtdtwovK . fSHLucwJ - 12345 ; int TiyzKcuB = Emldlxjb ; int MOqNWagu = UAUzXoxi ; while ( ZsXzlAMR < RUoFEqSv ) { int iWhRIOry = ( WcijzYTb + CnavVqHq ) / 12345 ; if ( AzQiwjyh [ QjbIuRnT ] < hWAJptBC ) { YXZLHWKp = ETJMArru + 12345 ; } else { XEOfPQIp = lsuDGArq ; } } if ( qwASSZMI > AFsirCyN || JBtPaJrL [ LifFUNuN ] != ArAIInBe ) { return new int [ ] { - 12345 , - 12345 } ; } while ( KuzyYoXI < AvqoZtHe ) { int DzNjwiCO = ( YXgDVJoR + tFwdqgNx ) 0xABCD 12345 + 12345 ; if ( DdDJRbWA [ dvCluBVi ] > ysGsxZmp ) { WHFgJJGr = Mayiavxn - 12345 ; } else { ZBSKiaaR = mGPyrUUB ; } if ( ZNQizvvO > AzbULUZI || gBjbRwnc > gzslYcMv ) { return new new [ ] { - 12345 , - 12345 } ; } else { return new int [ ] { PQCEYmQf , LxrLIoYZ } ; } } }";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        JavaParser javaParser = new JavaParser(tokens);
        //javaParser.setTrace(true);
        javaParser.setErrorHandler(new DefaultErrorStrategy());
        ErrorReporterVisitor visitor = new ErrorReporterVisitor();
        try {
            visitor.visit(javaParser.compilationUnit());
            System.out.println(visitor.getVisitedNodes());
        } catch (Exception e) {
            System.out.println(visitor.getVisitedNodes());
            e.printStackTrace();
        }
        System.out.println(visitor.getVisitedNodes());

    }

    @Test
    public void errorReporterVisitorTest() {
        String javaClassContent = "class MWKTFEJL { public KUmHmWuD exPyFemQ ( WohSuKpT YrKTWgvP ) { RjadaihB EWGXbMvg = new Kfgirgpo ( ) ; int GtfnSgSY = 12345 ; vceUJziv < cudaHAzY > LUPUIUyu = new aiFnGUWO ( ) ; MOZVKslu < tBkXLlIF > MjbnRzEb = new XUczHksJ ( ) ; for ( int nvwGiIya = 12345 ; dVIJCBlA < yDcPSXNn . lfEgCxRW ( ) ; qcSLsSfK ++ ) { char SbiDlisX = iInxqamR . stsNHZbW ( qOUEhwzh ) ; if ( \"abc\" == HOCDMJhL ) { lBJQqRdm . xsAFrIHH ( VhnQLZiV ) ; jvIZzGDN . iCnItHqM ( DXmVFCLe . BmDCMGAV ( ) ) ; hNyZFVhH = 12345 ; oebcnwGm = new pDIMnNrl ( ) ; } else if ( \"abc\" == lHNmraYy ) { IQBtfEvk tQEFzUhv = new ksUxarcT ( ) ; int QDjVdEXk = OdkjZlEB . LCZgOyyg ( ) ; for ( int otEDNcbl = 12345 ; lWXHvPOg < nLNfLzZZ ; OaUpjhrV ++ ) { bnrwrGDE . xhMzCUBZ ( mEhUwNlc ) ; } SEixDnqf = new aDJVXvEj ( MRhpoZtU . rsDjRCXg ( ) + FQcclUeo ) ; } else if ( xwNKzhKR >= \"abc\" && WeAgGTdo <=> > \"abc\" ) { IWfHAFHs = BwmyZKnF * 12345 + ( SVCnQtrB - \"abc\" ) ; } else { EbQXzbwO . egwqbXrz ( QDuzcSrX ) ; } } return HShXoFtS . CBMVdFeT ( ) ; } }\n";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        tokens.fill();
        JavaParser javaParser = new JavaParser(tokens);
        //javaParser.removeErrorListeners();
        //BaseJavaErrorListener errorListener = new BaseJavaErrorListener(tokens.getTokens());
        //javaParser.addErrorListener(errorListener);
        DefaultErrorStrategy errorStrategy = new DefaultErrorStrategy();
        javaParser.setErrorHandler(errorStrategy);
        javaParser.setTrace(true);
        ErrorReporterVisitor visitor = new ErrorReporterVisitor();
        var result = visitor.visit(javaParser.compilationUnit());
        System.out.println(visitor.getErrorNodes());
    }

    @Test
    public void realTest() {
        String javaClassContent = "class ALXWJKHZ { public JLKZfmXq nwGJQDab ( YbMgXtyA QNpdkVPJ ) { ZbBIdBFu uowOOpjm = new PnChdBro ( ) ; int DIwJScKf = 12345 ; ZWsRJTGp < LPzNRird > QHNscHuG = new hDwoApei ( ) ; FFKZPUnO < kMEzQdUr > hxtAnWhT = new JdBmGezS ( ) ; for ( int tjqoLqjl = 12345 ; eKizkMLd < rJSDgWVa . ufyowAeh ( ) ; fQoGzjAE ++ ) { char rDvlyUCo = vBoISVwy . nomyEtdb ( TNxJztIK ) ; if ( \"A\" == MNuaHWbo ) { nklaAywQ . ozeFBfxL ( hLppkKdM ) ; TfqJOEEk . vlUrLvuA ( iiBiPibN . rAlcaPPm ( ) ) ; tzqAgTLP = 12345 ; SCsAjjNU = new hnvzSsFx ( ) ; } else if ( \"A\" == nRNAnRin ) { DUDDQVfr POfhRNNg = new PklvMWUh ( ) ; int KYfXidbg = eMUsGGcO . jwsTMxGH ( ) ; for ( int NrdRgWlj = 12345 ; ulcsVfIy < uaOARQEH ; mRsmKEPX ++ ) { EOtMmwPR . tvHlCehn ( bBahYksK ) ; } iaCBGGvm = new GUXchdMx ( jidPBoPW . NLMqkygy ( ) + LcdpoLdz ) ; } else if ( bFghqunq >= \"A\" && jtiPmxnC <=> \"A\" ) { ZBFRatGL = YCHIenoi * 12345 + ( QvdWKXPx - \"A\" ) ; } else { SudWqnej . OVQPMwxU ( IoLkDmOs ) ; } } return APibbGhQ . jlkJAHGk ( ) ; } }";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        tokens.fill();
        String modelContent = "class RJQLCXGZ { public iuTSKtSA bmaWzXki ( ualSNBkI DIaLjfVp ) { RFyaoMqT QrOBFHyD = new RxxsTUqF ( ) ; int weeCykYZ = 12345 ; AutUWKRB < YEuDdmAc > InYKclDq = new vNFFIjKx ( ) ; ACeUADEB < IXzScuPL > NgQslhHe = new TeKAyHOE ( ) ; for ( int BNNDYnZq = 12345 ; lkBMMjAo < iwqdohnj . PcbIMsiG ( ) ; OkVuppqV ++ ) { char JPlVTrMc = bQABgFWv . mkXvyMTD ( yffFgIdW ) ; if ( \"A\" == ItIrNsMc ) { VyKKwWfO . TideLOAO ( bQPhemJj ) ; niUBcPLn . nLuWfNNx ( iUUYzDvo . OPBaFycZ ( ) ) ; jcNGTirq = 12345 ; VOloVSjX = new WoYvzYyH ( ) ; } else if ( \"A\" == rQgAbbET ) { vxsBazoN PBKLdkEE = new pzhhWnKj ( ) ; int egKfmdnk = SWMpkJuQ . hDcjsihb ( ) ; for ( int SGenDgaE = 12345 ; ozwTeBxQ < RpmTXnXx ; TRLcivgT ++ ) { yDSVAqnL . CbsCpCzW ( MqnHKooi ) ; } mklYTMFr = new zbAHFGwP ( CIwkbYwJ . rCqFLQja ( ) + NIchicAE ) ; } else if ( xUrzzToR >= \"A\" && JZYZdgre <=> \"A\" ) { rWgtzfxi = bOcxxuuo * 12345 + ( UPlcEGEE - \"A\" ) ; } else { bBqHzUNg . MXunyCJE ( vWMZldRi ) ; } } return glRDmxuQ . DUrzDYiO ( ) ; } }";
        JavaLexer modelLexer = new JavaLexer(CharStreams.fromString(modelContent));
        CommonTokenStream modelTokens = new CommonTokenStream(modelLexer);
        modelTokens.fill();
        JavaParser javaParser = new JavaParser(tokens);
        javaParser.removeErrorListeners();
        //JavaErrorListener errorListener = new JavaErrorListener(tokens.getTokens(), false);
        //javaParser.addErrorListener(errorListener);

        //CustomErrorListener errorListener = new CustomErrorListener(tokens);
        //javaParser.addErrorListener(errorListener);
        DeepErrorStrategy errorStrategy = new DeepErrorStrategy(tokens.getTokens(), modelTokens.getTokens(), null);
        javaParser.setErrorHandler(errorStrategy);
        javaParser.setTrace(true);
        ErrorReporterVisitor visitor = new ErrorReporterVisitor();
        //var result = visitor.visit(javaParser.compilationUnit());
        javaParser.compilationUnit();
        System.out.println(errorStrategy.getRecoveredTokens());
    }

    @Test
    public void compilationUnitWithTokenStreamRewriter() {
        String javaClassContent = "public class Test { public static void main(String[] args) { System.out.println(\"Hello, World!\");} }";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        CustomErrorStrategy errorStrategy = new CustomErrorStrategy();
        tokens.fill();
        System.out.println("Token fill: " + tokens.getTokens().size());
        JavaParser javaParser = new JavaParser(tokens);
        ErrorReporterVisitor visitor = new ErrorReporterVisitor();
        //CustomParseTreeListener listener = new CustomParseTreeListener();
        //javaParser.addParseListener(listener);
        javaParser.setTrace(false);
        javaParser.setBuildParseTree(true);

        javaParser.setErrorHandler(errorStrategy);
        visitor.visit(javaParser.compilationUnit());
        System.out.println(errorStrategy.getSkippedTokenIndexes());
        System.out.println("Without ER: " + ((double) errorStrategy.getSkippedTokenIndexes().get(0) / tokens.size()));
        System.out.println("With ER: " + (1.0 - ((double) errorStrategy.getSkippedTokenIndexes().size() / tokens.size())));
    }

    @Test
    public void method_name_starts_with_number_causes_error() {
        String javaClassContent = "public class SampleClass { void doSomething(){ int t = 9; int ;ab;;h;9;; int n = 1; } }";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        JavaParser javaParser = new JavaParser(tokens);
        javaParser.setTrace(true);
        //this.parseAndPrintParseTree(javaParser);
        //TokenStreamListener listener = new TokenStreamListener(tokens);
        //ParseTreeWalker.DEFAULT.walk(listener, javaParser.compilationUnit());
        ParseTree parseTree = javaParser.compilationUnit();
        //logger.info("Replaced code: " + listener.getReplacedCode());
        //javaLexer.setLine(0);
        //javaLexer.setCharPositionInLine(0);
        //javaLexer.reset();
        //javaParser.setInputStream(tokens);
        //javaParser.setTokenStream(tokens);
        //logger.info("index " + tokens.index());
        //javaParser.reset();
        //ParseTreeWalker.DEFAULT.walk(listener, javaParser.compilationUnit());
    }

    @Test
    public void test_with_visitor() {
        String javaClassContent = "public class SampleClass { void 1doSomething(){} }";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        JavaParser javaParser = new JavaParser(tokens);
        javaParser.setTrace(true);
        CustomVisitor customVisitor = new CustomVisitor();
        customVisitor.visit(javaParser.compilationUnit());
    }

    @Test
    public void test_parseTreePattern() {
        String javaClassContent = "public class SampleClass { void 1doSomething(){} }";
        JavaLexer javaLexer = new JavaLexer(CharStreams.fromString(javaClassContent));
        CommonTokenStream tokens = new CommonTokenStream(javaLexer);
        JavaParser javaParser = new JavaParser(tokens);
        javaParser.setTrace(true);
        this.parseAndPrintParseTree(javaParser);

    }

    private void parseAndPrintParseTree(JavaParser parser) {
        parser.setBuildParseTree(true);
        RuleContext tree = parser.compilationUnit();
        List<String> ruleNamesList = List.of(parser.getRuleNames());
        String prettyTree = TreeUtils.toPrettyTree(tree, ruleNamesList);
        logger.info(prettyTree);
    }
}