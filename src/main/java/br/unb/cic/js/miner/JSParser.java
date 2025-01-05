package br.unb.cic.js.miner;

import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.ParseCancellationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class JSParser {
    private static final Logger log = LoggerFactory.getLogger(JSParser.class);

    private final ExceptionBasedErrorListener errorListener;

    public JSParser() {
        this.errorListener = new ExceptionBasedErrorListener();
    }

    public JavaScriptParser.ProgramContext parse(String content) {
        JavaScriptParser parser = proccessContent(content);
        return parser.program();
    }

    public void printParseTree(String content) {
        JavaScriptParser parser = proccessContent(content);

        parser.setBuildParseTree(true);
        RuleContext tree = parser.program();

        if (log.isInfoEnabled()) {
            log.info(tree.toStringTree(parser));
        }
    }

    private JavaScriptParser proccessContent(String content) {
        CharStream charStream = CharStreams.fromString(content);
        JavaScriptLexer lexer = new JavaScriptLexer(charStream);
        JavaScriptParser parser = new JavaScriptParser(new CommonTokenStream(lexer));

        lexer.removeErrorListeners();
        lexer.addErrorListener(errorListener);
        parser.removeErrorListeners();
        parser.addErrorListener(errorListener);

        return parser;
    }

    static class ExceptionBasedErrorListener extends BaseErrorListener {
        @Override
        public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol, int line, int charPositionInLine, String msg, RecognitionException e) {
            throw new ParseCancellationException(String.format("line: %d : %d - %s", line, charPositionInLine, msg));
        }
    }
}

