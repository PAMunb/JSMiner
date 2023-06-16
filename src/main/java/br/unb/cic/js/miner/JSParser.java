package br.unb.cic.js.miner;

import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.ParseCancellationException;

public class JSParser {
    public JavaScriptParser.ProgramContext parse(String content) throws Exception {
    	CharStream charStream = CharStreams.fromString(content);
        ExceptionBasedErrorListener listener = new ExceptionBasedErrorListener();
        JavaScriptLexer lexer = configureLexer(charStream, listener);
        JavaScriptParser parser = configureParser(lexer, listener);

        return parser.program();
    }

    private JavaScriptParser configureParser(JavaScriptLexer lexer, ExceptionBasedErrorListener listener) {
        JavaScriptParser parser = new JavaScriptParser(new CommonTokenStream(lexer));
        parser.removeErrorListeners();
        parser.addErrorListener(listener);
        return parser;
    }

    private JavaScriptLexer configureLexer(CharStream stream, ExceptionBasedErrorListener listener) {
        JavaScriptLexer lexer = new JavaScriptLexer(stream);
        lexer.removeErrorListeners();
        lexer.addErrorListener(new ExceptionBasedErrorListener());
        return lexer;
    }

    static class ExceptionBasedErrorListener extends BaseErrorListener {
        @Override
        public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol, int line, int charPositionInLine, String msg, RecognitionException e) {
            throw new ParseCancellationException(String.format("line: %d : %d - %s ", line, charPositionInLine, msg));
        }
    }
}

