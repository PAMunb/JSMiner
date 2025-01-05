package br.unb.cic.js;

import br.unb.cic.js.date.Formatter;
import br.unb.cic.js.walker.Walker;
import com.beust.jcommander.JCommander;
import lombok.val;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.text.ParseException;

public class App {

    public static final String FAILED = "failed to parse date arguments";

    public static void main(String[] args) {
        val logger = LoggerFactory.getLogger(App.class);

        val arguments = new Args();

        val cli = JCommander.newBuilder()
                .addObject(arguments)
                .build();

        try {
            cli.parse(args);
        } catch (RuntimeException ex) {
            cli.usage();
        }

        try {
            val walker = Walker.builder()
            		.merges(arguments.merges)
                    .path(arguments.directory)
                    .project(arguments.project)
                    .steps(arguments.steps)
                    .hash(arguments.hash)
                    .projectThreads(arguments.threadsProjects)
                    .filesThreads(arguments.threadsFiles)
                    .initialDate(Formatter.format.parse(arguments.initialDate))
                    .endDate(Formatter.format.parse(arguments.endDate))
                    .build();

            walker.traverse();
        } catch (ParseException | IOException ex) {
            logger.error(FAILED);
        } catch (InterruptedException e) {
            logger.error(FAILED);
            Thread.currentThread().interrupt();
        }
    }
}
