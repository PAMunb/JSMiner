package br.unb.cic.js;

import br.unb.cic.js.date.Formatter;
import br.unb.cic.js.walker.Walker;
import com.beust.jcommander.JCommander;
import lombok.val;

import java.text.ParseException;

public class App {

    public static void main(String []args) {

        val arguments = new Args();

        val cli = JCommander.newBuilder()
                .addObject(arguments)
                .build();

        try {
            cli.parse(args);
        } catch(RuntimeException ex) {
            cli.usage();
        }

        try {
            val walker = Walker.builder()
                    .path(arguments.directory)
                    .project(arguments.project)
                    .steps(arguments.steps)
                    .initialDate(Formatter.format.parse(arguments.initialDate))
                    .endDate(Formatter.format.parse(arguments.endDate))
                    .build();

            walker.traverse();
        } catch(ParseException ex) {
            ex.printStackTrace();
        }
    }
}
