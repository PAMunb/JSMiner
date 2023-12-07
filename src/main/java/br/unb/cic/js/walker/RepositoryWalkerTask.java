package br.unb.cic.js.walker;

import br.unb.cic.js.date.Interval;
import br.unb.cic.js.miner.metrics.Summary;
import lombok.Builder;
import lombok.val;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;

@Builder
public final class RepositoryWalkerTask implements Runnable {
    private final Logger logger = LoggerFactory.getLogger(getClass());

    // The report is the csv file generated by the program after all observations of a given project are finished.
    public final Path output;

    public final RepositoryWalker walker;

    public final Interval interval;

    // Threads represents the number of visitors working concurrently on a given revision
    public final int threads;

    public final int steps;

    // Option to allow the walker to collect metrics about a single point in a given repository
    public final String hash;

    @Override
    public void run() {
        // build report and report errors file
        val reportFile = Paths.get(output.toString(), walker.project + ".csv");
        val reportErrors = Paths.get(output.toString(), walker.project + "-errors.txt");

        val content = new StringBuilder();
        val errors = new StringBuilder();

        try {
            var summaries = Collections.synchronizedList(new ArrayList<Summary>());

            if (hash.length() > 0) {
                summaries = walker.traverse(interval, hash, threads);
            } else {
                summaries = walker.traverse(interval, steps, threads);
            }

            summaries.forEach(s -> {
                if (!s.values().isEmpty()) {
                    content.append(s.values())
                            .append("\n");
                }

                s.errors.forEach((k, v) -> {
                    errors.append(k)
                            .append("\n")
                            .append(v)
                            .append("\n-----------------------\n");
                });
            });

            reportFile.toFile().createNewFile();
            reportErrors.toFile().createNewFile();

            try (val reportWriter = new BufferedWriter(new FileWriter(reportFile.toFile()));
                 val errorsWriter = new BufferedWriter(new FileWriter(reportErrors.toFile()))) {

                reportWriter.write(Summary.header());
                reportWriter.write(content.toString());
                reportWriter.flush();

                errorsWriter.write(errors.toString());
                errorsWriter.flush();
            } catch (IOException ex) {
                logger.error("failed to write on report/errors file for project {}", walker.project);
                throw new RuntimeException(ex);
            }
        } catch (IOException ex) {
            logger.error("failed to create a report/errors file for project {}", walker.project);
        } catch (Exception ex) {
            logger.error("failed to traverse project {}, reason {}", walker.project, ex.getMessage());
        }
    }
}
