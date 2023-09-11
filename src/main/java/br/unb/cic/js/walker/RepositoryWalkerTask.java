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
import java.util.concurrent.ConcurrentLinkedQueue;

@Builder
public class RepositoryWalkerTask implements Runnable {
    private final Logger logger = LoggerFactory.getLogger(getClass());

    public final Path output;
    public final RepositoryWalker walker;
    public final Interval interval;
    public final int threads;
    public final int steps;
    public final BufferedWriter results;

    @Override
    public void run() {
        val reportFile = Paths.get(output.toString(), walker.project + ".csv");
        val reportErrors = Paths.get(output.toString(), walker.project + "-errors.txt");

        try {
            ConcurrentLinkedQueue<Summary> summaries = new ConcurrentLinkedQueue<>();

            walker.traverse(interval.begin, interval.end, steps, threads)
                    .stream()
                    .filter(summary -> summary != null)
                    .forEach(summaries::add);

            try (val reportWriter = new BufferedWriter(new FileWriter(reportFile.toFile()));
                 val errorsWriter = new BufferedWriter(new FileWriter(reportErrors.toFile()))) {

                reportWriter.write(Summary.header());

                summaries.forEach(summary -> {
                    try {
                        if (summary != null) {
                            reportWriter.write(summary.values() + "\n");
                        }
                    } catch (IOException e) {
                        logger.error("Failed to write summary data for project {}", walker.project);
                        e.printStackTrace();
                    }
                });

                reportWriter.flush();

                errorsWriter.write(summariesToErrors(summaries));
                errorsWriter.flush();
            } catch (IOException ex) {
                logger.error("Failed to write on report/errors file for project {}", walker.project);
                ex.printStackTrace();
            }

        } catch (IOException ex) {
            logger.error("Failed to create a report/errors file for project {}", walker.project);
            ex.printStackTrace();
        } catch (Exception ex) {
            logger.error("Failed to traverse project {}", walker.project);
            ex.printStackTrace();
        }
    }

    private String summariesToErrors(ConcurrentLinkedQueue<Summary> summaries) {
        StringBuilder errors = new StringBuilder();
        
        summaries.forEach(summary -> {
            if (summary.errors != null) {
                summary.errors.forEach((k, v) -> {
                    errors.append(k).append("\n").append(v).append("\n-----------------------\n");
                });
            }
        });

        return errors.toString();
    }
}