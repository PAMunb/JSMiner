package br.unb.cic.js.walker;

import br.unb.cic.js.date.Interval;
import br.unb.cic.js.miner.metrics.Summary;
import lombok.Builder;
import lombok.val;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.nio.file.Path;
import java.nio.file.Paths;

@Builder
public class RepositoryWalkerTask implements Runnable {
    private final Logger logger = LoggerFactory.getLogger(getClass());

    // The report is the csv file generated by the program after all observations of a given project are finished.
    public final Path output;

    public final BufferedWriter results;

    public final RepositoryWalker walker;

    public final Interval interval;

    // Threads represents the number of visitors working concurrently on a given revision
    public final int threads;

    public final int steps;

    @Override
    public void run() {
        try {
            val summaries = walker.traverse(interval.begin, interval.end, steps, threads);

            // build report and report errors file
            val reportFile = Paths.get(output.toString(), walker.project + ".csv");
            val reportErrors = Paths.get(output.toString(), walker.project + "-errors.txt");

            reportFile.toFile().createNewFile();
            reportErrors.toFile().createNewFile();

            val content = new StringBuilder();
            val errors = new StringBuilder();
            
            summaries.forEach(s -> {
                        content.append(s.values())
                       .append("\n");

                        s.errors.forEach((k, v) -> {
                            errors.append(k)
                                    .append("\n")
                                    .append(v)
                                    .append("\n-----------------------\n");
                        });
            });

            val reportWriter = new BufferedWriter(new FileWriter(reportFile.toFile()));
            val errorsWriter = new BufferedWriter(new FileWriter(reportErrors.toFile()));

            reportWriter.write(Summary.header());
            reportWriter.write(content.toString());
            reportWriter.flush();

            errorsWriter.write(errors.toString());
            errorsWriter.flush();

            synchronized (results) {
                results.write(content.toString());
                results.flush();
            }

            reportWriter.close();
            errorsWriter.close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
