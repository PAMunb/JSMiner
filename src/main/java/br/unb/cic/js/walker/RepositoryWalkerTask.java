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
import java.lang.ref.WeakReference;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.ConcurrentLinkedQueue;

@Builder
public class RepositoryWalkerTask implements Runnable {
    private final Logger logger = LoggerFactory.getLogger(getClass());

    public final Path output;
    public final BufferedWriter results;
    public final RepositoryWalker walker;
    public final Interval interval;
    public final int threads;
    public final int steps;

    @Override
    public void run() {
        val reportFile = Paths.get(output.toString(), walker.project + ".csv");
        val reportErrors = Paths.get(output.toString(), walker.project + "-errors.txt");

        try {
            // Mantenha summaries como ConcurrentLinkedQueue<Summary>
            ConcurrentLinkedQueue<WeakReference<Summary>> weakSummaries = new ConcurrentLinkedQueue<>();

            // Preencha weakSummaries com referências fracas para cada resumo
            walker.traverse(interval.begin, interval.end, steps, threads)
                    .forEach(summary -> weakSummaries.add(new WeakReference<>(summary)));

            val content = new StringBuilder();
            val errors = new StringBuilder();

            weakSummaries.forEach(weakSummary -> {
                val summary = weakSummary.get();

                if (summary != null) {
                    content.append(summary.values()).append("\n");

                    val errorsMap = summary.errors.get();

                    if (errorsMap != null) {
                        errorsMap.forEach((k, v) -> {
                            errors.append(k).append("\n").append(v).append("\n-----------------------\n");
                        });
                    }
                }
            });

            try (val reportWriter = new BufferedWriter(new FileWriter(reportFile.toFile()));
                    val errorsWriter = new BufferedWriter(new FileWriter(reportErrors.toFile()))) {

                reportWriter.write(Summary.header());
                reportWriter.write(content.toString());
                reportWriter.flush();

                errorsWriter.write(errors.toString());
                errorsWriter.flush();
            } catch (IOException ex) {
                logger.error("failed to write on report/errors file for project {}", walker.project);
            }

        } catch (IOException ex) {
            logger.error("failed to create a report/errors file for project {}", walker.project);
        } catch (Exception ex) {
            logger.error("failed to traverse project {}", walker.project);
        }
    }
}