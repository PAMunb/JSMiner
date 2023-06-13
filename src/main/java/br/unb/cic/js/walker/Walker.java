package br.unb.cic.js.walker;

import br.unb.cic.js.date.Interval;
import br.unb.cic.js.miner.metrics.Summary;
import lombok.Builder;
import lombok.val;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

@Builder
public class Walker {

    private static final Logger logger = LogManager.getLogger(Walker.class);
    public final String path;
    public final String project;
    public final int steps;
    public final int threads;
    public final Date initialDate;
    public final Date endDate;

    public void traverse() {
        logger.info("initializing git traversal");
        logger.info(
                "path: {} | project: {} | steps: {} | threads: {} | initial date: {} | end date: {}",
                path,
                project,
                steps,
                threads,
                initialDate,
                endDate
        );

        val f = new File(path);
        val p = Path.of(path);

        try {
            if (f.exists() && f.isDirectory()) {
                List<Path> repositories = new ArrayList<>();

                // checking a file attribute to verify if it's a directory is slow, be careful with the amount of
                // folders you'll be checking against.

                if (project.isEmpty()) {
                    repositories.addAll(Files.find(p, 1, (path, attrs) -> {
                        val isDirectory = attrs.isDirectory();
                        val isGitDirectory = path.resolve(".git").toFile().isDirectory();

                        return isDirectory && isGitDirectory;
                    }).collect(Collectors.toList()));
                } else {
                    repositories.addAll(Files.find(p, 1, (path, attrs) -> {
                        val pathParts = path.toString().split("/");

                        val isEqualPath = pathParts[pathParts.length-1].equals(project);
                        val isDirectory = attrs.isDirectory();
                        val isGitDirectory = path.resolve(".git").toFile().isDirectory();

                        return isEqualPath && isDirectory && isGitDirectory;
                    }).collect(Collectors.toList()));
                }

                if (repositories.isEmpty()) {
                    logger.info("couldn't find any git folder in {}", p);
                    return;
                }

                // create a report directory and file that will contain the results
                val output = Paths.get(p.toAbsolutePath().getParent().toString(), "js-miner-out");
                if (!output.toFile().exists()) {
                    Files.createDirectory(output);
                }

                val results = output.resolve("results.csv");
                val writer = new BufferedWriter(new FileWriter(results.toFile()));

                writer.write(Summary.header());

                for (Path repositoryPath: repositories) {
                    val repositoryPathSplit = repositoryPath.toString().split("/");
                    val repositoryName = repositoryPathSplit[repositoryPathSplit .length-1];

                    logger.info("project: {}", repositoryName);

                    val walker = RepositoryWalker.builder()
                            .path(repositoryPath)
                            .project(repositoryName)
                            .build();


                    val interval = Interval.builder()
                            .begin(initialDate)
                            .end(endDate)
                            .build();

                    val task = RepositoryWalkerTask.builder()
                            .walker(walker)
                            .results(writer)
                            .output(output)
                            .interval(interval)
                            .steps(steps)
                            .build();

                    task.run();
                }

                // flush any pending text and close the results.csv writer
                writer.flush();
                writer.close();
            } else {
                logger.warn("path {} does not exist or isn't a directory", p);
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
}
