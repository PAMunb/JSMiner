package br.unb.cic.js.walker;

import br.unb.cic.js.date.Interval;
import lombok.Builder;
import lombok.val;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

/**
 * The entire logic of the miner is verified, build and sent to execution here.
 */
@Builder
public final class Walker {

    private static final Logger logger = LogManager.getLogger(Walker.class);

    public final String path;
    public final String project;
    public final String hash;
    public final int steps;
    public final int projectThreads;
    public final int filesThreads;
    public final Date initialDate;
    public final Date endDate;
    public final Boolean merges;

    public void traverse() {
        logger.info("initializing git traversal");
        
		logger.info(
                "path: {} | project: {} | steps: {} | project threads: {} |  files threads: {} | initial date: {} | end date: {} | merges: {}",
                path,
                project,
                steps,
                projectThreads,
                filesThreads,
                initialDate,
                endDate,
                merges);

        val f = new File(path);
        val p = Path.of(path);

        if (!f.exists() || !f.isDirectory()) {
            logger.warn("path {} does not exist or isn't a directory", p);

            return;
        }

        try {
            List<Path> repositories = new ArrayList<>();

            // checking a file attribute to verify if it's a directory is slow, be careful
            // with the amount of
            // folders you'll be checking against.

            if (project.isEmpty()) {
                repositories.addAll(Files.find(p, 1, (path, attrs) -> {
                    val isDirectory = attrs.isDirectory();
                    val isGitDirectory = path.resolve(".git").toFile().isDirectory();

                    return isDirectory && isGitDirectory;
                }).collect(Collectors.toList()));
            } else {
                // allow to glob more than one project on specification
                val projects = project.split(",");

                repositories.addAll(Files.find(p, 1, (path, attrs) -> {
                    val pathParts = path.toString().split("/");

                    var isEqualPath = Arrays.stream(projects)
                            .anyMatch(project -> pathParts[pathParts.length - 1].equals(project));

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
            val output = Paths.get(p.toAbsolutePath().getParent().toString(), "../jsminer-out");
            if (!output.toFile().exists()) {
                Files.createDirectory(output);
            }

            // check if a hash has been submitted, it will invalidate almost all of the
            // settings and execute the
            // walker for a single commit hash.
            if (hash.length() > 0) {
                assert (repositories.size() == 1);
            }

            val pool = Executors.newFixedThreadPool(projectThreads);
            val tasks = new Vector<Future<?>>();

            for (Path repositoryPath : repositories) {
                val repositoryPathSplit = repositoryPath.toString().split("/");
                val repositoryName = repositoryPathSplit[repositoryPathSplit.length - 1];

                logger.info("project: {}", repositoryName);

                val walker = RepositoryWalker.builder()
                        .path(repositoryPath)
                        .project(repositoryName)
                        .merges(merges)
                        .build();

                val interval = Interval.builder()
                        .begin(initialDate)
                        .end(endDate)
                        .build();

                val task = RepositoryWalkerTask.builder()
                        .walker(walker)
                        .output(output)
                        .interval(interval)
                        .steps(steps)
                        .hash(hash)
                        .threads(filesThreads)
                        .build();

                tasks.add(pool.submit(task));
            }

            // wait for every task to finish
            for (Future<?> task : tasks) {
                task.get();
            }

            pool.shutdown();
        } catch (IOException ex) {
            ex.printStackTrace();
        } catch (java.lang.InterruptedException | java.util.concurrent.ExecutionException ex) {
            logger.error("failed to execute a concurrent task, reason {}", ex.getMessage());
            ex.printStackTrace();
        }
    }
}
