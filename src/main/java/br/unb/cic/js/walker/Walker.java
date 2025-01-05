package br.unb.cic.js.walker;

import br.unb.cic.js.date.Interval;
import lombok.Builder;
import lombok.val;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.*;
import java.util.concurrent.*;

/**
 * The entire logic of the miner is verified, built, and sent to execution here.
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

    public void traverse() throws IOException, InterruptedException {
        logger.info("Initializing git traversal");

        logger.info(
                "Path: {} | Project: {} | Steps: {} | Project threads: {} | Files threads: {} | Initial date: {} | End date: {} | Merges: {}",
                path, project, steps, projectThreads, filesThreads, initialDate, endDate, merges);

        val projectPath = Path.of(path);

        if (!Files.exists(projectPath) || !Files.isDirectory(projectPath)) {
            logger.warn("Path {} does not exist or is not a directory", projectPath);
            return;
        }

        List<Path> repositories = findRepositories(projectPath);

        if (repositories.isEmpty()) {
            logger.info("No git repositories found in {}", projectPath);
            return;
        }

        Path output = setupOutputDirectory(projectPath);

        if (hash != null && !hash.isEmpty() && repositories.size() != 1) {
                throw new IllegalStateException("Hash mode requires exactly one repository");
            }


        int adjustedThreads = Math.min(projectThreads, Runtime.getRuntime().availableProcessors());
        ExecutorService pool = Executors.newFixedThreadPool(adjustedThreads);

        getRepositories(repositories, pool, output);

    }

    private void getRepositories(List<Path> repositories, ExecutorService pool, Path output) throws InterruptedException {
        try {
            List<Future<?>> tasks = new ArrayList<>();

            for (Path repositoryPath : repositories) {
                tasks.add(createTask(pool, repositoryPath, output));
            }

            for (Future<?> task : tasks) {
                task.get();
            }

        } catch (InterruptedException | ExecutionException e) {
            logger.error("Failed to execute a concurrent task", e);
            Thread.currentThread().interrupt();
        } finally {
            pool.shutdown();
            if (!pool.awaitTermination(60, TimeUnit.SECONDS)) {
                pool.shutdownNow();
            }
            repositories.clear();
        }
    }

    private List<Path> findRepositories(Path basePath) throws IOException {
        List<Path> repositories = new ArrayList<>();

        if (project.isEmpty()) {
            Files.walkFileTree(basePath, EnumSet.noneOf(FileVisitOption.class), Integer.MAX_VALUE, new SimpleFileVisitor<>() {
                @Override
                public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) {
                    if (Files.isDirectory(dir.resolve(".git"))) {
                        repositories.add(dir);
                    }
                    return FileVisitResult.CONTINUE;
                }
            });
        } else {
            Set<String> projectsSet = new HashSet<>(Arrays.asList(project.split(",")));
            for (String repository : projectsSet) {
                Files.walkFileTree(basePath.resolve(repository), EnumSet.noneOf(FileVisitOption.class), 1, new SimpleFileVisitor<>() {
                    @Override
                    public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) {
                        String dirName = dir.getFileName().toString();
                        if (Files.isDirectory(dir.resolve(".git"))) {
                            Path fullPath = basePath.resolve(dirName);
                            repositories.add(fullPath);
                        }
                        return FileVisitResult.CONTINUE;
                    }
                });
            }
        }

        return repositories;
    }

    private Path setupOutputDirectory(Path basePath) throws IOException {
        Path output = basePath.getParent().resolve("../jsminer-out").normalize();
        if (!Files.exists(output)) {
            Files.createDirectory(output);
        }
        return output;
    }

    private Future<?> createTask(ExecutorService pool, Path repositoryPath, Path output) {
        String repositoryName = repositoryPath.getFileName().toString();

        logger.info("Processing project: {}", repositoryName);

        val walker = RepositoryWalker.builder()
                .path(repositoryPath)
                .project(repositoryName)
                .merges(merges)
                .build();

        val interval = Interval.builder()
                .begin(initialDate)
                .end(endDate)
                .build();

        return pool.submit(RepositoryWalkerTask.builder()
                .walker(walker)
                .output(output)
                .interval(interval)
                .steps(steps)
                .hash(hash)
                .threads(filesThreads)
                .build());
    }
}