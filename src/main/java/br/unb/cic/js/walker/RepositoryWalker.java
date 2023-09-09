package br.unb.cic.js.walker;

import br.unb.cic.js.date.Formatter;
import br.unb.cic.js.date.Interval;
import br.unb.cic.js.miner.JSParser;
import br.unb.cic.js.miner.JSVisitor;
import br.unb.cic.js.miner.metrics.Metric;
import br.unb.cic.js.miner.metrics.Profiler;
import br.unb.cic.js.miner.metrics.Summary;
import br.unb.cic.js.walker.rules.DirectoriesRule;
import lombok.Builder;
import lombok.val;

import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.ListBranchCommand;
import org.eclipse.jgit.api.ResetCommand;
import org.eclipse.jgit.lib.ObjectId;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.revwalk.filter.CommitTimeRevFilter;
import org.eclipse.jgit.revwalk.filter.RevFilter;
import org.eclipse.jgit.storage.file.FileRepositoryBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * This class represents a git project to be analyzed.
 */
@Builder
public class RepositoryWalker {
    private final Logger logger = LoggerFactory.getLogger(RepositoryWalker.class);

    public final String project;
    public final Path path;

    private final List<Summary> summaries = new ArrayList<>();

    private Repository repository;

    /**
     * Traverse the git project from an initial date to an end date.
     *
     * @param initial The initial date of the traversal
     * @param end     The end date of the traversal
     * @param steps   How many days should the traverse use to group a set of commits?
     * @param threads How many threads to use when analyzing a revision
     * @throws Exception
     */
    public List<Summary> traverse(Date initial, Date end, int steps, int threads) throws Exception {
        logger.info("{} -- processing project", project);

        repository = FileRepositoryBuilder.create(path.toAbsolutePath().resolve(".git").toFile());

        val git = new Git(repository);

        val branches = git.branchList()
                .setListMode(ListBranchCommand.ListMode.REMOTE)
                .call()
                .stream()
                .filter(n -> n.getName().equals("refs/remotes/origin/HEAD"))
                .findFirst();

        var mainBranch = "";

        if (branches.isPresent()) {
            mainBranch = branches.get().getTarget().getName().substring("refs/remotes/origin/".length());
        } else {
            logger.error("{} -- failed to get the project main branch", project);
            git.close();
            return summaries;
        }

        git.reset().setMode(ResetCommand.ResetType.HARD).call();
        git.checkout().setName(mainBranch).call();

        val head = repository.resolve("refs/heads/" + mainBranch);

        val revisions = git.log()
                .add(head)
                .setRevFilter(CommitTimeRevFilter.between(initial, end))
                .setRevFilter(RevFilter.NO_MERGES)
                .call();

        val commits = new HashMap<Date, ObjectId>();
        val commitDates = new ArrayList<Date>();

        Date previous = null;

        // fill the commits map with commits that will be analyzed given that they
        // belong to the defined interval
        for (val revision : revisions) {
            val author = revision.getAuthorIdent();
            val current = author.getWhen();

            if (current.compareTo(initial) >= 0 && current.compareTo(end) <= 0) {
                // only add commits that fit the interval
                if (previous == null || Interval.diff(current, previous, Interval.Unit.Days) >= steps) {
                    commitDates.add(current);

                    previous = current;
                }

                commits.put(current, revision.toObjectId());
            }
        }

        Collections.sort(commitDates);

        var traversed = 0;

        val totalGroups = commitDates.size();
        val totalCommits = commits.size();

        logger.info("{} -- number of commits {} ", project, totalCommits);
        logger.info("{} -- number of groups {} ", project, totalGroups);

        val profiler = new Profiler();

        // Use ExecutorService para gerenciar as threads fora do loop
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        List<Future<Summary>> taskResults = new ArrayList<>();

        for (Date current : commitDates) {
            final int currentTraversed = traversed;
            traversed++;
            logger.info("{} -- visiting commit group {} of {} (took {}ms to collect in the last run)",
                    project, currentTraversed, totalGroups, profiler.last());

            profiler.start();

            // Crie a tarefa e adicione-a à lista de tarefas
            Callable<Summary> task = () -> collect(head, current, commits, threads);
            taskResults.add(executor.submit(task));

            profiler.stop();
        }

        // Espere todas as tarefas serem concluídas
        for (Future<Summary> taskResult : taskResults) {
            try {
                val summary = taskResult.get();
                summaries.add(summary);
            } catch (Exception ex) {
                logger.error("Error while collecting data for project: {}", project, ex);
            }
        }

        executor.shutdown();
        executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);

        logger.info("{} -- finished, took {}ms in average to collect each commit group", project, profiler.average());

        git.close();

        return summaries;
    }

    /**
     * Collect metrics about a given commit interval
     */
    private Summary collect(ObjectId head, Date current, Map<Date, ObjectId> commits, int threads) {
        val id = commits.get(current);
        val summary = Summary.builder();

        try (Git git = new Git(repository)) {
            val commit = repository.parseCommit(id).getId().toString().split(" ")[1];

            git.reset().setMode(ResetCommand.ResetType.HARD).call();
            git.checkout().setName(id.getName()).call();

            val walker = Files.walk(path, FileVisitOption.FOLLOW_LINKS);
            val files = walker.collect(Collectors.toList())
                    .stream()
                    .filter(DirectoriesRule::walk)
                    .filter(filepath -> {
                        val file = filepath.toFile();

                        val isDirectory = file.isDirectory();
                        val isJsFile = file.toString().endsWith(".js");

                        return !isDirectory && isJsFile;
                    }).collect(Collectors.toList());

            walker.close();

            val parser = new JSParser();
            val visitor = new JSVisitor();

            var errors = new HashMap<String, String>();

            val metrics = new ArrayList<Metric>();

            metrics.add(Metric.builder().name("project").value(project).build());
            metrics.add(Metric.builder().name("date (dd-mm-yyyy)").value(Formatter.format.format(current)).build());
            metrics.add(Metric.builder().name("revision").value(commit).build());
            metrics.add(Metric.builder().name("files").value(files.size()).build());

            val tasks = new ArrayList<Future>(threads);
            val pool = Executors.newFixedThreadPool(threads);

            for (Path p : files) {
                Runnable task = () -> {
                    try {
                        val content = new String(Files.readAllBytes(p));
                        val program = parser.parse(content);

                        program.accept(visitor);

                    } catch (Exception ex) {
                        errors.put(p + "-" + commit, ex.getMessage());
                    }
                };

                tasks.add(pool.submit(task));
            }

            for (val task : tasks) {
                task.get();
            }

            pool.shutdown();

            metrics.add(Metric.builder().name("async-declarations").value(visitor.getTotalAsyncDeclarations().get()).build());
            metrics.add(Metric.builder().name("await-declarations").value(visitor.getTotalAwaitDeclarations().get()).build());
            metrics.add(Metric.builder().name("const-declarations").value(visitor.getTotalConstDeclaration().get()).build());
            metrics.add(Metric.builder().name("class-declarations").value(visitor.getTotalClassDeclarations().get()).build());
            metrics.add(Metric.builder().name("arrow-function-declarations").value(visitor.getTotalArrowDeclarations().get()).build());
            metrics.add(Metric.builder().name("let-declarations").value(visitor.getTotalLetDeclarations().get()).build());
            metrics.add(Metric.builder().name("export-declarations").value(visitor.getTotalExportDeclarations().get()).build());
            metrics.add(Metric.builder().name("yield-declarations").value(visitor.getTotalYieldDeclarations().get()).build());
            metrics.add(Metric.builder().name("import-statements").value(visitor.getTotalImportStatements().get()).build());
            metrics.add(Metric.builder().name("promise-declarations").value(visitor.getTotalNewPromises().get()).build());
            metrics.add(Metric.builder().name("promise-all-and-then").value(visitor.getTotalPromiseAllAndThenIdiom().get()).build());
            metrics.add(Metric.builder().name("default-parameters").value(visitor.getTotalDefaultParameters().get()).build());
            metrics.add(Metric.builder().name("rest-statements").value(visitor.getTotalRestStatements().get()).build());
            metrics.add(Metric.builder().name("spread-arguments").value(visitor.getTotalSpreadArguments().get()).build());
            metrics.add(Metric.builder().name("array-destructuring").value(visitor.getTotalArrayDestructuring().get()).build());
            metrics.add(Metric.builder().name("object-destructuring").value(visitor.getTotalObjectDestructuring().get()).build());
            metrics.add(Metric.builder().name("errors").value(errors.size()).build());
            metrics.add(Metric.builder().name("statements").value(visitor.getTotalStatements().get()).build());

            summary.date(current)
                    .revision(head.toString())
                    .metrics(metrics)
                    .errors(errors);
        } catch (Exception ex) {
            val commit = commits.get(current).toString().split(" ")[1];

            logger.error("failed to collect data for project: {} on revision: {}", project, commit);
            System.out.println(ex.getStackTrace());
        }

        return summary.build();
    }
}
