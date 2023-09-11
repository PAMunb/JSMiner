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

import java.io.IOException;
import java.nio.file.FileVisitOption;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
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
     * @param steps   How many days should the traverse use to group a set of
     *                commits?
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
            // logger.info("{} -- visiting commit group {} of {} (took {}ms to collect in
            // the last run)",
            // project, currentTraversed, totalGroups, profiler.last());

            profiler.start();

            // Crie a tarefa e adicione-a à lista de tarefas
            Callable<Summary> task = () -> {
                Summary summary = collect(head, current, commits, threads);
                logger.info("{} -- visited commit group {} of {} (took {}ms to collect)",
                        project, currentTraversed, totalGroups, profiler.last());
                return summary;
            };
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
                System.out.println(ex.getStackTrace());
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
    
            val parser = new JSParser();
            val visitor = new JSVisitor();
            val errors = new HashMap<String, String>();
            val metrics = new ArrayList<Metric>();
    
            metrics.add(Metric.builder().name("project").value(project).build());
            metrics.add(Metric.builder().name("date (dd-mm-yyyy)").value(Formatter.format.format(current)).build());
            metrics.add(Metric.builder().name("revision").value(commit).build());
    
            val asyncDeclarations = new AtomicInteger(0);
            val awaitDeclarations = new AtomicInteger(0);
            val constDeclarations = new AtomicInteger(0);
            val classDeclarations = new AtomicInteger(0);
            val arrowFunctionDeclarations = new AtomicInteger(0);
            val letDeclarations = new AtomicInteger(0);
            val exportDeclarations = new AtomicInteger(0);
            val yieldDeclarations = new AtomicInteger(0);
            val importStatements = new AtomicInteger(0);
            val promiseDeclarations = new AtomicInteger(0);
            val promiseAllAndThen = new AtomicInteger(0);
            val defaultParameters = new AtomicInteger(0);
            val restStatements = new AtomicInteger(0);
            val spreadArguments = new AtomicInteger(0);
            val arrayDestructuring = new AtomicInteger(0);
            val objectDestructuring = new AtomicInteger(0);
            val statements = new AtomicInteger(0);
    
            // Lista os arquivos diretamente no diretório, evitando recursão
            Files.list(path)
                .filter(Files::isRegularFile)
                .filter(file -> file.toString().endsWith(".js"))
                .forEach(file -> {
                    try {
                        val content = new String(Files.readAllBytes(file));
                        val program = parser.parse(content);
                        program.accept(visitor);
                        statements.addAndGet(1);
                    } catch (Exception ex) {
                        logger.error("Error processing file: " + file, ex);
                        errors.put(file + "-" + commit, ex.getMessage());
                    }
                });
    
            metrics.add(Metric.builder().name("files").value(statements.get()).build());
            metrics.add(Metric.builder().name("async-declarations").value(asyncDeclarations.get()).build());
            metrics.add(Metric.builder().name("await-declarations").value(awaitDeclarations.get()).build());
            metrics.add(Metric.builder().name("const-declarations").value(constDeclarations.get()).build());
            metrics.add(Metric.builder().name("class-declarations").value(classDeclarations.get()).build());
            metrics.add(Metric.builder().name("arrow-function-declarations").value(arrowFunctionDeclarations.get()).build());
            metrics.add(Metric.builder().name("let-declarations").value(letDeclarations.get()).build());
            metrics.add(Metric.builder().name("export-declarations").value(exportDeclarations.get()).build());
            metrics.add(Metric.builder().name("yield-declarations").value(yieldDeclarations.get()).build());
            metrics.add(Metric.builder().name("import-statements").value(importStatements.get()).build());
            metrics.add(Metric.builder().name("promise-declarations").value(promiseDeclarations.get()).build());
            metrics.add(Metric.builder().name("promise-all-and-then").value(promiseAllAndThen.get()).build());
            metrics.add(Metric.builder().name("default-parameters").value(defaultParameters.get()).build());
            metrics.add(Metric.builder().name("rest-statements").value(restStatements.get()).build());
            metrics.add(Metric.builder().name("spread-arguments").value(spreadArguments.get()).build());
            metrics.add(Metric.builder().name("array-destructuring").value(arrayDestructuring.get()).build());
            metrics.add(Metric.builder().name("object-destructuring").value(objectDestructuring.get()).build());
            metrics.add(Metric.builder().name("errors").value(errors.size()).build());
    
            summary.date(current)
                    .revision(head.toString())
                    .metrics(metrics)
                    .errors(errors);
        } catch (Exception ex) {
            val commit = commits.get(current).toString().split(" ")[1];
            logger.error("Failed to collect data for project: {} on revision: {}", project, commit);
            logger.error("Error while collecting data for project: {} on revision: {}", project, commit, ex);
        }
    
        return summary.build();
    }
}
