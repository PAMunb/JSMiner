package br.unb.cic.js.walker;

import br.unb.cic.js.date.Formatter;
import br.unb.cic.js.date.Interval;
import br.unb.cic.js.miner.JSParser;
import br.unb.cic.js.miner.JSVisitor;
import br.unb.cic.js.miner.metrics.Metric;
import br.unb.cic.js.miner.metrics.Profiler;
import br.unb.cic.js.miner.metrics.Summary;
import lombok.Builder;
import lombok.val;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.ListBranchCommand;
import org.eclipse.jgit.api.ResetCommand;
import org.eclipse.jgit.lib.ObjectId;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.revwalk.RevCommit;
import org.eclipse.jgit.storage.file.FileRepositoryBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

/**
 * This class represents a git project to be analyzed.
 */
@Builder
public class RepositoryWalker {

    private final Logger logger = LoggerFactory.getLogger(RepositoryWalker.class);

    public final String project;
    public final Path path;

    private Repository repository;

    private final List<Summary> summaries = new ArrayList<>();

    /**
     * Traverse the git project from an initial date to an end date.
     *
     * @param initial The initial date of the traversal
     * @param end The end date of the traversal
     * @param steps How many days should the traverse use to group a set of commits?
     * @throws Exception
     */
    public List<Summary> traverse(Date initial, Date end, int steps) throws Exception {
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
            return summaries;
        }

        git.reset().setMode(ResetCommand.ResetType.HARD).call();
        git.checkout().setName(mainBranch).call();

        val head = repository.resolve("refs/heads/" + mainBranch);

        val commits = new HashMap<Date, ObjectId>();
        val commitDates = new ArrayList<Date>();

        Date previous = null;

        // fill the commits map with commits that will be analyzed given that they
        // belong to the defined interval
        for (RevCommit commit : git.log().add(head).call()) {
            val author = commit.getAuthorIdent();
            val current = author.getWhen();

            if (current.compareTo(initial) >= 0 && current.compareTo(end) <= 0) {
                // only add commits that fit the interval
                if (previous == null || Interval.diff(current, previous, Interval.Unit.Days) >= steps) {
                    commitDates.add(current);

                    previous = current;
                }

                commits.put(current, commit.toObjectId());
            }
        }

        Collections.sort(commitDates);

        var traversed = 0;

        val totalGroups = commitDates.size();
        val totalCommits = commits.size();

        logger.info("{} -- number of commits {} ", project, totalCommits);
        logger.info("{} -- number of groups {} ", project, totalGroups);

        val profiler = new Profiler();

        for (Date current : commitDates) {
            traversed++;
            logger.info("{} -- visiting commit group {} of {} (took {}ms to collect in the last run)", project, traversed, totalGroups, profiler.last());

            profiler.start();

            collect(head, current, commits);

            profiler.stop();
        }

        logger.info("{} -- finished, took {}ms in average to collect each commit group", project, profiler.average());

        git.close();

        return summaries;
    }

    /**
     * Collect metrics about a given commit interval
     */
    private void collect(ObjectId head, Date current, Map<Date, ObjectId> commits) {
        val id = commits.get(current);

        try (Git git = new Git(repository)) {
            val commit = repository.parseCommit(id).getId().toString().split(" ")[1];

            git.reset().setMode(ResetCommand.ResetType.HARD).call();
            git.checkout().setName(id.getName()).call();

            val walker = Files.walk(path, FileVisitOption.FOLLOW_LINKS);
            val files = walker.collect(Collectors.toList())
                    .stream()
                    .filter(filepath -> {
                        val file = filepath.toFile();

                        val isDirectory = file.isDirectory();
                        val isJsFile = file.toString().endsWith(".js");

                        return !isDirectory && isJsFile;
                    }).collect(Collectors.toList());

            val parser = new JSParser();
            val visitor = new JSVisitor();

            var errors = new HashMap<String, String>();

            val metrics = new ArrayList<Metric>();

            metrics.add(Metric.builder().name("project").value(project).build());
            metrics.add(Metric.builder().name("date (dd-mm-yyyy)").value(Formatter.format.format(current)).build());
            metrics.add(Metric.builder().name("revision").value(commit).build());
            metrics.add(Metric.builder().name("files").value(files.size()).build());

            val cores = Runtime.getRuntime().availableProcessors();
            val tasks = new ArrayList<Future>();
            val pool = Executors.newFixedThreadPool(cores);

            for (Path p : files) {
                Runnable task = () -> {
                    try {
                        val content = new String(Files.readAllBytes(p));
                        val program = parser.parse(content);

                        program.accept(visitor);

                    } catch (Exception ex) {
                        errors.put(p+"-"+commit, ex.getMessage());
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

            System.gc();

            val summary = Summary.builder()
                    .date(current)
                    .revision(head.toString())
                    .metrics(metrics)
                    .errors(errors)
                    .build();

            synchronized (summaries) {
                summaries.add(summary);
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
