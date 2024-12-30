package br.unb.cic.js.walker;

import br.unb.cic.js.date.Formatter;
import br.unb.cic.js.date.Interval;
import br.unb.cic.js.miner.JSParser;
import br.unb.cic.js.miner.JSVisitor;
import br.unb.cic.js.miner.JSVisitor.Feature;
import br.unb.cic.js.miner.metrics.Metric;
import br.unb.cic.js.miner.metrics.Profiler;
import br.unb.cic.js.miner.metrics.Summary;
import br.unb.cic.js.walker.rules.DirectoriesRule;
import lombok.Builder;
import lombok.val;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.ResetCommand;
import org.eclipse.jgit.lib.ObjectId;
import org.eclipse.jgit.lib.Repository;
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
public final class RepositoryWalker {
    private final Logger logger = LoggerFactory.getLogger(RepositoryWalker.class);

    public final String project;
    public final Path path;
    public final Boolean merges;

    private final List<Summary> summaries = Collections.synchronizedList(new ArrayList<>());

    private Repository repository;

    /**
     * Traverse the git project from an initial date to an end date.
     *
     * @param interval The delta date of the traversal
     * @param steps    How many days should the traverse use to group a set of
     *                 commits?
     * @param threads  How many threads to use when analyzing a revision
     * @throws Exception
     */
    public List<Summary> traverse(final Interval interval, final int steps, final int threads) throws Exception {
        logger.info("{} -- processing project", project);

        repository = FileRepositoryBuilder.create(path.toAbsolutePath().resolve(".git").toFile());

        val head = RepositoryWalkerGit.head(repository);
        val revisions = RepositoryWalkerGit.revisions(repository, merges);

        val commits = new HashMap<Date, ObjectId>();
        val commitDates = new HashSet<Date>(); // Use a HashSet for unique dates

        Date previous = null;

        // fill the commits map with commits that will be analyzed given that they
        // belong to the defined interval
        for (val revision : revisions) {
            // val author = revision.getAuthorIdent();
            val commitTimeInSeconds = revision.getCommitTime();
            val current = new Date((long) commitTimeInSeconds * 1000);
            if (current.compareTo(interval.begin) >= 0 && current.compareTo(interval.end) <= 0) {
                // only add commits that fit the interval

                if (previous == null || Interval.diff(current, previous, Interval.Unit.Days) >= steps) {
                    commitDates.add(current);
                    previous = current;

                    // add just the date are not added
                    if (!commits.containsKey(current)) {
                        commits.put(current, revision.toObjectId());
                    }
                }
            }
        }

        List<Date> sortedCommitDates = new ArrayList<>(commitDates);
        Collections.sort(sortedCommitDates);

        var traversed = 0;

        val totalGroups = sortedCommitDates.size();
        val totalCommits = commits.size();

        logger.info("{} -- total of commits {} ", project, totalCommits);
        logger.info("{} -- number of commit {} ", project, totalGroups);

        val profiler = new Profiler();

        for (Date current : sortedCommitDates) {
            traversed++;

            profiler.start();

            val summary = collect(head, current, commits, threads);

            profiler.stop();

            logger.info("{} -- collected commit {} of {} (took {}ms to collect)", project, traversed, totalGroups,
                    profiler.last());

            summaries.add(summary);
        }

        val average = profiler.average();
        val total = (double) profiler.total() / 1000.0;

        logger.info("{} -- finished, took {}ms in average to collect each commit and {}s in total", project, average,
                total);

        return summaries;
    }

    /**
     * Traverse the git project to look for a given hash and then collect metrics
     * about that specific point.
     *
     * @param interval The delta date of the traversal
     * @param hash     The hash of a given revision
     * @param threads  How many threads to use when analyzing a revision
     * @return
     * @throws Exception
     */
    public List<Summary> traverse(final Interval interval, final String hash, final int threads) throws Exception {
        logger.info("{} -- processing project for a single revision", project);

        repository = FileRepositoryBuilder.create(path.toAbsolutePath().resolve(".git").toFile());

        val head = RepositoryWalkerGit.head(repository);
        val revisions = RepositoryWalkerGit.revisions(repository, merges);

        val commits = new HashMap<Date, ObjectId>();

        for (val revision : revisions) {
            val id = revision.toObjectId();
            val commit = repository.parseCommit(id).getId().toString().split(" ")[1];

            if (Arrays.asList(hash).contains(commit)) {
                val commitTimeInSeconds = revision.getCommitTime();
                val current = new Date((long) commitTimeInSeconds * 1000);

                commits.put(current, revision.toObjectId());
                break;
            }
        }

        val current = commits.keySet().stream().findFirst().get();

        // collect only one summary
        summaries.add(collect(head, current, commits, threads));

        return summaries;
    }

    /**
     * Collect metrics about a given commit interval
     */
    private Summary collect(ObjectId head, Date current, Map<Date, ObjectId> commits, int threads) {
        val id = commits.get(current);
        val summary = Summary.builder();

        val metrics = new ArrayList<Metric<?>>();

        metrics.add(Metric.builder().name("project").value(project).build());
        metrics.add(Metric.builder().name("date (dd-mm-yyyy)").value(Formatter.format.format(current)).build());

        var errors = new HashMap<String, String>();

        try (Git git = new Git(repository)) {
            val commit = repository.parseCommit(id).getId().toString().split(" ")[1];

            metrics.add(Metric.builder().name("revision").value(commit).build());

            git.reset().setMode(ResetCommand.ResetType.HARD).setRef("origin/HEAD").call();
            git.checkout().setName(id.getName()).call();

            val walker = Files.walk(path, FileVisitOption.FOLLOW_LINKS);
            val files = walker.collect(Collectors.toList()).stream().filter(DirectoriesRule::walk)
                    .filter(Files::isRegularFile).filter(file -> file.toString().endsWith(".js"))
                    .collect(Collectors.toList());

            walker.close();

            metrics.add(Metric.builder().name("files").value(files.size()).build());

            val parser = new JSParser();
            val visitor = new JSVisitor();

            val tasks = new ArrayList<Future<?>>(threads);
            val pool = Executors.newFixedThreadPool(threads);

            for (Path p : files) {
                Runnable task = () -> {
                    try {
                        val content = new String(Files.readAllBytes(p));
                        val program = parser.parse(content);
                        visitor.setFile(p.getFileName().toString());
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

            metrics.add(Metric.builder().name("async-declarations").value(visitor.getTotalAsyncDeclarations().get())
                    .build());
            metrics.add(Metric.builder().name("await-declarations").value(visitor.getTotalAwaitDeclarations().get())
                    .build());
            metrics.add(Metric.builder().name("const-declarations").value(visitor.getTotalConstDeclaration().get())
                    .build());
            metrics.add(Metric.builder().name("class-declarations").value(visitor.getTotalClassDeclarations().get())
                    .build());
            metrics.add(Metric.builder().name("arrow-function-declarations")
                    .value(visitor.getTotalArrowDeclarations().get()).build());
            metrics.add(
                    Metric.builder().name("let-declarations").value(visitor.getTotalLetDeclarations().get()).build());
            metrics.add(Metric.builder().name("export-declarations").value(visitor.getTotalExportDeclarations().get())
                    .build());
            metrics.add(Metric.builder().name("yield-declarations").value(visitor.getTotalYieldDeclarations().get())
                    .build());
            metrics.add(
                    Metric.builder().name("import-statements").value(visitor.getTotalImportStatements().get()).build());
            metrics.add(Metric.builder().name("default-parameters").value(visitor.getTotalDefaultParameters().get())
                    .build());
            metrics.add(Metric.builder().name("rest-statements").value(visitor.getTotalRestStatements().get()).build());
            metrics.add(
                    Metric.builder().name("spread-arguments").value(visitor.getTotalSpreadArguments().get()).build());
            metrics.add(Metric.builder().name("array-destructuring").value(visitor.getTotalArrayDestructuring().get())
                    .build());
            metrics.add(Metric.builder().name("object-destructuring").value(visitor.getTotalObjectDestructuring().get())
                    .build());

            metrics.add(Metric.builder().name("optional-chain").value(visitor.getTotalOptionalChain().get()).build());
            metrics.add(Metric.builder().name("template-string-expressions")
                    .value(visitor.getTotalTemplateStringExpressions().get()).build());
            metrics.add(Metric.builder().name("null-coalesce-operators")
                    .value(visitor.getTotalNullCoalesceOperators().get()).build());
            metrics.add(Metric.builder().name("exponentiation-assignments")
                    .value(visitor.getTotalExponentiationAssignments().get()).build());
            metrics.add(Metric.builder().name("private-fields").value(visitor.getTotalPrivateFields().get()).build());
            metrics.add(Metric.builder().name("numeric-separator")
                    .value(visitor.getTotalNumericLiteralSeparators().get()).build());
            metrics.add(Metric.builder().name("big-int").value(visitor.getTotalBigInt().get()).build());

            /* TODO: @Walter, review this code, please */
            metrics.add(Metric.builder().name("enhanced-property-assignments").value(visitor.getTotalEnhancedPropertyAssignments().get()).build());
            metrics.add(Metric.builder().name("computed-property-assignments").value(visitor.getTotalComputedPropertyAssignments().get()).build());
            metrics.add(Metric.builder().name("function-property-declaration").value(visitor.getTotalFunctionPropertyDeclarations().get()).build());

            metrics.add(Metric.builder().name("async-declarations-files")
                    .value(visitor.occurrences(Feature.AsyncDeclarations)).build());
            metrics.add(Metric.builder().name("await-declarations-files")
                    .value(visitor.occurrences(Feature.AwaitDeclarations)).build());
            metrics.add(Metric.builder().name("const-declarations-files")
                    .value(visitor.occurrences(Feature.ConstDeclaration)).build());
            metrics.add(Metric.builder().name("class-declarations-files")
                    .value(visitor.occurrences(Feature.ClassDeclarations)).build());
            metrics.add(Metric.builder().name("arrow-function-declarations-files")
                    .value(visitor.occurrences(Feature.ArrowArrowDeclarations)).build());
            metrics.add(Metric.builder().name("let-declarations-files")
                    .value(visitor.occurrences(Feature.LetDeclarations)).build());
            metrics.add(Metric.builder().name("export-declarations-files")
                    .value(visitor.occurrences(Feature.ExportDeclarations)).build());
            metrics.add(Metric.builder().name("yield-declarations-files")
                    .value(visitor.occurrences(Feature.YieldDeclarations)).build());
            metrics.add(Metric.builder().name("import-statements-files")
                    .value(visitor.occurrences(Feature.ImportStatements)).build());
            metrics.add(Metric.builder().name("default-parameters-files")
                    .value(visitor.occurrences(Feature.DefaultParameters)).build());
            metrics.add(Metric.builder().name("rest-statements-files")
                    .value(visitor.occurrences(Feature.RestStatements)).build());
            metrics.add(Metric.builder().name("spread-arguments-files")
                    .value(visitor.occurrences(Feature.SpreadArguments)).build());
            metrics.add(Metric.builder().name("array-destructuring-files")
                    .value(visitor.occurrences(Feature.ArrayDestructuring)).build());
            metrics.add(Metric.builder().name("object-destructuring-files")
                    .value(visitor.occurrences(Feature.ObjectDestructuring)).build());
            metrics.add(Metric.builder().name("optional-chain-files").value(visitor.occurrences(Feature.OptionalChain))
                    .build());
            metrics.add(Metric.builder().name("template-string-expressions-files")
                    .value(visitor.occurrences(Feature.TemplateStringExpressions)).build());
            metrics.add(Metric.builder().name("null-coalesce-operators-files")
                    .value(visitor.occurrences(Feature.NullCoalesceOperators)).build());
            metrics.add(Metric.builder().name("exponentiation-assignments-files")
                    .value(visitor.occurrences(Feature.ExponentiationAssignments)).build());
            metrics.add(Metric.builder().name("private-fields-files").value(visitor.occurrences(Feature.PrivateFields))
                    .build());
            metrics.add(Metric.builder().name("numeric-separator-files").value(visitor.occurrences(Feature.NumericLiteralSeparators)).build());
            metrics.add(Metric.builder().name("big-int-files").value(visitor.occurrences(Feature.BigInt)).build());
            /* TODO: @Walter, review this code, please */
            metrics.add(Metric.builder().name("enhanced-property-assignments-files").value(visitor.occurrences(Feature.EnhancedPropertyAssignments)).build());
            metrics.add(Metric.builder().name("computed-property-assignments-files").value(visitor.occurrences(Feature.ComputedPropertyAssignments)).build());
            metrics.add(Metric.builder().name("function-property-declaration-files").value(visitor.occurrences(Feature.FunctionPropertyDeclarations)).build());

            metrics.add(Metric.builder().name("errors").value(errors.size()).build());
            metrics.add(Metric.builder().name("statements").value(visitor.getTotalStatements().get()).build());

            metrics.add(Metric.builder().name("forof_statements_files").value(visitor.getTotalForOfStatements()).build());
            metrics.add(Metric.builder().name("forin_statements_files").value(visitor.getTotalForInStatements()).build());
            
            metrics.add(Metric.builder().name("forawaitof_files").value(visitor.getTotalForAwaitOf()).build());

            summary.date(current).revision(head.toString()).metrics(metrics).errors(errors);
        } catch (Exception ex) {
            val commit = commits.get(current).toString().split(" ")[1];

            logger.error("failed to collect data for project {} on revision: {}", project, commit);
            ex.printStackTrace();

            errors.put("exception", ex.getMessage());
        } finally {
            summary.date(current).revision(head.toString()).metrics(metrics).errors(errors);
        }

        return summary.build();
    }
}
