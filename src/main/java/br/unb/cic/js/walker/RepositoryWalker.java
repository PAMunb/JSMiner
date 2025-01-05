package br.unb.cic.js.walker;

import br.unb.cic.js.date.Formatter;
import br.unb.cic.js.date.Interval;
import br.unb.cic.js.miner.JSParser;
import br.unb.cic.js.miner.JSVisitor;
import br.unb.cic.js.miner.JSVisitor.Feature;
import br.unb.cic.js.miner.metrics.Metric;
import br.unb.cic.js.miner.metrics.Summary;
import br.unb.cic.js.walker.rules.DirectoriesRule;
import lombok.Builder;
import lombok.val;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.ResetCommand;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.eclipse.jgit.lib.ObjectId;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.revwalk.RevCommit;
import org.eclipse.jgit.storage.file.FileRepositoryBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.SimpleDateFormat;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * This class represents a git project to be analyzed.
 */
@Builder
public final class RepositoryWalker {
    public static final String DATE_FORMAT = "yyyy-MM-dd";
    private final Logger logger = LoggerFactory.getLogger(RepositoryWalker.class);

    public final String project;
    public final Path path;
    public final Boolean merges;

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

        // Use try-with-resources to ensure the repository is closed properly
        try (Repository repository = FileRepositoryBuilder
                .create(path.toAbsolutePath().resolve(".git").toFile())) {
            val head = RepositoryWalkerGit.head(repository);
            Iterable<RevCommit> revisions = RepositoryWalkerGit.revisions(repository, merges);

            val commits = new HashMap<LocalDate, ObjectId>();
            val commitDates = new HashSet<LocalDate>();

            LocalDate previous = null;

            LocalDate beginDate = interval.begin.toInstant()
                    .atZone(ZoneId.systemDefault())
                    .toLocalDate();
            LocalDate endDate = interval.end.toInstant()
                    .atZone(ZoneId.systemDefault())
                    .toLocalDate();

            // Populate the commits map with commits that fit the defined interval
            for (val revision : Objects.requireNonNull(revisions)) {
                val commitTimeInSeconds = revision.getCommitTime();
                val current = LocalDate.ofInstant(Instant.ofEpochSecond(commitTimeInSeconds), ZoneId.systemDefault());

                if (!current.isBefore(beginDate) && !current.isAfter(endDate) && (previous == null || Math.abs(ChronoUnit.DAYS.between(previous, current)) >= (long) steps)) {
                        commitDates.add(current);
                        previous = current;
                        commits.putIfAbsent(current, revision.toObjectId());
                    }

            }

            List<LocalDate> sortedCommitDates = new ArrayList<>(commitDates);
            sortedCommitDates.sort(Comparator.naturalOrder());

            int traversed = 1;
            int totalGroups = sortedCommitDates.size();
            int totalCommits = commits.size();

            logger.info("{} -- total commits: {}", project, totalCommits);
            logger.info("{} -- number of groups: {}", project, totalGroups);

            List<Summary> summaries = Collections.synchronizedList(new ArrayList<>());
            long totalDuration = 0;

            try {
                for (LocalDate current : sortedCommitDates) {
                    // Converter LocalDate para Date
                    Date currentDate = Date.from(current.atStartOfDay(ZoneId.systemDefault()).toInstant());

                    long taskStartTime = System.nanoTime();
                    Summary result = collect(head, currentDate, commits, threads);
                    long taskEndTime = System.nanoTime();
                    long taskDurationMs = (taskEndTime - taskStartTime) / 1_000_000;
                    totalDuration += taskDurationMs;

                    logger.info("{} -- Task for commit {} of {} completed in {} ms",
                            project, traversed, totalGroups, taskDurationMs);

                    summaries.add(result);
                    traversed++;
                }
            } catch (Exception e) {
                logger.error(e.getMessage(), e);
            }

            logger.info("{} -- finished, took {}s in total",
                    project, totalDuration);

            commits.clear();
            commitDates.clear();
            sortedCommitDates.clear();

            return summaries;
        }
    }

    /**
     * Traverse the git project to look for a given hash and then collect metrics
     * about that specific point.
     *
     * @param hash     The hash of a given revision
     * @param threads  How many threads to use when analyzing a revision
     * @return
     * @throws Exception
     */
    public List<Summary> traverse(final String hash, final int threads) throws Exception {
        logger.info("{} -- processing project for a single revision", project);

        List<Summary> summaries = Collections.synchronizedList(new ArrayList<>());

        try (Repository repository = FileRepositoryBuilder.create(path.toAbsolutePath().resolve(".git").toFile())) {

            val head = RepositoryWalkerGit.head(repository);
            val revisions = RepositoryWalkerGit.revisions(repository, merges);

            Set<String> hashSet = new HashSet<>(Arrays.asList(hash));
            val commits = new HashMap<LocalDate, ObjectId>();

            for (val revision : revisions) {
                val id = revision.toObjectId();
                val commit = repository.parseCommit(id).getId().toString().split(" ")[1];

                if (hashSet.contains(commit)) {
                    val commitTimeInSeconds = revision.getCommitTime();
                    val current = LocalDate.ofInstant(Instant.ofEpochSecond(commitTimeInSeconds), ZoneId.systemDefault());

                    commits.put(current, revision.toObjectId());
                    break;
                }
            }

            val current = commits.keySet().stream().findFirst().get();

            ZoneId zoneId = ZoneId.systemDefault();
            Date currentDate = Date.from(current.atStartOfDay(zoneId).toInstant());


            summaries.add(collect(head, currentDate, commits, threads));

            return summaries;
        }
    }

    private Summary collect(ObjectId head, Date current, Map<LocalDate, ObjectId> commits, int threads) {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern(DATE_FORMAT);
        SimpleDateFormat sdf = new SimpleDateFormat(DATE_FORMAT);
        val id = commits.get(LocalDate.parse(sdf.format(current), formatter));
        val summary = Summary.builder();
        val metrics = new ArrayList<Metric<?>>();
        val errors = new HashMap<String, String>();

        try (Repository repository = FileRepositoryBuilder.create(path.toAbsolutePath().resolve(".git").toFile())) {

            revisionAnalyzer(head, current, threads, repository, id, metrics, errors, summary);

        } catch (IOException e) {
            handleException(e, commits, current, errors);
        }
        return summary.build();
    }

    private void revisionAnalyzer(ObjectId head, Date current, int threads, Repository repository, ObjectId id, ArrayList<Metric<?>> metrics, HashMap<String, String> errors, Summary.SummaryBuilder summary) {
        try (Git git = new Git(repository)) {
            String commit = retrieveCommitId(id, repository);
            metrics.addAll(prepareMetrics(current, commit));

            resetAndCheckout(git, id);

            val files = getJavaScriptFiles();
            metrics.add(Metric.builder().name("files").value(files.size()).build());

            val parser = new JSParser();
            val visitor = new JSVisitor();

            processFilesWithThreads(files, threads, errors, parser, visitor, commit);

            metrics.addAll(collectVisitorMetrics(visitor));

            files.clear();

            metrics.add(Metric.builder().name("errors").value(errors.size()).build());
            metrics.add(Metric.builder().name("statements").value(visitor.getTotalStatements().get()).build());

        } catch (Exception ex) {
            logger.error("failed to collect metrics for project {}, cause: {}", project, ex.getMessage());
            Thread.currentThread().interrupt();
        } finally {
            summary.date(current).revision(head.toString()).metrics(metrics).errors(errors);
        }
    }


    private String retrieveCommitId(ObjectId id, Repository repository) throws IOException {
        return repository.parseCommit(id).getId().toString().split(" ")[1];
    }


    private List<Metric<?>> prepareMetrics(Date current, String commit) {
        List<Metric<?>> metrics = new ArrayList<>();
        metrics.add(Metric.builder().name("project").value(project).build());
        metrics.add(Metric.builder().name("date (dd-mm-yyyy)").value(Formatter.format.format(current)).build());
        metrics.add(Metric.builder().name("revision").value(commit).build());
        return metrics;
    }


    private void resetAndCheckout(Git git, ObjectId id) throws GitAPIException {
        git.reset().setMode(ResetCommand.ResetType.HARD).setRef("origin/HEAD").call();
        git.checkout().setName(id.getName()).call();
    }


    private List<Path> getJavaScriptFiles() throws IOException {
        try (Stream<Path> walker = Files.walk(path, FileVisitOption.FOLLOW_LINKS)) {
            return walker.filter(DirectoriesRule::walk)
                    .filter(Files::isRegularFile)
                    .filter(file -> file.toString().endsWith(".js"))
                    .collect(Collectors.toList());
        }
    }


    private void processFilesWithThreads(List<Path> files, int threads, Map<String, String> errors, JSParser parser, JSVisitor visitor, String commit) throws InterruptedException, ExecutionException {

        int adjustedThreads = Math.min(threads, Runtime.getRuntime().availableProcessors());

        val tasks = new ArrayList<Future<?>>(adjustedThreads);
        val pool = Executors.newFixedThreadPool(adjustedThreads);

        try {
            for (Path p : files) {
                Runnable task = () -> processFile(p, parser, visitor, errors, commit);
                tasks.add(pool.submit(task));
            }

            for (val task : tasks) {
                task.get();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            logger.error("Thread interrupted during processing", e);
        } finally {
            pool.shutdown();
            if (!pool.awaitTermination(60, TimeUnit.SECONDS)) {
                pool.shutdownNow();
            }
        }
    }


    private void processFile(Path p, JSParser parser, JSVisitor visitor, Map<String, String> errors, String commit) {
        try {
            val content = new String(Files.readAllBytes(p));
            val program = parser.parse(content);
            visitor.setFile(p.getFileName().toString());
            program.accept(visitor);
        } catch (Exception ex) {
            errors.put(p + "-" + commit, ex.getMessage());
        }
    }

    private List<Metric<?>> collectVisitorMetrics(JSVisitor visitor) {
        List<Metric<?>> metrics = new ArrayList<>();
        metrics.add(Metric.builder().name("async-declarations")
                .value(visitor.getTotalAsyncDeclarations().get())
                .build());
        metrics.add(Metric.builder().name("await-declarations")
                .value(visitor.getTotalAwaitDeclarations().get())
                .build());
        metrics.add(Metric.builder().name("const-declarations")
                .value(visitor.getTotalConstDeclaration().get())
                .build());
        metrics.add(Metric.builder().name("class-declarations")
                .value(visitor.getTotalClassDeclarations().get())
                .build());
        metrics.add(Metric.builder().name("arrow-function-declarations")
                .value(visitor.getTotalArrowDeclarations().get()).build());
        metrics.add(
                Metric.builder().name("let-declarations")
                        .value(visitor.getTotalLetDeclarations().get()).build());
        metrics.add(Metric.builder().name("export-declarations")
                .value(visitor.getTotalExportDeclarations().get())
                .build());
        metrics.add(Metric.builder().name("yield-declarations")
                .value(visitor.getTotalYieldDeclarations().get())
                .build());
        metrics.add(
                Metric.builder().name("import-statements")
                        .value(visitor.getTotalImportStatements().get()).build());
        metrics.add(Metric.builder().name("default-parameters")
                .value(visitor.getTotalDefaultParameters().get())
                .build());
        metrics.add(Metric.builder().name("rest-statements")
                .value(visitor.getTotalRestStatements().get()).build());
        metrics.add(
                Metric.builder().name("spread-arguments")
                        .value(visitor.getTotalSpreadArguments().get()).build());
        metrics.add(Metric.builder().name("array-destructuring")
                .value(visitor.getTotalArrayDestructuring().get())
                .build());
        metrics.add(Metric.builder().name("object-destructuring")
                .value(visitor.getTotalObjectDestructuring().get())
                .build());

        metrics.add(Metric.builder().name("optional-chain").value(visitor.getTotalOptionalChain().get())
                .build());
        metrics.add(Metric.builder().name("template-string-expressions")
                .value(visitor.getTotalTemplateStringExpressions().get()).build());
        metrics.add(Metric.builder().name("null-coalesce-operators")
                .value(visitor.getTotalNullCoalesceOperators().get()).build());
        metrics.add(Metric.builder().name("exponentiation-assignments")
                .value(visitor.getTotalExponentiationAssignments().get()).build());
        metrics.add(Metric.builder().name("private-fields").value(visitor.getTotalPrivateFields().get())
                .build());
        metrics.add(Metric.builder().name("numeric-separator")
                .value(visitor.getTotalNumericLiteralSeparators().get()).build());
        metrics.add(Metric.builder().name("big-int").value(visitor.getTotalBigInt().get()).build());


        metrics.add(Metric.builder().name("enhanced-property-assignments")
                .value(visitor.getTotalEnhancedPropertyAssignments().get()).build());
        metrics.add(Metric.builder().name("computed-property-assignments")
                .value(visitor.getTotalComputedPropertyAssignments().get()).build());
        metrics.add(Metric.builder().name("function-property-declaration")
                .value(visitor.getTotalFunctionPropertyDeclarations().get()).build());

        metrics.add(Metric.builder().name("for-of-statements")
                .value(visitor.getTotalForOfStatements().get()).build());
        metrics.add(Metric.builder().name("for-await-of").value(visitor.getTotalForAwaitOf().get())
                .build());
        metrics.add(Metric.builder().name("static-block-in-classes")
                .value(visitor.getTotalStaticBlockInClasses().get()).build());
        metrics.add(Metric.builder().name("optional-catch-binding-declarations")
                .value(visitor.getTotalOptionalCatchBindingDeclarations().get()).build());
        metrics.add(Metric.builder().name("private_methods")
                .value(visitor.getTotalPrivateMethods().get()).build());
        metrics.add(Metric.builder().name("assignment-operators")
                .value(visitor.getTotalAssignmentOperators().get()).build());
        metrics.add(Metric.builder().name("spread-in-objects")
                .value(visitor.getTotalSpreadInObjects().get()).build());
        metrics.add(Metric.builder().name("rest-in-objects")
                .value(visitor.getTotalRestInObjects().get()).build());

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
        metrics.add(Metric.builder().name("optional-chain-files")
                .value(visitor.occurrences(Feature.OptionalChain))
                .build());
        metrics.add(Metric.builder().name("template-string-expressions-files")
                .value(visitor.occurrences(Feature.TemplateStringExpressions)).build());
        metrics.add(Metric.builder().name("null-coalesce-operators-files")
                .value(visitor.occurrences(Feature.NullCoalesceOperators)).build());
        metrics.add(Metric.builder().name("exponentiation-assignments-files")
                .value(visitor.occurrences(Feature.ExponentiationAssignments)).build());
        metrics.add(Metric.builder().name("private-fields-files")
                .value(visitor.occurrences(Feature.PrivateFields))
                .build());
        metrics.add(Metric.builder().name("numeric-separator-files")
                .value(visitor.occurrences(Feature.NumericLiteralSeparators)).build());
        metrics.add(Metric.builder().name("big-int-files").value(visitor.occurrences(Feature.BigInt))
                .build());

        metrics.add(Metric.builder().name("enhanced-property-assignments-files")
                .value(visitor.occurrences(Feature.EnhancedPropertyAssignments)).build());
        metrics.add(Metric.builder().name("computed-property-assignments-files")
                .value(visitor.occurrences(Feature.ComputedPropertyAssignments)).build());
        metrics.add(Metric.builder().name("function-property-declaration-files")
                .value(visitor.occurrences(Feature.FunctionPropertyDeclarations)).build());

        metrics.add(Metric.builder().name("for-of-statements-files")
                .value(visitor.occurrences(Feature.ForOfStatements)).build());
        metrics.add(Metric.builder().name("for-await-of-files")
                .value(visitor.occurrences(Feature.ForAwaitOf))
                .build());
        metrics.add(Metric.builder().name("static-block-in-classes-files")
                .value(visitor.occurrences(Feature.StaticBlockInClasses)).build());

        metrics.add(Metric.builder().name("optional-catch-binding-declarations-files")
                .value(visitor.occurrences(Feature.OptionalCatchBindingDeclarations)).build());
        metrics.add(Metric.builder().name("private_methods-files")
                .value(visitor.occurrences(Feature.PrivateMethods)).build());
        metrics.add(Metric.builder().name("assignment-operators-files")
                .value(visitor.occurrences(Feature.AssignmentOperators)).build());
        metrics.add(Metric.builder().name("spread-in-objects-files")
                .value(visitor.occurrences(Feature.SpreadInObjects)).build());
        metrics.add(Metric.builder().name("rest-in-objects-files")
                .value(visitor.occurrences(Feature.RestInObjects)).build());

        return metrics;
    }

    private void handleException(Exception ex, Map<LocalDate, ObjectId> commits, Date current, Map<String, String> errors) {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern(DATE_FORMAT);
        SimpleDateFormat sdf = new SimpleDateFormat(DATE_FORMAT);
        val commit = commits.get(LocalDate.parse(sdf.format(current), formatter)).toString().split(" ")[1];
        logger.error("failed to collect data for project {} on revision: {}", project, commit);
        errors.put("exception", ex.getMessage());
    }
}
