package br.unb.cic.js;

import com.beust.jcommander.Parameter;

import java.time.Instant;
import java.time.Year;

public class Args {
    @Parameter(names = {"-d", "--directory"}, required = true, description = "The path to a/set of git repository(ies) containing JS code")
    public String directory = "";

    @Parameter(names = {"-p", "--project"}, description = "Given the project flag it will only mine for that project")
    public String project = "";

    @Parameter(names = {"-s", "--steps"}, description = "The delta interval through each commit")
    public Integer steps = 7;

    @Parameter(names = {"-t", "--threads"}, description = "How many threads to use when processing multiple projects")
    public Integer threads = 1;

    @Parameter(names = {"-id", "--initial-date"}, description = "When to start walking a project (dd-mm-yyyy)")
    public String initialDate = "01-01-2010";

    @Parameter(names = {"-ed", "--end-date"}, description = "When to stop walking the project (dd-mm-yyyy)")
    public String endDate = "30-06-2023";

    @Parameter(names = {"-h", "--help"}, help = true, description = "Show help")
    private boolean help;
}
