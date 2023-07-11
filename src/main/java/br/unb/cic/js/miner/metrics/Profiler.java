package br.unb.cic.js.miner.metrics;

import java.util.ArrayList;
import java.util.List;

/**
 * Profiler abstracts a set of points whose value is the time it took to do some computation. The precision of the
 * profiler is given in milliseconds.
 */
public class Profiler {

    // Set of points for a given runtime
    private final List<Long> points;

    private Long timer;

    public Profiler() {
        points = new ArrayList<>();
    }

    public Long average() {
        if (points.size() == 0) {
            return 0L;
        }

        return points.stream().reduce(Long::sum).orElse(0L)/points.size();
    }

    public Long last() {
        if (points.size() == 0) {
            return 0L;
        }

        return points.get(points.size()-1);
    }

    /**
     * Start an internal timer with millisecond resolution.
     */
    public void start() {
        timer = System.currentTimeMillis();
    }

    /**
     * Stop the timer and add the data point to an internal structure
     */
    public void stop() {
        this.points.add(System.currentTimeMillis()-timer);

        timer = 0L;
    }
}
