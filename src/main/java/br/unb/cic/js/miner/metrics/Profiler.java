package br.unb.cic.js.miner.metrics;

import lombok.val;

import java.util.ArrayList;
import java.util.List;

/**
 * Profiler abstracts a set of points whose value is the time it took to do some computation.
 */
public class Profiler {

    // Set of points for a given runtime
    public final List<Long> points;

    public Profiler() {
        points = new ArrayList<Long>();
    }

    public Double average() {
        if (points.size() == 0) {
            return 0D;
        }

        val avg = points.stream().reduce(Long::sum).orElse(0L).doubleValue()/points.size();

        return avg;
    }
}
