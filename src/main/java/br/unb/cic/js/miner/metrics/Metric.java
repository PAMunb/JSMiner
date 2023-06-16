package br.unb.cic.js.miner.metrics;

import lombok.Builder;

/**
 * Metric abstracts the collected metric name and value to be exported later.
 */
@Builder
public class Metric<T> {
    public final String name;

    public T value;
}