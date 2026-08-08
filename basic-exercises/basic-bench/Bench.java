import java.lang.management.ManagementFactory;
import java.lang.management.RuntimeMXBean;
import java.text.DecimalFormat;
import java.time.Instant;
import java.util.Locale;
import java.util.Random;

/**
 * Bang-for-buck Java benchmark:
 * - Prints host + JVM stats up front
 * - Runs a dense matrix multiply (DGEMM-like) with warmup
 * - Reports GFLOPS and checksum for result validation
 *
 * Defaults are conservative so it runs on Raspberry Pi and Apple Silicon.
 *
 * Usage:
 *   javac Bench.java
 *   java Bench [size] [targetSeconds] [warmup] [threads]
 *
 * Example:
 *   java Bench 512 12 2 4
 */
public class Bench {
    private static final DecimalFormat DF2 = new DecimalFormat("0.00");
    private static final DecimalFormat DF3 = new DecimalFormat("0.000");

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);

        int n = parseArg(args, 0, 512);
        int targetSeconds = parseArg(args, 1, 12);
        int warmup = parseArg(args, 2, 2);
        int threads = parseArg(args, 3, Runtime.getRuntime().availableProcessors());

        if (n < 64) {
            n = 64;
        }
        if (targetSeconds < 1) {
            targetSeconds = 1;
        }
        if (warmup < 0) {
            warmup = 0;
        }
        if (threads < 1) {
            threads = 1;
        }

        printHeader(n, targetSeconds, warmup, threads);

        final long seed = 42L;
        double[] a = new double[n * n];
        double[] b = new double[n * n];
        double[] c = new double[n * n];
        fillRandom(a, seed);
        fillRandom(b, seed + 1);

        // Warmup lets HotSpot JIT optimize hot code paths before measurement.
        for (int i = 0; i < warmup; i++) {
            runMatMul(a, b, c, n, threads);
            zero(c);
        }

        double totalMs = 0.0;
        double bestMs = Double.MAX_VALUE;
        double worstMs = 0.0;
        double checksum = 0.0;

        int runs = 0;
        while (totalMs < targetSeconds * 1000.0 || runs == 0) {
            runs++;
            long t0 = System.nanoTime();
            runMatMul(a, b, c, n, threads);
            long t1 = System.nanoTime();

            double elapsedMs = (t1 - t0) / 1_000_000.0;
            totalMs += elapsedMs;
            bestMs = Math.min(bestMs, elapsedMs);
            worstMs = Math.max(worstMs, elapsedMs);

            checksum = checksum(c);
            double gflops = gflops(n, elapsedMs);
                System.out.println("run=" + runs
                    + " time_ms=" + DF3.format(elapsedMs)
                    + " gflops=" + DF3.format(gflops)
                    + " checksum=" + DF3.format(checksum));

            zero(c);
        }

        double avgMs = totalMs / runs;
        System.out.println();
        System.out.println("summary:");
        System.out.println("total_runs=" + runs);
        System.out.println("total_sec=" + DF3.format(totalMs / 1000.0));
        System.out.println("avg_ms=" + DF3.format(avgMs));
        System.out.println("best_ms=" + DF3.format(bestMs));
        System.out.println("worst_ms=" + DF3.format(worstMs));
        System.out.println("avg_gflops=" + DF3.format(gflops(n, avgMs)));
        System.out.println("best_gflops=" + DF3.format(gflops(n, bestMs)));
        System.out.println("last_checksum=" + DF3.format(checksum));
    }

    private static int parseArg(String[] args, int index, int defaultValue) {
        if (index >= args.length) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(args[index]);
        } catch (NumberFormatException ignored) {
            return defaultValue;
        }
    }

    private static void printHeader(int n, int targetSeconds, int warmup, int threads) {
        Runtime rt = Runtime.getRuntime();
        RuntimeMXBean mx = ManagementFactory.getRuntimeMXBean();

        long maxHeap = rt.maxMemory();
        long totalHeap = rt.totalMemory();
        long freeHeap = rt.freeMemory();

        System.out.println("java-bench: dense-matmul");
        System.out.println("timestamp=" + Instant.now());
        System.out.println("os.name=" + System.getProperty("os.name"));
        System.out.println("os.arch=" + System.getProperty("os.arch"));
        System.out.println("os.version=" + System.getProperty("os.version"));
        System.out.println("available.cores=" + rt.availableProcessors());
        System.out.println("java.version=" + System.getProperty("java.version"));
        System.out.println("java.vendor=" + System.getProperty("java.vendor"));
        System.out.println("jvm.name=" + System.getProperty("java.vm.name"));
        System.out.println("jvm.version=" + System.getProperty("java.vm.version"));
        System.out.println("jvm.uptime_ms=" + mx.getUptime());
        System.out.println("heap.max_mb=" + DF2.format(bytesToMb(maxHeap)));
        System.out.println("heap.total_mb=" + DF2.format(bytesToMb(totalHeap)));
        System.out.println("heap.free_mb=" + DF2.format(bytesToMb(freeHeap)));
        System.out.println("benchmark.size=" + n);
        System.out.println("benchmark.target_seconds=" + targetSeconds);
        System.out.println("benchmark.warmup=" + warmup);
        System.out.println("benchmark.threads=" + threads);
        System.out.println("benchmark.mode=" + (threads == 1 ? "single-thread" : "multi-thread"));
        System.out.println();
    }

    private static void runMatMul(double[] a, double[] b, double[] c, int n, int threads) {
        if (threads <= 1) {
            matMul(a, b, c, n);
            return;
        }
        matMulParallel(a, b, c, n, threads);
    }

    private static void fillRandom(double[] data, long seed) {
        Random r = new Random(seed);
        for (int i = 0; i < data.length; i++) {
            data[i] = r.nextDouble() - 0.5;
        }
    }

    private static void zero(double[] data) {
        for (int i = 0; i < data.length; i++) {
            data[i] = 0.0;
        }
    }

    /**
     * Straightforward i-k-j loop ordering, which is cache-friendly for B/C row-major access.
     */
    private static void matMul(double[] a, double[] b, double[] c, int n) {
        for (int i = 0; i < n; i++) {
            int iBase = i * n;
            for (int k = 0; k < n; k++) {
                double aik = a[iBase + k];
                int kBase = k * n;
                for (int j = 0; j < n; j++) {
                    c[iBase + j] += aik * b[kBase + j];
                }
            }
        }
    }

    private static void matMulParallel(double[] a, double[] b, double[] c, int n, int threads) {
        Thread[] workers = new Thread[threads];
        int chunk = (n + threads - 1) / threads;

        for (int t = 0; t < threads; t++) {
            final int startRow = t * chunk;
            final int endRow = Math.min(n, startRow + chunk);

            workers[t] = new Thread(() -> {
                for (int i = startRow; i < endRow; i++) {
                    int iBase = i * n;
                    for (int k = 0; k < n; k++) {
                        double aik = a[iBase + k];
                        int kBase = k * n;
                        for (int j = 0; j < n; j++) {
                            c[iBase + j] += aik * b[kBase + j];
                        }
                    }
                }
            });
            workers[t].start();
        }

        for (Thread worker : workers) {
            try {
                worker.join();
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Benchmark interrupted", ie);
            }
        }
    }

    private static double checksum(double[] data) {
        double s = 0.0;
        for (double v : data) {
            s += v;
        }
        return s;
    }

    private static double gflops(int n, double elapsedMs) {
        // Dense matrix multiply is approximately 2*n^3 floating point operations.
        double flops = 2.0 * n * n * n;
        double seconds = elapsedMs / 1000.0;
        return flops / seconds / 1_000_000_000.0;
    }

    private static double bytesToMb(long bytes) {
        return bytes / (1024.0 * 1024.0);
    }
}