package com.github.tatercertified.certaterfiedcarpetaddon.utils;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcurand.curandGenerator;
import static jcuda.jcurand.JCurand.*;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class JCurand {
    private static final int N = 64;
    private static final int MEMORY_CLEAR_INTERVAL = 100000; // Clear memory every 100000 random number generations

    private static curandGenerator cudaRNG;
    private static Pointer deviceData;
    private static float[] hostData;
    private static long randomNumberCount = 0;

    // Bit-twiddling constants
    static final long EVEN_CHUNKS = 0x7c1f07c1f07c1fL;
    static final long ODD_CHUNKS  = EVEN_CHUNKS << 5;

    /**
     * Start RNG
     */
    public static void initialize() {
        hostData = new float[N];
        deviceData = new Pointer();
        cudaMalloc(deviceData, N * Sizeof.FLOAT);

        cudaRNG = new curandGenerator();
        curandCreateGenerator(cudaRNG, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(cudaRNG, 0L);
    }

    /**
     * Stop RNG
     */
    public static void shutdown() {
        cudaFree(deviceData);
        curandDestroyGenerator(cudaRNG);
    }

    /**
     * Get next random Integer
     * @param min smallest int
     * @param max largest int
     * @return random Integer
     */
    public static int nextInt(int min, int max) {
        if (min >= max) {
            return min;
        }
        int range = max - min + 1;
        return min + (int) (nextFloat() * range);
    }

    /**
     * Get next random Float
     * @param min smallest float
     * @param max largest float
     * @return random Float
     */
    public static float nextFloat(float min, float max) {
        if (min >= max) {
            return min;
        }
        return min + nextFloat() * (max - min);
    }

    /**
     * Get next random Double
     * @param min smallest double
     * @param max largest double
     * @return random Double
     */
    public static double nextDouble(double min, double max) {
        if (min >= max) {
            return min;
        }
        return min + nextDouble() * (max - min);
    }

    /**
     * Get next random Integer
     * @return random Integer
     */
    public static int nextInt() {
        return nextInt(Integer.MIN_VALUE, Integer.MAX_VALUE);
    }

    /**
     * Get next random Float
     * @return random Float
     */
    public static float nextFloat() {
        generateRandomNumbers();
        return hostData[N - 1];
    }

    /**
     * Get next random Double
     * @return random Double
     */
    public static double nextDouble() {
        generateRandomNumbers();
        return hostData[N - 1];
    }

    public static long nextLong() {
        generateRandomLongs();
        return (long) hostData[N - 1];
    }

    /**
     * Get next random Gaussian
     * @return random Gaussian
     */
    public static double nextGaussian() {
        generateRandomNumbers();
        return quickGaussian((long) (hostData[N - 1] * (1L << 31)));
    }

    private static double quickGaussian(long randomBits) {
        long evenChunks = randomBits & EVEN_CHUNKS;
        long oddChunks = (randomBits & ODD_CHUNKS) >>> 5;
        long sum = chunkSum(evenChunks + oddChunks) - 186; // Mean = 31*12 / 2
        return sum / 31.0;
    }

    private static long chunkSum(long bits) {
        long sum = bits + (bits >>> 40);
        sum += sum >>> 20;
        sum += sum >>> 10;
        sum &= (1L << 10) - 1;
        return sum;
    }

    private static void generateRandomNumbers() {
        curandGenerateUniform(cudaRNG, deviceData, N);
        cudaMemcpy(Pointer.to(hostData), deviceData, N * Sizeof.FLOAT, cudaMemcpyDeviceToHost);

        randomNumberCount++;

        // Check if it's time to clear memory
        if (randomNumberCount % MEMORY_CLEAR_INTERVAL == 0) {
            clearMemory();
        }
    }

    private static void generateRandomLongs() {
        curandGenerateLongLong(cudaRNG, deviceData, N);
        cudaMemcpy(Pointer.to(hostData), deviceData, N * Sizeof.LONG, cudaMemcpyDeviceToHost);

        randomNumberCount++;

        // Check if it's time to clear memory
        if (randomNumberCount % MEMORY_CLEAR_INTERVAL == 0) {
            clearMemory();
        }
    }

    private static void clearMemory() {
        cudaFree(deviceData);
        cudaMalloc(deviceData, N * Sizeof.FLOAT);
    }
}