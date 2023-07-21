package com.github.tatercertified.certaterfiedcarpetaddon.utils;

import jcuda.*;
import jcuda.jcurand.curandGenerator;

import static jcuda.jcurand.JCurand.*;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class JCurand {
    public static curandGenerator cudaRNG;
    public static void startCUDARandom() {
        int n = 4;
    // Allocate device memory
        Pointer deviceData = new Pointer();
        cudaMalloc(deviceData, n * Sizeof.FLOAT);

    // Create and initialize a pseudo-random number generator
        cudaRNG = new curandGenerator();
        curandCreateGenerator(cudaRNG, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(cudaRNG, 0L);

    // Generate random numbers
        curandGenerateUniform(cudaRNG, deviceData, n);

    // Copy the random numbers from the device to the host
        float[] hostData = new float[n];
        cudaMemcpy(Pointer.to(hostData), deviceData,
                n * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
    }

    /**
     * Sets the seed of the CUDA RNG
     * @param seed seed as long
     */
    public static void setSeed(long seed) {
        curandSetPseudoRandomGeneratorSeed(cudaRNG, seed);
    }
}
