package com.github.tatercertified.certaterfiedcarpetaddon.mixin.optimizations.random;

import com.github.tatercertified.certaterfiedcarpetaddon.CertaterfiedCarpetAddon;
import com.github.tatercertified.certaterfiedcarpetaddon.Rules;
import com.github.tatercertified.certaterfiedcarpetaddon.utils.CUDARandom;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.random.Random;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.util.UUID;

@Mixin(MathHelper.class)
public class MathHelperMixin {

    @Inject(method = "nextGaussian", at = @At("HEAD"), cancellable = true)
    private static void optimizedGaussian(Random random, float mean, float deviation, CallbackInfoReturnable<Float> cir) {
        if (Rules.optimizedRandom) {
            if (Rules.cursedOptimizedRandom) {
                cir.setReturnValue((float) nextFastGaussian());
            } else if (Rules.optimizedRandomCUDA) {
                cir.setReturnValue((float) (CUDARandom.nextGaussian() * deviation));
            } else {
                cir.setReturnValue((float) (mean + nextFastGaussian() * deviation));
            }
        }
    }

    @Inject(method = "nextBetween(Lnet/minecraft/util/math/random/Random;II)I", at = @At("HEAD"), cancellable = true)
    private static void optimizedNextBetween(Random random, int min, int max, CallbackInfoReturnable<Integer> cir) {
        if (Rules.optimizedRandom) {
            if (Rules.optimizedRandomCUDA) {
                cir.setReturnValue(CUDARandom.nextInt(min, max));
            } else {
                cir.setReturnValue(CertaterfiedCarpetAddon.fastRandom.nextBetween(min, max));
            }
        }
    }

    @Inject(method = "nextBetween(Lnet/minecraft/util/math/random/Random;FF)F", at = @At("HEAD"), cancellable = true)
    private static void optimizedNextBetween(Random random, float min, float max, CallbackInfoReturnable<Float> cir) {
        if (Rules.optimizedRandom) {
            if (Rules.optimizedRandomCUDA) {
                cir.setReturnValue(CUDARandom.nextFloat(min, max));
            } else {
                cir.setReturnValue(CertaterfiedCarpetAddon.fastRandom.nextFloat() * (max - min) + min);
            }
        }
    }

    private static long counter = 0L;

    @Inject(method = "randomUuid(Lnet/minecraft/util/math/random/Random;)Ljava/util/UUID;", at = @At("HEAD"), cancellable = true)
    private static void optimizedRandomUUID(Random random, CallbackInfoReturnable<UUID> cir) {
        if (Rules.optimizedRandom) {
            long m;
            long l;
            if (Rules.cursedOptimizedRandom) {
                m = 0x8000000000000000L | (counter & 0x0000FFFFFFFFFFFFL);
                l = Long.MIN_VALUE | ((counter >> 48) & 0x0000000000000FFFL) | 0x4000000000000000L;
                counter++;
            } else {
                m = CertaterfiedCarpetAddon.fastRandom.nextLong() & 0xFFFFFFFFFFFF0FFFL | 0x4000L;
                l = CertaterfiedCarpetAddon.fastRandom.nextLong() & 0x3FFFFFFFFFFFFFFFL | Long.MIN_VALUE;
            }
            cir.setReturnValue(new UUID(l, m));


            // TODO Fixes CUDA longs
            // } else if (Rules.optimizedRandomCUDA) {
            //     long m = CUDARandom.nextLong() & 0xFFFFFFFFFFFF0FFFL | 0x4000L;
            //     long l = CUDARandom.nextLong() & 0x3FFFFFFFFFFFFFFFL | Long.MIN_VALUE;
            //     cir.setReturnValue(new UUID(l, m));
        }
    }

    @Inject(method = "nextInt", at = @At("HEAD"), cancellable = true)
    private static void optimizedNextInt(Random random, int min, int max, CallbackInfoReturnable<Integer> cir) {
        if (Rules.optimizedRandom) {
            if (min >= max) {
                cir.setReturnValue(min);
            } else if (Rules.cursedOptimizedRandom) {
                cir.setReturnValue(CertaterfiedCarpetAddon.fastRandom.nextInt());
            } else if (Rules.optimizedRandomCUDA) {
                cir.setReturnValue(CUDARandom.nextInt(min, max));
            } else {
                cir.setReturnValue(CertaterfiedCarpetAddon.fastRandom.nextBetween(min, max));
            }
        }
    }

    @Inject(method = "nextFloat", at = @At("HEAD"), cancellable = true)
    private static void optimizedNextFloat(Random random, float min, float max, CallbackInfoReturnable<Float> cir) {
        if (Rules.optimizedRandom) {
            if (min >= max) {
                cir.setReturnValue(min);
            } else if (Rules.cursedOptimizedRandom) {
                cir.setReturnValue(CertaterfiedCarpetAddon.fastRandom.nextFloat());
            } else if (Rules.optimizedRandomCUDA) {
                cir.setReturnValue(CUDARandom.nextFloat(min, max));
            } else {
                cir.setReturnValue(CertaterfiedCarpetAddon.fastRandom.nextFloat() * (max - min) + min);
            }
        }
    }

    @Inject(method = "nextDouble", at = @At("HEAD"), cancellable = true)
    private static void optimizedNextDouble(Random random, double min, double max, CallbackInfoReturnable<Double> cir) {
        if (Rules.optimizedRandom) {
            if (min >= max) {
                cir.setReturnValue(min);
            } else if (Rules.cursedOptimizedRandom) {
                cir.setReturnValue(CertaterfiedCarpetAddon.fastRandom.nextDouble());
            } else if (Rules.optimizedRandomCUDA) {
                cir.setReturnValue(CUDARandom.nextDouble(min, max));
            } else {
                cir.setReturnValue(CertaterfiedCarpetAddon.fastRandom.nextDouble() * (max - min) + min);
            }
        }
    }

    private static double nextFastGaussian() {
        long randomBits = CertaterfiedCarpetAddon.fastRandom.nextLong();
        return quickGaussian(randomBits);
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
        sum &= (1<<10)-1;
        return sum;
    }

    private static final long EVEN_CHUNKS = 0x7c1f07c1f07c1fL;
    private static final long ODD_CHUNKS  = EVEN_CHUNKS << 5;
}
