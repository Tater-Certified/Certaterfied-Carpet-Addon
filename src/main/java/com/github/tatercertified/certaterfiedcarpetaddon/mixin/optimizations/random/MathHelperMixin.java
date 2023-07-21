package com.github.tatercertified.certaterfiedcarpetaddon.mixin.optimizations.random;

import com.github.tatercertified.certaterfiedcarpetaddon.CertaterfiedCarpetAddon;
import com.github.tatercertified.certaterfiedcarpetaddon.Rules;
import com.github.tatercertified.certaterfiedcarpetaddon.utils.JCurand;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.random.Random;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.util.UUID;

@Mixin(MathHelper.class)
public class MathHelperMixin {

    @Unique


    @Inject(method = "nextGaussian", at = @At("HEAD"), cancellable = true)
    private static void optimizedGaussian(Random random, float mean, float deviation, CallbackInfoReturnable<Float> cir) {
        if (Rules.optimizedRandom) {
            if (Rules.cursedOptimizedRandom) {
                cir.setReturnValue((float) CertaterfiedCarpetAddon.fastRandom.nextGaussian());
            }
            cir.setReturnValue(mean + (float) CertaterfiedCarpetAddon.fastRandom.nextGaussian() * deviation);
        } else if (Rules.optimizedRandomCUDA) {
            cir.setReturnValue((float) (JCurand.nextGaussian() * deviation));
        }
    }

    @Inject(method = "nextBetween(Lnet/minecraft/util/math/random/Random;II)I", at = @At("HEAD"), cancellable = true)
    private static void optimizedNextBetween(Random random, int min, int max, CallbackInfoReturnable<Integer> cir) {
        if (Rules.optimizedRandom) {
            cir.setReturnValue(CertaterfiedCarpetAddon.fastRandom.nextBetween(min, max));
        } else if (Rules.optimizedRandomCUDA) {
            cir.setReturnValue(JCurand.nextInt(min, max));
        }
    }

    @Inject(method = "nextBetween(Lnet/minecraft/util/math/random/Random;FF)F", at = @At("HEAD"), cancellable = true)
    private static void optimizedNextBetween(Random random, float min, float max, CallbackInfoReturnable<Float> cir) {
        if (Rules.optimizedRandom) {
            cir.setReturnValue(CertaterfiedCarpetAddon.fastRandom.nextFloat() * (max - min) + min);
        } else if (Rules.optimizedRandomCUDA) {
            cir.setReturnValue(JCurand.nextFloat(min, max));
        }
    }

    @Inject(method = "randomUuid(Lnet/minecraft/util/math/random/Random;)Ljava/util/UUID;", at = @At("HEAD"), cancellable = true)
    private static void optimizedRandomUUID(Random random, CallbackInfoReturnable<UUID> cir) {
        if (Rules.optimizedRandom) {
            long m = CertaterfiedCarpetAddon.fastRandom.nextLong() & 0xFFFFFFFFFFFF0FFFL | 0x4000L;
            long l = CertaterfiedCarpetAddon.fastRandom.nextLong() & 0x3FFFFFFFFFFFFFFFL | Long.MIN_VALUE;
            cir.setReturnValue(new UUID(l, m));
        } else if (Rules.optimizedRandomCUDA) {
            long m = JCurand.nextLong() & 0xFFFFFFFFFFFF0FFFL | 0x4000L;
            long l = JCurand.nextLong() & 0x3FFFFFFFFFFFFFFFL | Long.MIN_VALUE;
            cir.setReturnValue(new UUID(l, m));
        }
    }

    @Inject(method = "nextInt", at = @At("HEAD"), cancellable = true)
    private static void optimizedNextInt(Random random, int min, int max, CallbackInfoReturnable<Integer> cir) {
        if (Rules.optimizedRandom) {
            if (min >= max) {
                cir.setReturnValue(min);
            }
            if (Rules.cursedOptimizedRandom) {
                cir.setReturnValue(CertaterfiedCarpetAddon.fastRandom.nextInt());
            }
            cir.setReturnValue(CertaterfiedCarpetAddon.fastRandom.nextBetween(min, max));
        } else if (Rules.optimizedRandomCUDA) {
            cir.setReturnValue(JCurand.nextInt(min, max));
        }
    }

    @Inject(method = "nextFloat", at = @At("HEAD"), cancellable = true)
    private static void optimizedNextFloat(Random random, float min, float max, CallbackInfoReturnable<Float> cir) {
        if (Rules.optimizedRandom) {
            if (min >= max) {
                cir.setReturnValue(min);
            }
            if (Rules.cursedOptimizedRandom) {
                cir.setReturnValue(CertaterfiedCarpetAddon.fastRandom.nextFloat());
            }
            cir.setReturnValue(CertaterfiedCarpetAddon.fastRandom.nextFloat() * (max - min) + min);
        } else if (Rules.optimizedRandomCUDA) {
            cir.setReturnValue(JCurand.nextFloat(min, max));
        }
    }

    @Inject(method = "nextDouble", at = @At("HEAD"), cancellable = true)
    private static void optimizedNextDouble(Random random, double min, double max, CallbackInfoReturnable<Double> cir) {
        if (Rules.optimizedRandom) {
            if (min >= max) {
                cir.setReturnValue(min);
            }
            if (Rules.cursedOptimizedRandom) {
                cir.setReturnValue(CertaterfiedCarpetAddon.fastRandom.nextDouble());
            }
            cir.setReturnValue(CertaterfiedCarpetAddon.fastRandom.nextDouble() * (max - min) + min);
        } else if (Rules.optimizedRandomCUDA) {
            cir.setReturnValue(JCurand.nextDouble(min, max));
        }
    }
}
