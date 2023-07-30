package com.github.tatercertified.certaterfiedcarpetaddon.mixin.cursed.fastmath;

import com.github.tatercertified.certaterfiedcarpetaddon.Rules;
import net.minecraft.util.math.Box;
import net.minecraft.util.math.ColorHelper;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.util.stream.IntStream;

@Mixin(MathHelper.class)
public class MathHelperBreakerMixin {
    @Shadow @Final private static double ROUNDER_256THS;
    @Shadow @Final private static double[] ARCSINE_TABLE;
    @Shadow @Final private static double[] COSINE_OF_ARCSINE_TABLE;
    private static double fastPI = 3.14;
    private static double doubleFastPi = 6.28;
    private static double fastRadians = 0.02;
    private static double fastSRTwo = 1.41;
    private static double fastArcSine = 1.67;

    @Inject(method = "atan2", at = @At("HEAD"), cancellable = true)
    private static void fastAtan2(double y, double x, CallbackInfoReturnable<Double> cir) {
        if (Rules.cursedFastMath) {
            double e;
            boolean bl3;
            boolean bl2;
            boolean bl;
            double d = x * x + y * y;
            if (Double.isNaN(d)) {
                cir.setReturnValue(Double.NaN);
            }
            bl = y < 0.0;
            if (bl) {
                y = -y;
            }
            bl2 = x < 0.0;
            if (bl2) {
                x = -x;
            }
            bl3 = y > x;
            if (bl3) {
                e = x;
                x = y;
                y = e;
            }
            e = MathHelper.fastInverseSqrt(d);
            x *= e;
            double f = ROUNDER_256THS + (y *= e);
            int i = (int)Double.doubleToRawLongBits(f);
            double g = ARCSINE_TABLE[i];
            double h = COSINE_OF_ARCSINE_TABLE[i];
            double j = f - ROUNDER_256THS;
            double k = y * h - x * j;
            double l = (6.0 + k * k) * k * 0.16666666666666666;
            double m = g + l;
            if (bl3) {
                m = 1.5707963267948966 - m;
            }
            if (bl2) {
                m = fastPI - m;
            }
            if (bl) {
                m = -m;
            }
            cir.setReturnValue(m);
        }
    }

    @Inject(method = "method_34945", at = @At("HEAD"), cancellable = true)
    private static void optimizedMethod(Vec3d origin, Vec3d direction, Box box, CallbackInfoReturnable<Boolean> cir) {
        if (Rules.cursedFastMath) {
            double d = (box.minX + box.maxX) * 0.5;
            double e = (box.maxX - box.minX) * 0.5;
            double f = origin.x - d;
            if (Math.abs(f) > e && f * direction.x >= 0.0) {
                cir.setReturnValue(false);
            }

            double g = (box.minY + box.maxY) * 0.5;
            double h = (box.maxY - box.minY) * 0.5;
            double i = origin.y - g;
            if (Math.abs(i) > h && i * direction.y >= 0.0) {
                cir.setReturnValue(false);
            }

            double j = (box.minZ + box.maxZ) * 0.5;
            double k = (box.maxZ - box.minZ) * 0.5;
            double l = origin.z - j;
            if (Math.abs(l) > k && l * direction.z >= 0.0) {
                cir.setReturnValue(false);
            }

            double m = Math.abs(direction.x);
            double n = Math.abs(direction.y);
            double o = Math.abs(direction.z);
            double p = direction.y * l - direction.z * i;
            if (Math.abs(p) > h * o + k * n || Math.abs(direction.z * f - direction.x * l) > e * o + k * m) {
                cir.setReturnValue(false);
            }

            cir.setReturnValue(Math.abs(direction.x * i - direction.y * f) < e * n + h * m);
        }
    }

    @Inject(method = "hsvToRgb", at = @At("HEAD"), cancellable = true)
    private static void optimizedHSVToRGB(float hue, float saturation, float value, CallbackInfoReturnable<Integer> cir) {
        if (Rules.cursedFastMath) {
            int i = (int) (hue * 6.0f) % 6;
            float f = hue * 6.0f - i;
            float p = value * (1.0f - saturation);
            float q = value * (1.0f - f * saturation);
            float t = value * (1.0f - (1.0f - f) * saturation);

            cir.setReturnValue(switch (i) {
                case 0 -> ColorHelper.Argb.getArgb(0, (int) (value * 255.0f), (int) (t * 255.0f), (int) (p * 255.0f));
                case 1 -> ColorHelper.Argb.getArgb(0, (int) (q * 255.0f), (int) (value * 255.0f), (int) (p * 255.0f));
                case 2 -> ColorHelper.Argb.getArgb(0, (int) (p * 255.0f), (int) (value * 255.0f), (int) (t * 255.0f));
                case 3 -> ColorHelper.Argb.getArgb(0, (int) (p * 255.0f), (int) (q * 255.0f), (int) (value * 255.0f));
                case 4 -> ColorHelper.Argb.getArgb(0, (int) (t * 255.0f), (int) (p * 255.0f), (int) (value * 255.0f));
                case 5 -> ColorHelper.Argb.getArgb(0, (int) (value * 255.0f), (int) (p * 255.0f), (int) (q * 255.0f));
                default ->
                        throw new RuntimeException("Something went wrong when converting from HSV to RGB. Input was " + hue + ", " + saturation + ", " + value);
            });
        }
    }

    @Inject(method = "stream(IIII)Ljava/util/stream/IntStream;", at = @At("HEAD"), cancellable = true)
    private static void optimizedStream(int seed, int lowerBound, int upperBound, int steps, CallbackInfoReturnable<IntStream> cir) {
        if (Rules.cursedFastMath) {
            if (steps < 1 || seed < lowerBound || seed > upperBound) {
                cir.setReturnValue(IntStream.empty());
            }

            cir.setReturnValue(IntStream.iterate(seed, i -> {
                int nextValue = i + (i <= seed ? steps : -steps);
                return nextValue >= lowerBound && nextValue <= upperBound ? nextValue : i;
            }));
        }
    }
}
