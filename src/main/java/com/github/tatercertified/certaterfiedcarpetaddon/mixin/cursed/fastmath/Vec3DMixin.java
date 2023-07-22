package com.github.tatercertified.certaterfiedcarpetaddon.mixin.cursed.fastmath;

import com.github.tatercertified.certaterfiedcarpetaddon.Rules;
import net.minecraft.util.math.Vec3d;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Mutable;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(Vec3d.class)
public class Vec3DMixin {
    @Mutable
    @Shadow @Final public double x;

    @Mutable
    @Shadow @Final public double y;

    @Mutable
    @Shadow @Final public double z;
    private Vec3d cachedNormalized = null;

    @Inject(method = "unpackRgb", at = @At("HEAD"), cancellable = true)
    private static void optimizedUnpackRGB(int rgb, CallbackInfoReturnable<Vec3d> cir) {
        if (Rules.cursedFastMath) {
            double d = (double)(rgb >> 16 & 0xFF) / 255.0;
            double e = (double)(rgb >> 8 & 0xFF) / 255.0;
            double f = (double)(rgb & 0xFF) / 255.0;
            cir.setReturnValue(new Vec3d(d, e, f));
        }
    }

    @Inject(method = "normalize", at = @At("HEAD"), cancellable = true)
    private void optimizedNormalize(CallbackInfoReturnable<Vec3d> cir) {
        if (Rules.cursedFastMath) {
            if (cachedNormalized == null) {
                double squaredLength = x * x + y * y + z * z;
                if (squaredLength < 1.0E-8) { // Tweak this threshold value if needed
                    cachedNormalized = Vec3d.ZERO;
                } else {
                    double invLength = 1.0 / Math.sqrt(squaredLength);
                    cachedNormalized = new Vec3d(x * invLength, y * invLength, z * invLength);
                }
            }
            cir.setReturnValue(cachedNormalized);
        }
    }

    @Inject(method = "relativize", at = @At("HEAD"))
    private void optimizedRelativize(Vec3d vec, CallbackInfoReturnable<Vec3d> cir) {
        if (Rules.cursedFastMath) {
            this.x -= vec.x;
            this.y -= vec.y;
            this.z -= vec.z;
        }
    }
}
