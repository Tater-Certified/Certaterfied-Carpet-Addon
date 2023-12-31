package com.github.tatercertified.certaterfiedcarpetaddon.mixin.optimizations.collisions;

import com.github.tatercertified.certaterfiedcarpetaddon.Rules;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityDimensions;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.dimension.NetherPortal;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(NetherPortal.class)
public class NetherPortalMixin {
    @Inject(method = "findOpenPosition", at = @At("HEAD"), cancellable = true)
    private static void findOpenPos(Vec3d fallback, ServerWorld world, Entity entity, EntityDimensions dimensions, CallbackInfoReturnable<Vec3d> cir) {
        if (Rules.optimizedPortalCollisions) {
            cir.setReturnValue(fallback);
        }
    }
}
