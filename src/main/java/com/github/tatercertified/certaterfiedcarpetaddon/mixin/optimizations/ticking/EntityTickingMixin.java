package com.github.tatercertified.certaterfiedcarpetaddon.mixin.optimizations.ticking;

import com.github.tatercertified.certaterfiedcarpetaddon.Rules;
import net.fabricmc.api.EnvType;
import net.fabricmc.api.Environment;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityType;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.damage.DamageTracker;
import net.minecraft.world.World;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(LivingEntity.class)
abstract class EntityTickingMixin extends Entity {
    public EntityTickingMixin(EntityType<?> type, World world) {
        super(type, world);
    }

    @Shadow protected abstract void sendEquipmentChanges();

    @Shadow public abstract DamageTracker getDamageTracker();

    @Shadow public abstract boolean isSleeping();

    @Shadow protected abstract boolean isSleepingInBed();

    @Shadow public abstract void wakeUp();

    @Environment(EnvType.SERVER)
    @Redirect(method = "tick", at = @At(value = "FIELD", target = "Lnet/minecraft/world/World;isClient:Z"))
    private boolean redirectGetClient(World instance) {
        if (Rules.optimizedEntityTicking) {
            return this.getType() == EntityType.PLAYER;
        } else {
            return true;
        }
    }

    @Environment(EnvType.SERVER)
    @Inject(method = "tick", at = @At(value = "INVOKE", target = "Lnet/minecraft/entity/LivingEntity;isRemoved()Z"))
    private void injectExtraCode(CallbackInfo ci) {
        if (Rules.optimizedEntityTicking && this.getType() != EntityType.PLAYER) {
            this.sendEquipmentChanges();
            if (this.age % 20 == 0) {
                this.getDamageTracker().update();
            }

            if (this.isSleeping() && !this.isSleepingInBed()) {
                this.wakeUp();
            }
        }
    }

}
