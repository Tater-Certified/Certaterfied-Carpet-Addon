package com.github.tatercertified.certaterfiedcarpetaddon.mixin.optimizations.random;

import com.github.tatercertified.certaterfiedcarpetaddon.CertaterfiedCarpetAddon;
import com.github.tatercertified.certaterfiedcarpetaddon.Rules;
import net.minecraft.entity.Entity;
import net.minecraft.sound.SoundEvent;
import net.minecraft.sound.SoundEvents;
import net.minecraft.util.math.random.Random;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(Entity.class)
public abstract class EntityMixin {
    @Shadow public abstract void playSound(SoundEvent sound, float volume, float pitch);

    @Shadow protected abstract SoundEvent getSwimSound();

    @Shadow public abstract void remove(Entity.RemovalReason reason);

    @Shadow public abstract double offsetX(double widthScale);

    @Shadow public abstract double offsetZ(double widthScale);

    @Shadow public abstract double getBodyY(double heightScale);

    @Inject(method = "setOnFireFromLava", at = @At(value = "INVOKE", target = "Lnet/minecraft/entity/Entity;playSound(Lnet/minecraft/sound/SoundEvent;FF)V"), cancellable = true)
    private void optimizedSetOnFireFromLava(CallbackInfo ci) {
        if (Rules.optimizedRandom) {
            this.playSound(SoundEvents.ENTITY_GENERIC_BURN, 0.4F, 2.0F + CertaterfiedCarpetAddon.fastRandom.nextFloat() * 0.4F);
            ci.cancel();
        }
    }

    @Inject(method = "playExtinguishSound", at = @At(value = "HEAD"), cancellable = true)
    private void optimizedPlayExtinguishSound(CallbackInfo ci) {
        if (Rules.optimizedRandom) {
            this.playSound(SoundEvents.ENTITY_GENERIC_EXTINGUISH_FIRE, 0.7F, 1.6F + CertaterfiedCarpetAddon.fastRandom.nextFloat() * 0.4F);
            ci.cancel();
        }
    }

    @Redirect(method = "playAmethystChimeSound", at = @At(value = "INVOKE", target = "Lnet/minecraft/util/math/random/Random;nextFloat()F"))
    private float optimizedPlayAmethystChimeSound(Random instance) {
        if (Rules.optimizedRandom) {
            return CertaterfiedCarpetAddon.fastRandom.nextFloat();
        } else {
            return instance.nextFloat();
        }
    }

    @Inject(method = "playSwimSound(F)V", at = @At("HEAD"))
    private void optimizedPlaySwingSound(float volume, CallbackInfo ci) {
        if (Rules.optimizedRandom) {
            this.playSound(this.getSwimSound(), volume, 1.0F + CertaterfiedCarpetAddon.fastRandom.nextFloat() * 0.4F);
        }
    }

    @Redirect(method = "onSwimmingStart", at = @At(value = "INVOKE", target = "Lnet/minecraft/util/math/random/Random;nextDouble()D"))
    private double optimizedOnSwimmingStart(Random instance) {
        if (Rules.optimizedRandom) {
            return CertaterfiedCarpetAddon.fastRandom.nextDouble();
        } else {
           return instance.nextDouble();
        }
    }

    @Redirect(method = "spawnSprintingParticles", at = @At(value = "INVOKE", target = "Lnet/minecraft/util/math/random/Random;nextDouble()D"))
    private double optimizedSpawnSprintingParticles(Random instance) {
        if (Rules.optimizedRandom) {
            return CertaterfiedCarpetAddon.fastRandom.nextDouble();
        } else {
            return instance.nextDouble();
        }
    }

    @Redirect(method = "pushOutOfBlocks", at = @At(value = "INVOKE", target = "Lnet/minecraft/util/math/random/Random;nextFloat()F"))
    private float optimizedPushOutOfBlocks(Random instance) {
        if (Rules.optimizedRandom) {
            return CertaterfiedCarpetAddon.fastRandom.nextFloat();
        } else {
            return instance.nextFloat();
        }
    }

    @Inject(method = "getRandomBodyY", at = @At("HEAD"), cancellable = true)
    private void optimizedGetRandomBodyY(CallbackInfoReturnable<Double> cir) {
        if (Rules.optimizedRandom) {
            cir.setReturnValue(this.getBodyY(CertaterfiedCarpetAddon.fastRandom.nextDouble()));
        }
    }

    @Inject(method = "getParticleX", at = @At("HEAD"), cancellable = true)
    private void optimizedGetParticleX(double widthScale, CallbackInfoReturnable<Double> cir) {
        if (Rules.optimizedRandom) {
            cir.setReturnValue(this.offsetX((2.0 * CertaterfiedCarpetAddon.fastRandom.nextDouble() - 1.0) * widthScale));
        }
    }

    @Inject(method = "getParticleZ", at = @At("HEAD"), cancellable = true)
    private void optimizedGetParticleZ(double widthScale, CallbackInfoReturnable<Double> cir) {
        if (Rules.optimizedRandom) {
            cir.setReturnValue(this.offsetZ((2.0 * CertaterfiedCarpetAddon.fastRandom.nextDouble() - 1.0) * widthScale));
        }
    }
}
