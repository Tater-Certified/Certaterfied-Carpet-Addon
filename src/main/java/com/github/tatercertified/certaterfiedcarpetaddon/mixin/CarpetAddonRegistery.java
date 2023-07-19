package com.github.tatercertified.certaterfiedcarpetaddon.mixin;

import com.github.tatercertified.certaterfiedcarpetaddon.CertaterfiedCarpetAddon;
import net.minecraft.util.crash.CrashReport;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(CrashReport.class)
public class CarpetAddonRegistery {
    @Inject(method = "initCrashReport", at = @At("HEAD"))
    private static void gameStarted(CallbackInfo ci) {
        CertaterfiedCarpetAddon.init();
    }
}
