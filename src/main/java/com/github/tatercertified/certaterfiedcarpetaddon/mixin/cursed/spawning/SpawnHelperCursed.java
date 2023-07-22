package com.github.tatercertified.certaterfiedcarpetaddon.mixin.cursed.spawning;

import com.github.tatercertified.certaterfiedcarpetaddon.Rules;
import net.minecraft.block.BlockState;
import net.minecraft.entity.EntityType;
import net.minecraft.entity.SpawnGroup;
import net.minecraft.entity.SpawnRestriction;
import net.minecraft.entity.mob.MobEntity;
import net.minecraft.fluid.FluidState;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.BlockView;
import net.minecraft.world.SpawnHelper;
import net.minecraft.world.WorldView;
import net.minecraft.world.biome.SpawnSettings;
import net.minecraft.world.gen.StructureAccessor;
import net.minecraft.world.gen.chunk.ChunkGenerator;
import org.jetbrains.annotations.Nullable;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(SpawnHelper.class)
public class SpawnHelperCursed {
    @Inject(method = "isClearForSpawn", at = @At("HEAD"), cancellable = true)
    private static void trickIsClearForSpawn(BlockView blockView, BlockPos pos, BlockState state, FluidState fluidState, EntityType<?> entityType, CallbackInfoReturnable<Boolean> cir) {
        if (Rules.cursedLagFreeSpawning) {
            if (state.isFullCube(blockView, pos)) {
                cir.setReturnValue(false);
            }
            cir.setReturnValue(true);
        }
    }

    @Inject(method = "canSpawn(Lnet/minecraft/entity/SpawnRestriction$Location;Lnet/minecraft/world/WorldView;Lnet/minecraft/util/math/BlockPos;Lnet/minecraft/entity/EntityType;)Z", at = @At("HEAD"), cancellable = true)
    private static void trickCanSpawn(SpawnRestriction.Location location, WorldView world, BlockPos pos, @Nullable EntityType<?> entityType, CallbackInfoReturnable<Boolean> cir) {
        if (Rules.cursedLagFreeSpawning) {
            cir.setReturnValue(true);
        }
    }

    @Inject(method = "canSpawn(Lnet/minecraft/server/world/ServerWorld;Lnet/minecraft/entity/SpawnGroup;Lnet/minecraft/world/gen/StructureAccessor;Lnet/minecraft/world/gen/chunk/ChunkGenerator;Lnet/minecraft/world/biome/SpawnSettings$SpawnEntry;Lnet/minecraft/util/math/BlockPos$Mutable;D)Z", at = @At("HEAD"), cancellable = true)
    private static void trickCanSpawn(ServerWorld world, SpawnGroup group, StructureAccessor structureAccessor, ChunkGenerator chunkGenerator, SpawnSettings.SpawnEntry spawnEntry, BlockPos.Mutable pos, double squaredDistance, CallbackInfoReturnable<Boolean> cir) {
        if (Rules.cursedLagFreeSpawning) {
            cir.setReturnValue(true);
        }
    }

    @Inject(method = "isValidSpawn", at = @At("HEAD"), cancellable = true)
    private static void trickIsValidSpawn(ServerWorld world, MobEntity entity, double squaredDistance, CallbackInfoReturnable<Boolean> cir) {
        if (Rules.cursedLagFreeSpawning) {
            cir.setReturnValue(true);
        }
    }

    @Inject(method = "shouldUseNetherFortressSpawns", at = @At("HEAD"), cancellable = true)
    private static void trickNetherFortress(BlockPos pos, ServerWorld world, SpawnGroup spawnGroup, StructureAccessor structureAccessor, CallbackInfoReturnable<Boolean> cir) {
        if (Rules.cursedLagFreeSpawning) {
            cir.setReturnValue(false);
        }
    }
}
