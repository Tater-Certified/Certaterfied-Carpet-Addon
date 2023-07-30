package com.github.tatercertified.certaterfiedcarpetaddon;

import carpet.api.settings.Rule;
import carpet.api.settings.RuleCategory;

public class Rules {
    // Optimizations
    @Rule(
            categories = {RuleCategory.OPTIMIZATION, "certaterfied"},
            strict = false
    )
    public static boolean optimizedRandom;

    @Rule(
            categories = {RuleCategory.OPTIMIZATION, "cursed", "certaterfied"},
            strict = false
    )
    public static boolean cursedOptimizedRandom;

    @Rule(
            categories = {RuleCategory.OPTIMIZATION, "nvidia", "certaterfied"},
            strict = false
    )
    public static boolean optimizedRandomCUDA;
    @Rule(
            categories = {RuleCategory.OPTIMIZATION, "certaterfied"},
            strict = false
    )
    public static boolean optimizedPortalCollisions;
    @Rule(
            categories = {RuleCategory.OPTIMIZATION, "certaterfied"},
            strict = false
    )
    public static boolean optimizedEntityTicking;

    // Cursed
    @Rule(
            categories = {"cursed", "certaterfied"},
            strict = false
    )
    public static boolean cursedLagFreeSpawning;
    @Rule(
            categories = {"cursed, certaterfied"},
            strict = false
    )
    public static boolean cursedFastMath;
}
