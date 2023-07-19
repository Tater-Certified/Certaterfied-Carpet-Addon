package com.github.tatercertified.certaterfiedcarpetaddon;

import carpet.api.settings.Rule;
import carpet.api.settings.RuleCategory;

public class Rules {
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
}
