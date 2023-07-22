package com.github.tatercertified.certaterfiedcarpetaddon;

import carpet.CarpetExtension;
import carpet.CarpetServer;
import com.github.tatercertified.certaterfiedcarpetaddon.utils.CUDARandom;
import com.google.common.reflect.TypeToken;
import com.google.gson.GsonBuilder;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.math.random.Xoroshiro128PlusPlusRandom;
import org.apache.commons.io.IOUtils;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Map;

public class CertaterfiedCarpetAddon implements CarpetExtension {
    public static final String MOD_ID = "certaterfied-carpet-addon";
    public static final Xoroshiro128PlusPlusRandom fastRandom = new Xoroshiro128PlusPlusRandom(0L);

    public static void init() {}
    static {
        CarpetServer.manageExtension(new CertaterfiedCarpetAddon());
    }

    @Override
    public void onGameStarted() {
        CarpetServer.settingsManager.parseSettingsClass(Rules.class);
        System.out.println("CUDA: " + Rules.optimizedRandomCUDA);
        // TODO Fix this check
        if (Rules.optimizedRandomCUDA) {
        CUDARandom.initialize();
        ServerLifecycleEvents.SERVER_STOPPING.register(server -> CUDARandom.shutdown());
        }
    }

    @Override
    public void onServerLoadedWorlds(MinecraftServer server) {
        fastRandom.setSeed(server.getOverworld().getSeed());
    }

    @Override
    public Map<String, String> canHasTranslations(String lang) {
        InputStream langFile = CertaterfiedCarpetAddon.class.getClassLoader().getResourceAsStream("assets/certaterfied-carpet-addon/lang/%s.json".formatted(lang));
        if (langFile == null) {
            return Collections.emptyMap();
        }
        String jsonData;
        try {
            jsonData = IOUtils.toString(langFile, StandardCharsets.UTF_8);
        } catch (IOException e) {
            return Collections.emptyMap();
        }
        return new GsonBuilder().create().fromJson(jsonData, new TypeToken<Map<String, String>>() {}.getType());
    }
}
