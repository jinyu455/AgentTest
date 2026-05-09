package com.emoagent.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Map;

public record JudgeRequest(
        String text,

        @JsonProperty("router_result")
        Map<String, Object> routerResult,

        @JsonProperty("emotion_result")
        Map<String, Object> emotionResult,

        @JsonProperty("sarcasm_result")
        Map<String, Object> sarcasmResult,

        @JsonProperty("mix_result")
        Map<String, Object> mixResult
) {
}
