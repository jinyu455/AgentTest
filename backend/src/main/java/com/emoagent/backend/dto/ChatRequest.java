package com.emoagent.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;

import java.util.List;
import java.util.Map;

public record ChatRequest(
        @NotBlank
        String text,

        @JsonProperty("user_id")
        @NotBlank
        String userId,

        @JsonProperty("conversation_id")
        String conversationId,

        @JsonProperty("judge_result")
        Map<String, Object> judgeResult,

        List<Map<String, Object>> history,

        Map<String, Object> metadata
) {
}
