package com.emoagent.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

public record ChatRequest(
        String text,

        @JsonProperty("user_id")
        String userId,

        @JsonProperty("conversation_id")
        String conversationId,

        @JsonProperty("judge_result")
        Map<String, Object> judgeResult,

        List<Map<String, Object>> history,

        Map<String, Object> metadata
) {
}
