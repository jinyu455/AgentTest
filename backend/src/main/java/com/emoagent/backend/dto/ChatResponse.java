package com.emoagent.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Map;

public record ChatResponse(
        @JsonProperty("conversation_id")
        String conversationId,

        String text,

        @JsonProperty("analysis_result")
        AnalyzeResponse analysisResult,

        @JsonProperty("chat_result")
        Map<String, Object> chatResult
) {
}
