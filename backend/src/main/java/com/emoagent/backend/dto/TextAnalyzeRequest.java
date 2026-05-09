package com.emoagent.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;

import java.util.Map;

public record TextAnalyzeRequest(
        @NotBlank
        String id,

        @JsonProperty("user_id")
        @NotBlank
        String userId,

        @NotBlank
        String text,

        @NotBlank
        String source,

        @JsonProperty("created_at")
        @NotBlank
        String createdAt,

        Map<String, Object> metadata
) {
}
