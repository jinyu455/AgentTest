package com.emoagent.backend.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonInclude(JsonInclude.Include.NON_NULL)
public record AuthResponse(
        String token,

        @JsonProperty("user_id")
        String userId,

        String username,

        String role,

        String message
) {
}
