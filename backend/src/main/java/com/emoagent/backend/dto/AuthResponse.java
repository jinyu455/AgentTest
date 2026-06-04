package com.emoagent.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record AuthResponse(
        String token,

        @JsonProperty("user_id")
        String userId,

        String username,

        String role
) {
}
