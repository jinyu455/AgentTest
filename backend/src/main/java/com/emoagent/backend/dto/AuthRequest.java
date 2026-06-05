package com.emoagent.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;

public record AuthRequest(
        @NotBlank String username,

        @NotBlank String password,

        @JsonProperty("auto_login")
        Boolean autoLogin
) {
}
