package com.emoagent.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;

public record RegisterRequest(
        @NotBlank
        String username,

        @NotBlank
        String password,

        @JsonProperty("captcha_code")
        @NotBlank
        String captchaCode,

        @JsonProperty("captcha_key")
        @NotBlank
        String captchaKey
) {
}
