package com.emoagent.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record CaptchaResponse(
        @JsonProperty("captcha_key")
        String captchaKey,

        @JsonProperty("captcha_image")
        String captchaImage
) {
}
