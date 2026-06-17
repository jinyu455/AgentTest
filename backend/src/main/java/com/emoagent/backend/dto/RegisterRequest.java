package com.emoagent.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

//前端注册必须传的参数，注册时才需要验证码
public record RegisterRequest(
                @NotBlank @Size(max = 20) String username,

                @NotBlank String password,

                @JsonProperty("captcha_code") @NotBlank String captchaCode,

                @JsonProperty("captcha_key") @NotBlank String captchaKey) {

}