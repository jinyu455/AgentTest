package com.emoagent.backend.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;

@Configuration
public class JwtConfig {

    @Value("${jwt.secret:emoagent-jwt-secret-key-2024-must-be-at-least-32-bytes}")
    private String secret;

    @Value("${jwt.expiration:86400000}")
    private long expiration;

    public String getSecret() {
        return secret;
    }

    public long getExpiration() {
        return expiration;
    }
}
