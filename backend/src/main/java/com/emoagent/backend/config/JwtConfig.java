package com.emoagent.backend.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;

@Configuration
public class JwtConfig {

    @Value("${jwt.secret:jwt-secret-emoagent-7f9e2d5c8b1a0s3k6m9n2l5p8r0t}")
    private String secret;

    @Value("${jwt.expiration:86400000}") // 过期时间1天
    private long expiration;

    public String getSecret() {
        return secret;
    }

    public long getExpiration() {
        return expiration;
    }
}
