package com.emoagent.backend.controller;

import com.emoagent.backend.dto.AuthRequest;
import com.emoagent.backend.dto.AuthResponse;
import com.emoagent.backend.dto.CaptchaResponse;
import com.emoagent.backend.dto.RegisterRequest;
import com.emoagent.backend.service.AuthService;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final AuthService authService;

    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/register")
    public AuthResponse register(@Valid @RequestBody RegisterRequest request) {
        return authService.register(request);
    }

    @PostMapping("/login")
    public AuthResponse login(@Valid @RequestBody AuthRequest request) {
        return authService.login(request.username(), request.password());
    }

    @GetMapping("/captcha")
    public CaptchaResponse captcha() {
        return authService.generateCaptcha();
    }
}
