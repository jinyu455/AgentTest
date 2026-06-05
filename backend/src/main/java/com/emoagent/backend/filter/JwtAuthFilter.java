package com.emoagent.backend.filter;

import com.emoagent.backend.service.AuthService;
import io.jsonwebtoken.Claims;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.HandlerInterceptor;

@Component
public class JwtAuthFilter implements HandlerInterceptor {

    public static final String ATTR_USER_ID = "authenticatedUserId";
    public static final String ATTR_USERNAME = "authenticatedUsername";
    public static final String ATTR_ROLE = "authenticatedRole";

    private final AuthService authService;

    public JwtAuthFilter(AuthService authService) {
        this.authService = authService;
    }

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler)
            throws Exception {
        // 跳过 OPTIONS 预检请求
        if ("OPTIONS".equalsIgnoreCase(request.getMethod())) {
            return true;
        }

        String authHeader = request.getHeader("Authorization");
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            response.setContentType("application/json;charset=UTF-8");
            response.getWriter().write("{\"error\":\"未提供认证令牌\"}");
            return false;
        }

        String token = authHeader.substring(7);
        try {
            Claims claims = authService.validateToken(token);
            request.setAttribute(ATTR_USER_ID, claims.getSubject());
            request.setAttribute(ATTR_USERNAME, claims.get("username", String.class));
            request.setAttribute(ATTR_ROLE, claims.get("role", String.class));
            return true;
        } catch (IllegalArgumentException e) {
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            response.setContentType("application/json;charset=UTF-8");
            response.getWriter().write("{\"error\":\"" + e.getMessage() + "\"}");
            return false;
        }
    }
}