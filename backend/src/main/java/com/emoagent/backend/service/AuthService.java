package com.emoagent.backend.service;

import com.emoagent.backend.config.JwtConfig;
import com.emoagent.backend.dto.AuthResponse;
import com.emoagent.backend.dto.CaptchaResponse;
import com.emoagent.backend.dto.RegisterRequest;
import com.emoagent.backend.entity.User;
import com.emoagent.backend.repository.UserRepository;
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.ExpiredJwtException;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.Base64;
import java.util.Date;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class AuthService {

    private final UserRepository userRepository;
    private final JwtConfig jwtConfig;
    private final SecretKey signingKey;

    @Value("${admin.username:admin}")
    private String adminUsername;

    @Value("${admin.password:admin123}")
    private String adminPassword;

    // 验证码存储：key -> {code, expireAt}
    private final Map<String, CaptchaEntry> captchaStore = new ConcurrentHashMap<>();
    private static final long CAPTCHA_TTL_MS = 5 * 60 * 1000; // 5 分钟自动过期的验证码缓存

    public AuthService(UserRepository userRepository, JwtConfig jwtConfig) {
        this.userRepository = userRepository;
        this.jwtConfig = jwtConfig;
        this.signingKey = Keys.hmacShaKeyFor(jwtConfig.getSecret().getBytes(StandardCharsets.UTF_8));
    }

    // 第一次启动检查有没有admin没有自动创建
    @PostConstruct
    public void initAdminUser() {
        if (!userRepository.existsByUsername(adminUsername)) {
            String salt = UUID.randomUUID().toString().replace("-", "");
            String hash = hashPassword(adminPassword, salt);
            String adminId = UUID.randomUUID().toString();
            User admin = new User(adminId, adminUsername, hash, salt, "admin", Instant.now());
            userRepository.save(admin);
            System.out.println("[AuthService] 默认管理员账号已创建: " + adminUsername);
        }
    }

    // 注册成功不返回token，前端跳转到登录页
    public AuthResponse register(RegisterRequest request) {
        // 1. 校验验证码
        validateCaptcha(request.captchaKey(), request.captchaCode());

        // 2. 检查用户名唯一
        if (userRepository.existsByUsername(request.username())) {
            throw new IllegalArgumentException("用户名已存在");
        }

        // 3. 生成盐值和密码哈希
        String salt = UUID.randomUUID().toString().replace("-", "");
        String passwordHash = hashPassword(request.password(), salt);

        // 4. 创建用户（注册只能创建 user 角色）
        String userId = UUID.randomUUID().toString();
        User user = new User(userId, request.username(), passwordHash, salt, "user", Instant.now());
        userRepository.save(user);

        // 5. 不返回token，返回成功消息，前端跳转登录页
        return new AuthResponse(null, null, null, null, "注册成功，请登录");
    }

    // 登录，根据autoLogin决定token有效期
    // 不勾选自动登录：5分钟，勾选：7天
    public AuthResponse login(String username, String password, Boolean autoLogin) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new IllegalArgumentException("用户名或密码错误"));

        String hash = hashPassword(password, user.getSalt());
        if (!hash.equals(user.getPasswordHash())) {
            throw new IllegalArgumentException("用户名或密码错误");
        }

        // 勾选自动登录7天，否则5分钟
        long duration = Boolean.TRUE.equals(autoLogin)
                ? 7 * 24 * 60 * 60 * 1000L   // 7天
                : 5 * 60 * 1000L;             // 5分钟

        String token = generateToken(user.getId(), user.getUsername(), user.getRole(), duration);
        return new AuthResponse(token, user.getId(), user.getUsername(), user.getRole(), null);
    }

    // 生成验证码，4位验证码+唯一key+svg图片
    public CaptchaResponse generateCaptcha() {
        String code = generateRandomCode();
        String key = UUID.randomUUID().toString();
        captchaStore.put(key, new CaptchaEntry(code, Instant.now().plusMillis(CAPTCHA_TTL_MS)));
        String imageBase64 = generateCaptchaImage(code);
        return new CaptchaResponse(key, imageBase64);
    }

    private void validateCaptcha(String key, String code) {
        CaptchaEntry entry = captchaStore.remove(key);
        if (entry == null) {
            throw new IllegalArgumentException("验证码已过期或不存在");
        }
        if (Instant.now().isAfter(entry.expireAt())) {
            throw new IllegalArgumentException("验证码已过期");
        }
        if (!entry.code.equalsIgnoreCase(code.trim())) {
            throw new IllegalArgumentException("验证码错误");
        }
    }

    private String generateRandomCode() {
        // 生成 4 位随机验证码（数字+大写字母，排除易混淆字符）
        String chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 4; i++) {
            int idx = (int) (Math.random() * chars.length());
            sb.append(chars.charAt(idx));
        }
        return sb.toString();
    }

    private String generateCaptchaImage(String code) {
        // 用字符串拼凑生成简单的验证码 SVG 图片（Base64 编码）
        int width = 120;
        int height = 40;
        StringBuilder svg = new StringBuilder();
        svg.append("<svg xmlns='http://www.w3.org/2000/svg' width='").append(width)
                .append("' height='").append(height).append("'>");
        svg.append("<rect width='100%' height='100%' fill='#f0f0f0'/>");

        // 绘制4条随机干扰线
        for (int i = 0; i < 4; i++) {
            int x1 = (int) (Math.random() * width);
            int y1 = (int) (Math.random() * height);
            int x2 = (int) (Math.random() * width);
            int y2 = (int) (Math.random() * height);
            svg.append("<line x1='").append(x1).append("' y1='").append(y1)
                    .append("' x2='").append(x2).append("' y2='").append(y2)
                    .append("' stroke='#ccc' stroke-width='1'/>");
        }

        // 绘制验证码文字，随机位置，随机颜色，随机旋转（-10° ~ 10°），字体加粗
        for (int i = 0; i < code.length(); i++) {
            int x = 10 + i * 25;
            int y = 25 + (int) (Math.random() * 10 - 5);
            int rotate = (int) (Math.random() * 20 - 10);
            String[] colors = { "#333", "#666", "#999", "#c00", "#060", "#009" };
            String color = colors[(int) (Math.random() * colors.length)];
            svg.append("<text x='").append(x).append("' y='").append(y)
                    .append("' font-size='22' font-weight='bold' fill='").append(color)
                    .append("' transform='rotate(").append(rotate).append(" ").append(x).append(" ").append(y)
                    .append(")'")
                    .append(">").append(code.charAt(i)).append("</text>");
        }

        svg.append("</svg>");
        // 转成base64,编码一串文本
        return "data:image/svg+xml;base64,"
                + Base64.getEncoder().encodeToString(svg.toString().getBytes(StandardCharsets.UTF_8));
    }

    // 生成jwt，duration决定过期时间
    public String generateToken(String userId, String username, String role, long durationMs) {
        Instant now = Instant.now();
        return Jwts.builder()
                .subject(userId)
                .claim("username", username)
                .claim("role", role)
                .issuedAt(Date.from(now))
                .expiration(Date.from(now.plusMillis(durationMs)))
                .signWith(signingKey)
                .compact();
    }

    // 验证签名是否正确，验证是否过期，解析出用户信息
    public Claims validateToken(String token) {
        try {
            return Jwts.parser()
                    .verifyWith(signingKey)
                    .build()
                    .parseSignedClaims(token)
                    .getPayload();
        } catch (ExpiredJwtException e) {
            throw new IllegalArgumentException("Token 已过期");
        } catch (Exception e) {
            throw new IllegalArgumentException("Token 无效");
        }
    }

    private String hashPassword(String password, String salt) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            String salted = salt + password;
            byte[] hash = digest.digest(salted.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (byte b : hash) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("SHA-256 不可用", e);
        }
    }

    private record CaptchaEntry(String code, Instant expireAt) {
    }
}
