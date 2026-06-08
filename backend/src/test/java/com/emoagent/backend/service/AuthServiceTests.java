package com.emoagent.backend.service;

import com.emoagent.backend.config.JwtConfig;
import com.emoagent.backend.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.xml.sax.InputSource;

import javax.xml.parsers.DocumentBuilderFactory;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.util.Base64;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class AuthServiceTests {
    @Mock
    private UserRepository userRepository;

    @Test
    void generateCaptchaReturnsValidSvgDataUri() throws Exception {
        JwtConfig jwtConfig = org.mockito.Mockito.mock(JwtConfig.class);
        when(jwtConfig.getSecret()).thenReturn("jwt-secret-emoagent-7f9e2d5c8b1a0s3k6m9n2l5p8r0t");
        AuthService service = new AuthService(userRepository, jwtConfig);

        String image = service.generateCaptcha().captchaImage();
        String prefix = "data:image/svg+xml;base64,";
        assertThat(image).startsWith(prefix);

        String svg = new String(Base64.getDecoder().decode(image.substring(prefix.length())), StandardCharsets.UTF_8);
        assertThat(svg).doesNotContain(")''");

        DocumentBuilderFactory.newInstance()
                .newDocumentBuilder()
                .parse(new InputSource(new StringReader(svg)));
    }
}
