package com.emoagent.backend.client;

import com.emoagent.backend.dto.ChatRequest;
import com.emoagent.backend.dto.TextAnalyzeRequest;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.util.Map;

@Component
public class AgentClient {
    private static final ParameterizedTypeReference<Map<String, Object>> MAP_RESPONSE =
            new ParameterizedTypeReference<>() {
            };

    private final RestClient restClient;

    public AgentClient(
            RestClient.Builder builder,
            @Value("${emo-agent.base-url}") String baseUrl
    ) {
        this.restClient = builder.baseUrl(baseUrl).build();
    }

    public Map<String, Object> health() {
        return restClient.get()
                .uri("/health")
                .retrieve()
                .body(MAP_RESPONSE);
    }

    public Map<String, Object> analyze(TextAnalyzeRequest request) {
        return post("/analyze", request);
    }

    public Map<String, Object> chat(ChatRequest request) {
        return post("/chat", request);
    }

    private Map<String, Object> post(String uri, Object request) {
        return restClient.post()
                .uri(uri)
                .body(request)
                .retrieve()
                .body(MAP_RESPONSE);
    }
}
