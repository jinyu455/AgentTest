package com.emoagent.backend.client;

import com.emoagent.backend.dto.JudgeRequest;
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

    public Map<String, Object> router(TextAnalyzeRequest request) {
        return postText("/router", request);
    }

    public Map<String, Object> emotion(TextAnalyzeRequest request) {
        return postText("/emotion", request);
    }

    public Map<String, Object> sarcasm(TextAnalyzeRequest request) {
        return postText("/sarcasm", request);
    }

    public Map<String, Object> mix(TextAnalyzeRequest request) {
        return postText("/mix", request);
    }

    public Map<String, Object> judge(JudgeRequest request) {
        return restClient.post()
                .uri("/judge")
                .body(request)
                .retrieve()
                .body(MAP_RESPONSE);
    }

    private Map<String, Object> postText(String uri, TextAnalyzeRequest request) {
        return restClient.post()
                .uri(uri)
                .body(request)
                .retrieve()
                .body(MAP_RESPONSE);
    }
}
