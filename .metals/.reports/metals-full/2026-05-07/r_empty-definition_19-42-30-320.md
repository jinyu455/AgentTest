error id: file:///D:/PracticalTraining/Agenttest/EmoAgent/backend/src/main/java/com/emoagent/backend/client/AgentClient.java:_empty_/ParameterizedTypeReference#
file:///D:/PracticalTraining/Agenttest/EmoAgent/backend/src/main/java/com/emoagent/backend/client/AgentClient.java
empty definition using pc, found symbol in pc: _empty_/ParameterizedTypeReference#
empty definition using semanticdb
empty definition using fallback
non-local guesses:

offset: 447
uri: file:///D:/PracticalTraining/Agenttest/EmoAgent/backend/src/main/java/com/emoagent/backend/client/AgentClient.java
text:
```scala
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
    private static final Parame@@terizedTypeReference<Map<String, Object>> MAP_RESPONSE =
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

```


#### Short summary: 

empty definition using pc, found symbol in pc: _empty_/ParameterizedTypeReference#