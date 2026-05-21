package com.emoagent.backend.service;

import com.emoagent.backend.client.AgentClient;
import com.emoagent.backend.dto.AnalyzeResponse;
import com.emoagent.backend.dto.ChatRequest;
import com.emoagent.backend.dto.ChatResponse;
import com.emoagent.backend.dto.TextAnalyzeRequest;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
public class EmotionAnalysisService {
    private static final ZoneId UTC = ZoneId.of("UTC");
    private static final DateTimeFormatter MYSQL_DATETIME = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private final AgentClient agentClient;
    private final ChatPersistenceService chatPersistenceService;

    public EmotionAnalysisService(AgentClient agentClient, ChatPersistenceService chatPersistenceService) {
        this.agentClient = agentClient;
        this.chatPersistenceService = chatPersistenceService;
    }

    public AnalyzeResponse analyze(TextAnalyzeRequest request) {
        return toAnalyzeResponse(agentClient.analyze(request), request.text());
    }

    public ChatResponse chat(ChatRequest request) {
        ChatPersistenceService.ChatTurn chatTurn = chatPersistenceService.startTurn(request);
        AnalyzeResponse analysisResult = analyze(toAnalyzeRequest(request));
        chatPersistenceService.saveEmotionRecord(chatTurn, analysisResult);
        List<Map<String, Object>> history = chatPersistenceService.historyBeforeTurn(chatTurn);
        ChatRequest agentRequest = new ChatRequest(
                request.text(),
                request.userId(),
                chatTurn.conversationId(),
                analysisResult.judgeResult(),
                history,
                metadata(request)
        );
        Map<String, Object> chatResult = agentClient.chat(agentRequest);
        chatPersistenceService.saveAssistantMessage(chatTurn.conversationId(), chatResult);

        return new ChatResponse(chatTurn.conversationId(), request.text(), analysisResult, chatResult);
    }

    public Map<String, Object> health() {
        return agentClient.health();
    }

    private TextAnalyzeRequest toAnalyzeRequest(ChatRequest request) {
        return new TextAnalyzeRequest(
                UUID.randomUUID().toString(),
                request.userId(),
                request.text(),
                "chat",
                LocalDateTime.now(UTC).format(MYSQL_DATETIME),
                metadata(request)
        );
    }

    private Map<String, Object> metadata(ChatRequest request) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        if (request.metadata() != null) {
            metadata.putAll(request.metadata());
        }
        return metadata;
    }

    private AnalyzeResponse toAnalyzeResponse(Map<String, Object> payload, String fallbackText) {
        if (payload == null) {
            return new AnalyzeResponse(fallbackText, null, null, null, null, null);
        }

        if (payload.get("judge_result") instanceof Map<?, ?>) {
            return new AnalyzeResponse(
                    stringValue(payload.get("text"), fallbackText),
                    mapValue(payload.get("router_result")),
                    mapValue(payload.get("emotion_result")),
                    mapValue(payload.get("sarcasm_result")),
                    mapValue(payload.get("mix_result")),
                    mapValue(payload.get("judge_result"))
            );
        }

        return new AnalyzeResponse(
                stringValue(payload.get("text"), fallbackText),
                buildRouterResult(payload),
                buildEmotionResult(payload),
                null,
                null,
                buildJudgeResult(payload)
        );
    }

    private Map<String, Object> buildRouterResult(Map<String, Object> payload) {
        String sampleType = stringValue(payload.get("sample_type"), null);
        if (sampleType == null) {
            return null;
        }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("sample_type", sampleType);
        result.put("need_sarcasm_check", "sarcasm_suspected".equals(sampleType));
        result.put("need_mix_check", "mix".equals(sampleType));
        result.put("routing_reason", payload.get("reason"));
        return result;
    }

    private Map<String, Object> buildEmotionResult(Map<String, Object> payload) {
        if (payload.get("emotion") == null && payload.get("tokens") == null && payload.get("emotion_words") == null) {
            return null;
        }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("emotion", payload.get("emotion"));
        result.put("intensity", payload.get("intensity"));
        result.put("confidence", payload.get("final_confidence"));
        result.put("tokens", payload.get("tokens"));
        result.put("emotion_words", payload.get("emotion_words"));
        result.put("reason", payload.get("reason"));
        return result;
    }

    private Map<String, Object> buildJudgeResult(Map<String, Object> payload) {
        if (payload.get("emotion") == null && payload.get("final_confidence") == null) {
            return null;
        }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("final_emotion", payload.get("emotion"));
        result.put("secondary_emotion", payload.get("secondary_emotion"));
        result.put("final_intensity", payload.get("intensity"));
        result.put("final_confidence", payload.get("final_confidence"));
        result.put("is_sarcasm", payload.get("is_sarcasm"));
        result.put("is_mixed", payload.get("is_mixed"));
        result.put("reason", payload.get("reason"));
        return result;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> mapValue(Object value) {
        return value instanceof Map<?, ?> map ? (Map<String, Object>) map : null;
    }

    private String stringValue(Object value, String fallback) {
        return value == null ? fallback : value.toString();
    }
}
