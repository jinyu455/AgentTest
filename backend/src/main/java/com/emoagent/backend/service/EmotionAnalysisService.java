package com.emoagent.backend.service;

import com.emoagent.backend.client.AgentClient;
import com.emoagent.backend.dto.AnalyzeResponse;
import com.emoagent.backend.dto.ChatRequest;
import com.emoagent.backend.dto.ChatResponse;
import com.emoagent.backend.dto.JudgeRequest;
import com.emoagent.backend.dto.TextAnalyzeRequest;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

@Service
public class EmotionAnalysisService {
    private final AgentClient agentClient;
    private final ChatPersistenceService chatPersistenceService;

    public EmotionAnalysisService(AgentClient agentClient, ChatPersistenceService chatPersistenceService) {
        this.agentClient = agentClient;
        this.chatPersistenceService = chatPersistenceService;
    }

    public AnalyzeResponse analyze(TextAnalyzeRequest request) {
        Map<String, Object> routerResult = agentClient.router(request);
        Map<String, Object> emotionResult = agentClient.emotion(request);

        Map<String, Object> sarcasmResult = null;
        if (isTrue(routerResult.get("need_sarcasm_check"))) {
            sarcasmResult = agentClient.sarcasm(request);
        }

        Map<String, Object> mixResult = null;
        if (isTrue(routerResult.get("need_mix_check"))) {
            mixResult = agentClient.mix(request);
        }

        JudgeRequest judgeRequest = new JudgeRequest(
                request.text(),
                routerResult,
                emotionResult,
                sarcasmResult,
                mixResult
        );
        Map<String, Object> judgeResult = agentClient.judge(judgeRequest);

        return new AnalyzeResponse(
                request.text(),
                routerResult,
                emotionResult,
                sarcasmResult,
                mixResult,
                judgeResult
        );
    }

    public ChatResponse chat(ChatRequest request) {
        ChatPersistenceService.ChatTurn chatTurn = chatPersistenceService.startTurn(request);
        AnalyzeResponse analysisResult = analyze(toAnalyzeRequest(request));
        chatPersistenceService.saveEmotionRecord(chatTurn, analysisResult);
        List<Map<String, Object>> history = mergeHistory(
                chatPersistenceService.historyBeforeTurn(chatTurn),
                request.history(),
                request.text()
        );
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
                Instant.now().toString(),
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

    private boolean isTrue(Object value) {
        return value instanceof Boolean bool && bool;
    }

    private List<Map<String, Object>> mergeHistory(
            List<Map<String, Object>> persistedHistory,
            List<Map<String, Object>> requestHistory,
            String currentText
    ) {
        List<Map<String, Object>> merged = new ArrayList<>();
        Set<String> seen = new LinkedHashSet<>();

        addHistoryItems(merged, seen, persistedHistory, currentText);
        addHistoryItems(merged, seen, requestHistory, currentText);

        int fromIndex = Math.max(0, merged.size() - 20);
        return List.copyOf(merged.subList(fromIndex, merged.size()));
    }

    private void addHistoryItems(
            List<Map<String, Object>> merged,
            Set<String> seen,
            List<Map<String, Object>> source,
            String currentText
    ) {
        if (source == null) {
            return;
        }

        for (Map<String, Object> item : source) {
            if (item == null) {
                continue;
            }
            String role = stringValue(item.get("role"));
            String content = stringValue(item.get("content"));
            if (!isHistoryRole(role) || content == null || content.isBlank()) {
                continue;
            }
            if ("user".equals(role) && currentText != null && content.strip().equals(currentText.strip())) {
                continue;
            }

            String key = role + "\n" + content;
            if (seen.add(key)) {
                Map<String, Object> normalized = new LinkedHashMap<>();
                normalized.put("role", role);
                normalized.put("content", content);
                merged.add(normalized);
            }
        }
    }

    private boolean isHistoryRole(String role) {
        return "user".equals(role) || "assistant".equals(role);
    }

    private String stringValue(Object value) {
        return value == null ? null : value.toString();
    }
}
