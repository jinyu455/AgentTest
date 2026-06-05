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

    // router->emotion->maybe(sarcasm/mix)->judge
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
                mixResult);
        Map<String, Object> judgeResult = agentClient.judge(judgeRequest);

        return new AnalyzeResponse(
                request.text(),
                routerResult,
                emotionResult,
                sarcasmResult,
                mixResult,
                judgeResult);
    }

    public ChatResponse chat(ChatRequest request) {
        ChatPersistenceService.ChatTurn chatTurn = chatPersistenceService.startTurn(request);
        // 调用analyse方法获取judge结果(情绪结果)
        AnalyzeResponse analysisResult = analyze(toAnalyzeRequest(request));
        // 保存情绪分析结果到数据库
        chatPersistenceService.saveEmotionRecord(chatTurn, analysisResult);
        // 合并前端用户问题和history
        List<Map<String, Object>> history = mergeHistory(
                chatPersistenceService.historyBeforeTurn(chatTurn),
                request.history(),
                request.text());
        // 聊天带上历史记录(history)，即带上情绪分析结果和之前的对话
        ChatRequest agentRequest = new ChatRequest(
                request.text(),
                request.userId(),
                chatTurn.conversationId(),
                analysisResult.judgeResult(),
                history,
                metadata(request));
        Map<String, Object> chatResult = agentClient.chat(agentRequest);
        // 保存ai回答结果到数据库
        chatPersistenceService.saveAssistantMessage(chatTurn.conversationId(), chatResult);
        // histroy中有user assitant的完整对话(20条)
        return new ChatResponse(chatTurn.conversationId(), request.text(), analysisResult, chatResult);
    }

    public Map<String, Object> health() {
        return agentClient.health();
    }

    // 将chat转换为input的格式
    private TextAnalyzeRequest toAnalyzeRequest(ChatRequest request) {
        return new TextAnalyzeRequest(
                UUID.randomUUID().toString(),
                request.userId(),
                request.text(),
                "chat",
                Instant.now().toString(),
                metadata(request));
    }

    // 前端额外信息，设备、渠道、扩展字段
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

    // 这里是将前端可能存在的没有写入数据库的信息和db里面的信息合并
    private List<Map<String, Object>> mergeHistory(
            List<Map<String, Object>> persistedHistory, // 数据库存的历史
            List<Map<String, Object>> requestHistory, // 前端带的历史
            String currentText) { // 当前用户说的话
        List<Map<String, Object>> merged = new ArrayList<>();
        Set<String> seen = new LinkedHashSet<>();

        addHistoryItems(merged, seen, persistedHistory, currentText);
        addHistoryItems(merged, seen, requestHistory, currentText);

        int fromIndex = Math.max(0, merged.size() - 20);
        return List.copyOf(merged.subList(fromIndex, merged.size()));
    }

    // 保留user assistant的content
    private void addHistoryItems(
            List<Map<String, Object>> merged,
            Set<String> seen,
            List<Map<String, Object>> source,
            String currentText) {
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
            // 忽略刚刚发送的信息防止重复
            if ("user".equals(role) && currentText != null && content.strip().equals(currentText.strip())) {
                continue;
            }
            // 去重
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
