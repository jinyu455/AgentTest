package com.emoagent.backend.controller;

import com.emoagent.backend.client.AgentClient;
import com.emoagent.backend.dto.AnalyzeResponse;
import com.emoagent.backend.dto.ChatRequest;
import com.emoagent.backend.dto.ChatResponse;
import com.emoagent.backend.dto.TextAnalyzeRequest;
import com.emoagent.backend.entity.EmotionRecord;
import com.emoagent.backend.filter.JwtAuthFilter;
import com.emoagent.backend.repository.EmotionRecordRepository;
import com.emoagent.backend.service.ChatPersistenceService;
import com.emoagent.backend.service.EmotionAnalysisService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/emotion")
public class EmotionAnalysisController {
    private final EmotionAnalysisService emotionAnalysisService;
    private final ChatPersistenceService chatPersistenceService;
    private final AgentClient agentClient;
    private final EmotionRecordRepository emotionRecordRepository;

    public EmotionAnalysisController(
            EmotionAnalysisService emotionAnalysisService,
            ChatPersistenceService chatPersistenceService,
            AgentClient agentClient,
            EmotionRecordRepository emotionRecordRepository
    ) {
        this.emotionAnalysisService = emotionAnalysisService;
        this.chatPersistenceService = chatPersistenceService;
        this.agentClient = agentClient;
        this.emotionRecordRepository = emotionRecordRepository;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        return emotionAnalysisService.health();
    }

    @PostMapping("/analyze")
    public AnalyzeResponse analyze(@Valid @RequestBody TextAnalyzeRequest request, HttpServletRequest httpRequest) {
        // 从 JWT 中注入 userId
        String userId = (String) httpRequest.getAttribute(JwtAuthFilter.ATTR_USER_ID);
        TextAnalyzeRequest enriched = new TextAnalyzeRequest(
                request.id(), userId, request.text(), request.source(), request.createdAt(), request.metadata()
        );
        return emotionAnalysisService.analyze(enriched);
    }

    @PostMapping("/chat")
    public ChatResponse chat(@Valid @RequestBody ChatRequest request, HttpServletRequest httpRequest) {
        // 从 JWT 中注入 userId
        String userId = (String) httpRequest.getAttribute(JwtAuthFilter.ATTR_USER_ID);
        ChatRequest enriched = new ChatRequest(
                request.text(), userId, request.conversationId(),
                request.judgeResult(), request.history(), request.metadata()
        );
        return emotionAnalysisService.chat(enriched);
    }

    @GetMapping("/conversations")
    public List<Map<String, Object>> conversations(HttpServletRequest httpRequest) {
        String userId = (String) httpRequest.getAttribute(JwtAuthFilter.ATTR_USER_ID);
        String role = (String) httpRequest.getAttribute(JwtAuthFilter.ATTR_ROLE);
        // admin 可以看所有用户的对话，user 只能看自己的
        if ("admin".equals(role)) {
            // admin 暂时也只看自己的，后续可扩展为查全部
            return chatPersistenceService.conversationsForUser(userId);
        }
        return chatPersistenceService.conversationsForUser(userId);
    }

    @GetMapping("/conversations/{conversationId}/messages")
    public List<Map<String, Object>> messages(
            @PathVariable String conversationId,
            HttpServletRequest httpRequest
    ) {
        String userId = (String) httpRequest.getAttribute(JwtAuthFilter.ATTR_USER_ID);
        return chatPersistenceService.messagesForConversation(conversationId, userId);
    }

    // ============================================================
    // Profile 接口
    // ============================================================

    @PostMapping("/profile")
    public Map<String, Object> profile(HttpServletRequest httpRequest) {
        String userId = (String) httpRequest.getAttribute(JwtAuthFilter.ATTR_USER_ID);
        return buildProfilePayload(userId);
    }

    @PostMapping("/profile/generate")
    public Map<String, Object> profileGenerate(HttpServletRequest httpRequest) {
        String role = (String) httpRequest.getAttribute(JwtAuthFilter.ATTR_ROLE);
        if (!"admin".equals(role)) {
            throw new SecurityException("仅管理员可访问用户画像接口");
        }
        String userId = (String) httpRequest.getAttribute(JwtAuthFilter.ATTR_USER_ID);
        Map<String, Object> payload = buildProfilePayload(userId);
        return agentClient.profileGenerate(payload);
    }

    /**
     * 查询指定用户（或所有用户）的情绪记录，构建 profile 请求载荷。
     * admin 传 null 时查询所有用户。
     */
    private Map<String, Object> buildProfilePayload(String userId) {
        List<EmotionRecord> records = emotionRecordRepository.findByUserId(userId);
        List<Map<String, Object>> emotionRecords = records.stream().map(er -> {
            Map<String, Object> m = new HashMap<>();
            m.put("id", er.getId());
            m.put("conversation_id", er.getConversationId());
            m.put("message_id", er.getMessageId());
            m.put("final_emotion", er.getFinalEmotion());
            m.put("secondary_emotion", er.getSecondaryEmotion());
            m.put("final_intensity", er.getFinalIntensity());
            m.put("final_confidence", er.getFinalConfidence());
            m.put("is_sarcasm", er.getSarcasm());
            m.put("is_mixed", er.getMixed());
            m.put("raw_analysis_json", er.getRawAnalysisJson());
            m.put("created_at", er.getCreatedAt() != null ? er.getCreatedAt().toString() : "");
            return m;
        }).toList();

        Map<String, Object> payload = new HashMap<>();
        payload.put("user_id", userId);
        payload.put("emotion_records", emotionRecords);
        return agentClient.profile(payload);
    }
}
