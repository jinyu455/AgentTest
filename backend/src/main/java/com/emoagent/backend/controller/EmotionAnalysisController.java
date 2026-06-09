package com.emoagent.backend.controller;

import com.emoagent.backend.dto.AnalyzeResponse;
import com.emoagent.backend.dto.ChatRequest;
import com.emoagent.backend.dto.ChatResponse;
import com.emoagent.backend.dto.TextAnalyzeRequest;
import com.emoagent.backend.filter.JwtAuthFilter;
import com.emoagent.backend.service.ChatPersistenceService;
import com.emoagent.backend.service.EmotionAnalysisService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/emotion")
public class EmotionAnalysisController {
    private final EmotionAnalysisService emotionAnalysisService;
    private final ChatPersistenceService chatPersistenceService;

    public EmotionAnalysisController(
            EmotionAnalysisService emotionAnalysisService,
            ChatPersistenceService chatPersistenceService) {
        this.emotionAnalysisService = emotionAnalysisService;
        this.chatPersistenceService = chatPersistenceService;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        return emotionAnalysisService.health();
    }

    // 全套情绪分析agent
    @PostMapping("/analyze")
    public AnalyzeResponse analyze(@Valid @RequestBody TextAnalyzeRequest request, HttpServletRequest httpRequest) {
        String userId = (String) httpRequest.getAttribute(JwtAuthFilter.ATTR_USER_ID);
        TextAnalyzeRequest enriched = new TextAnalyzeRequest(
                request.id(), userId, request.text(), request.source(), request.createdAt(), request.metadata());
        AnalyzeResponse response = emotionAnalysisService.analyze(enriched);
        chatPersistenceService.saveStandaloneEmotionRecord(userId, enriched.text(), response);
        return response;
    }

    // analyse+history+问题+meta
    @PostMapping("/chat")
    public ChatResponse chat(@Valid @RequestBody ChatRequest request, HttpServletRequest httpRequest) {
        String userId = (String) httpRequest.getAttribute(JwtAuthFilter.ATTR_USER_ID);
        ChatRequest enriched = new ChatRequest(
                request.text(), userId, request.conversationId(),
                request.judgeResult(), request.history(), request.metadata());
        return emotionAnalysisService.chat(enriched);
    }

    // admin 可选 target_user_id：不传看全部，传了看指定用户
    // user 只能看自己的
    // 查看所有对话
    @GetMapping("/conversations")
    public List<Map<String, Object>> conversations(
            @RequestParam(value = "target_user_id", required = false) String targetUserId,
            HttpServletRequest httpRequest) {
        String userId = (String) httpRequest.getAttribute(JwtAuthFilter.ATTR_USER_ID);
        String role = (String) httpRequest.getAttribute(JwtAuthFilter.ATTR_ROLE);

        if ("admin".equals(role)) {
            // admin：指定了用户就查该用户，没指定就查全部
            if (targetUserId != null && !targetUserId.isBlank()) {
                return chatPersistenceService.conversationsForUser(targetUserId);
            }
            return chatPersistenceService.allConversations();
        }
        // user 只能看自己的
        return chatPersistenceService.conversationsForUser(userId);
    }

    // 查看某一个对话下所有聊天记录
    @GetMapping("/conversations/{conversationId}/messages")
    public List<Map<String, Object>> messages(
            @PathVariable String conversationId,
            HttpServletRequest httpRequest) {
        String userId = (String) httpRequest.getAttribute(JwtAuthFilter.ATTR_USER_ID);
        return chatPersistenceService.messagesForConversation(conversationId, userId);
    }

    // user 看自己的统计，admin 可选 target_user_id
    // 查看情绪统计
    @PostMapping("/profile")
    public Map<String, Object> profile(
            @RequestParam(value = "target_user_id", required = false) String targetUserId,
            HttpServletRequest httpRequest) {
        String userId = (String) httpRequest.getAttribute(JwtAuthFilter.ATTR_USER_ID);
        String role = (String) httpRequest.getAttribute(JwtAuthFilter.ATTR_ROLE);

        String effectiveUserId = userId;
        // admin：指定了用户就查该用户，没指定就查全部
        if ("admin".equals(role)) {
            effectiveUserId = targetUserId != null && !targetUserId.isBlank() ? targetUserId : null;
        }
        return chatPersistenceService.profile(effectiveUserId);
    }

    // user 生成自己的画像；admin 可选 target_user_id
    // 查看情绪画像
    @PostMapping("/profile/generate")
    public Map<String, Object> profileGenerate(
            @RequestParam(value = "target_user_id", required = false) String targetUserId,
            @RequestParam(value = "force", defaultValue = "false") boolean force,
            HttpServletRequest httpRequest) {
        String role = (String) httpRequest.getAttribute(JwtAuthFilter.ATTR_ROLE);
        String userId = (String) httpRequest.getAttribute(JwtAuthFilter.ATTR_USER_ID);
        String effectiveUserId = userId;
        // admin：指定了用户就查该用户，没指定就查全部
        if ("admin".equals(role)) {
            effectiveUserId = targetUserId != null && !targetUserId.isBlank() ? targetUserId : null;
        }
        return chatPersistenceService.profileGenerate(effectiveUserId, force);
    }
}
