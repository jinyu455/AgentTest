package com.emoagent.backend.controller;

import com.emoagent.backend.dto.AnalyzeResponse;
import com.emoagent.backend.dto.ChatRequest;
import com.emoagent.backend.dto.ChatResponse;
import com.emoagent.backend.dto.TextAnalyzeRequest;
import com.emoagent.backend.service.ChatPersistenceService;
import com.emoagent.backend.service.EmotionAnalysisService;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
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
            ChatPersistenceService chatPersistenceService
    ) {
        this.emotionAnalysisService = emotionAnalysisService;
        this.chatPersistenceService = chatPersistenceService;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        return emotionAnalysisService.health();
    }

    @PostMapping("/analyze")
    public AnalyzeResponse analyze(@Valid @RequestBody TextAnalyzeRequest request) {
        return emotionAnalysisService.analyze(request);
    }

    @PostMapping("/chat")
    public ChatResponse chat(@Valid @RequestBody ChatRequest request) {
        return emotionAnalysisService.chat(request);
    }

    @GetMapping("/conversations")
    public List<Map<String, Object>> conversations(@RequestParam("user_id") String userId) {
        return chatPersistenceService.conversationsForUser(userId);
    }

    @GetMapping("/conversations/{conversationId}/messages")
    public List<Map<String, Object>> messages(
            @PathVariable String conversationId,
            @RequestParam("user_id") String userId
    ) {
        return chatPersistenceService.messagesForConversation(conversationId, userId);
    }
}
