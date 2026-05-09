package com.emoagent.backend.controller;

import com.emoagent.backend.dto.AnalyzeResponse;
import com.emoagent.backend.dto.TextAnalyzeRequest;
import com.emoagent.backend.service.EmotionAnalysisService;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/api/emotion")
public class EmotionAnalysisController {
    private final EmotionAnalysisService emotionAnalysisService;

    public EmotionAnalysisController(EmotionAnalysisService emotionAnalysisService) {
        this.emotionAnalysisService = emotionAnalysisService;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        return emotionAnalysisService.health();
    }

    @PostMapping("/analyze")
    public AnalyzeResponse analyze(@Valid @RequestBody TextAnalyzeRequest request) {
        return emotionAnalysisService.analyze(request);
    }
}
