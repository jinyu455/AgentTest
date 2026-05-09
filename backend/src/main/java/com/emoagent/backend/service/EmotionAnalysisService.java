package com.emoagent.backend.service;

import com.emoagent.backend.client.AgentClient;
import com.emoagent.backend.dto.AnalyzeResponse;
import com.emoagent.backend.dto.JudgeRequest;
import com.emoagent.backend.dto.TextAnalyzeRequest;
import org.springframework.stereotype.Service;

import java.util.Map;

@Service
public class EmotionAnalysisService {
    private final AgentClient agentClient;

    public EmotionAnalysisService(AgentClient agentClient) {
        this.agentClient = agentClient;
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

    public Map<String, Object> health() {
        return agentClient.health();
    }

    private boolean isTrue(Object value) {
        return value instanceof Boolean bool && bool;
    }
}
