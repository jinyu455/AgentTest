error id: file:///D:/PracticalTraining/Agenttest/EmoAgent/backend/src/main/java/com/emoagent/backend/controller/EmotionAnalysisController.java:EmotionAnalysisService#
file:///D:/PracticalTraining/Agenttest/EmoAgent/backend/src/main/java/com/emoagent/backend/controller/EmotionAnalysisController.java
empty definition using pc, found symbol in pc: 
empty definition using semanticdb
empty definition using fallback
non-local guesses:

offset: 694
uri: file:///D:/PracticalTraining/Agenttest/EmoAgent/backend/src/main/java/com/emoagent/backend/controller/EmotionAnalysisController.java
text:
```scala
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
    private final EmotionAnalysisService@@ emotionAnalysisService;

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

```


#### Short summary: 

empty definition using pc, found symbol in pc: 