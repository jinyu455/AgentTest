package com.emoagent.backend.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Lob;
import jakarta.persistence.Table;

import java.time.Instant;

@Entity
@Table(name = "emotion_records")
public class EmotionRecord {
    @Id
    @Column(nullable = false, updatable = false)
    private String id;

    @Column(name = "conversation_id", nullable = false)
    private String conversationId;

    @Column(name = "message_id", nullable = false)
    private String messageId;

    @Column(name = "final_emotion")
    private String finalEmotion;

    @Column(name = "secondary_emotion")
    private String secondaryEmotion;

    @Column(name = "final_intensity")
    private Integer finalIntensity;

    @Column(name = "final_confidence")
    private Double finalConfidence;

    @Column(name = "is_sarcasm")
    private Boolean sarcasm;

    @Column(name = "is_mixed")
    private Boolean mixed;

    @Lob
    @Column(name = "raw_analysis_json", nullable = false, columnDefinition = "LONGTEXT")
    private String rawAnalysisJson;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    protected EmotionRecord() {
    }

    public EmotionRecord(
            String id,
            String conversationId,
            String messageId,
            String finalEmotion,
            String secondaryEmotion,
            Integer finalIntensity,
            Double finalConfidence,
            Boolean sarcasm,
            Boolean mixed,
            String rawAnalysisJson,
            Instant createdAt
    ) {
        this.id = id;
        this.conversationId = conversationId;
        this.messageId = messageId;
        this.finalEmotion = finalEmotion;
        this.secondaryEmotion = secondaryEmotion;
        this.finalIntensity = finalIntensity;
        this.finalConfidence = finalConfidence;
        this.sarcasm = sarcasm;
        this.mixed = mixed;
        this.rawAnalysisJson = rawAnalysisJson;
        this.createdAt = createdAt;
    }
}
