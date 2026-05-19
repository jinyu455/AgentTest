package com.emoagent.backend.repository;

import com.emoagent.backend.entity.EmotionRecord;
import org.springframework.data.jpa.repository.JpaRepository;

public interface EmotionRecordRepository extends JpaRepository<EmotionRecord, String> {
}
