package com.emoagent.backend.repository;

import com.emoagent.backend.entity.EmotionRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface EmotionRecordRepository extends JpaRepository<EmotionRecord, String> {

    @Query("SELECT er FROM EmotionRecord er WHERE er.conversationId IN " +
           "(SELECT c.id FROM Conversation c WHERE c.userId = :userId)")
    List<EmotionRecord> findByUserId(@Param("userId") String userId);
}
