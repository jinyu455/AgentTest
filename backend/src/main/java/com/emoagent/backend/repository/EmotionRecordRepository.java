package com.emoagent.backend.repository;

import com.emoagent.backend.entity.EmotionRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

//两层表结构，通过用户id找到所有对话，再通过对话找到用户的全部情绪记录
public interface EmotionRecordRepository extends JpaRepository<EmotionRecord, String> {

    @Query("SELECT er FROM EmotionRecord er WHERE er.conversationId IN " +
            "(SELECT c.id FROM Conversation c WHERE c.userId = :userId)")
    List<EmotionRecord> findByUserId(@Param("userId") String userId);

    @Query("SELECT COUNT(er) FROM EmotionRecord er WHERE er.conversationId IN " +
            "(SELECT c.id FROM Conversation c WHERE c.userId = :userId)")
    long countByUserId(@Param("userId") String userId);

    // admin 查看所有用户的情绪记录
    List<EmotionRecord> findAll();
}
