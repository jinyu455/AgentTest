package com.emoagent.backend.repository;

import com.emoagent.backend.entity.ChatMessage;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

//chat表，查找一个对话所有消息，时间从早到晚排序，查找最近20条对话信息
public interface ChatMessageRepository extends JpaRepository<ChatMessage, String> {
    List<ChatMessage> findTop20ByConversationIdAndIdNotOrderByCreatedAtDesc(String conversationId, String id);

    List<ChatMessage> findByConversationIdOrderByCreatedAtAsc(String conversationId);
}