package com.emoagent.backend.repository;

import com.emoagent.backend.entity.ChatMessage;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface ChatMessageRepository extends JpaRepository<ChatMessage, String> {
    List<ChatMessage> findTop20ByConversationIdAndIdNotOrderByCreatedAtDesc(String conversationId, String id);

    List<ChatMessage> findByConversationIdOrderByCreatedAtAsc(String conversationId);
}
