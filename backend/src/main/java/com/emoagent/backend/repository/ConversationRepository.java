package com.emoagent.backend.repository;

import com.emoagent.backend.entity.Conversation;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface ConversationRepository extends JpaRepository<Conversation, String> {
    Optional<Conversation> findByIdAndUserId(String id, String userId);

    List<Conversation> findTop20ByUserIdOrderByUpdatedAtDesc(String userId);
}
