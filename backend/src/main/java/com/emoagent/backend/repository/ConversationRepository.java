package com.emoagent.backend.repository;

import com.emoagent.backend.entity.Conversation;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

//操作对话表，根据conversation_id和user_id查找用户,查询用户最近20条对话
public interface ConversationRepository extends JpaRepository<Conversation, String> {
    Optional<Conversation> findByIdAndUserId(String id, String userId);

    List<Conversation> findTop20ByUserIdOrderByUpdatedAtDesc(String userId);

    // admin 查看所有用户的最近20条对话
    List<Conversation> findTop20ByOrderByUpdatedAtDesc();
}
