package com.emoagent.backend.repository;

import com.emoagent.backend.entity.Conversation;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

//操作对话表，根据conversation_id和user_id查找用户
public interface ConversationRepository extends JpaRepository<Conversation, String> {
    Optional<Conversation> findByIdAndUserId(String id, String userId);

    Optional<Conversation> findByUserIdAndTitle(String userId, String title);

    List<Conversation> findByUserIdOrderByUpdatedAtDesc(String userId);

    List<Conversation> findByUserIdAndTitleNotOrderByUpdatedAtDesc(String userId, String title);

    List<Conversation> findTop20ByUserIdAndTitleNotOrderByUpdatedAtDesc(String userId, String title);

    // admin 查看所有用户的对话
    List<Conversation> findByOrderByUpdatedAtDesc();

    List<Conversation> findByTitleNotOrderByUpdatedAtDesc(String title);
}
