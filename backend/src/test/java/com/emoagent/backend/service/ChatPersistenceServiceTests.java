package com.emoagent.backend.service;

import com.emoagent.backend.client.AgentClient;
import com.emoagent.backend.dto.ChatRequest;
import com.emoagent.backend.entity.ChatMessage;
import com.emoagent.backend.entity.UserProfile;
import com.emoagent.backend.repository.ChatMessageRepository;
import com.emoagent.backend.repository.ConversationRepository;
import com.emoagent.backend.repository.EmotionRecordRepository;
import com.emoagent.backend.repository.UserProfileRepository;
import com.emoagent.backend.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;
import tools.jackson.databind.ObjectMapper;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.anyMap;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class ChatPersistenceServiceTests {
    @Mock
    private ConversationRepository conversationRepository;

    @Mock
    private ChatMessageRepository chatMessageRepository;

    @Mock
    private EmotionRecordRepository emotionRecordRepository;

    @Mock
    private UserProfileRepository userProfileRepository;

    @Mock
    private UserRepository userRepository;

    @Mock
    private AgentClient agentClient;

    @Test
    void startTurnRejectsConversationOwnedByAnotherUser() {
        ChatPersistenceService service = service();
        ChatRequest request = new ChatRequest(
                "hello",
                "user-2",
                "conversation-1",
                null,
                null,
                null
        );
        when(conversationRepository.findByIdAndUserId("conversation-1", "user-2"))
                .thenReturn(Optional.empty());
        when(conversationRepository.existsById("conversation-1")).thenReturn(true);

        assertThatThrownBy(() -> service.startTurn(request))
                .isInstanceOfSatisfying(ResponseStatusException.class, exception ->
                        assertThat(exception.getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN));
    }

    @Test
    void historyBeforeTurnReturnsPreviousMessagesInChronologicalOrder() {
        ChatPersistenceService service = service();
        ChatPersistenceService.ChatTurn turn = new ChatPersistenceService.ChatTurn("conversation-1", "current-user-message");
        ChatMessage userMessage = new ChatMessage(
                "previous-user-message",
                "conversation-1",
                "user",
                "first",
                Instant.parse("2026-05-19T10:00:00Z")
        );
        ChatMessage assistantMessage = new ChatMessage(
                "previous-assistant-message",
                "conversation-1",
                "assistant",
                "second",
                Instant.parse("2026-05-19T10:01:00Z")
        );
        when(chatMessageRepository.findTop20ByConversationIdAndIdNotOrderByCreatedAtDesc(
                "conversation-1",
                "current-user-message"
        )).thenReturn(List.of(assistantMessage, userMessage));

        List<Map<String, Object>> history = service.historyBeforeTurn(turn);

        assertThat(history).extracting(item -> item.get("role"))
                .containsExactly("user", "assistant");
        assertThat(history).extracting(item -> item.get("content"))
                .containsExactly("first", "second");
        verify(chatMessageRepository).findTop20ByConversationIdAndIdNotOrderByCreatedAtDesc(
                "conversation-1",
                "current-user-message"
        );
    }

    @Test
    void profileGenerateReturnsCachedProfileAsTopLevelFields() {
        ChatPersistenceService service = service();
        UserProfile cachedProfile = new UserProfile(
                "profile-1",
                "user-1",
                "{\"total_records\":10,\"dominant_emotion\":\"开心\",\"summary\":\"缓存画像\"}",
                10,
                Instant.parse("2026-06-09T07:00:00Z"),
                Instant.parse("2026-06-09T07:00:00Z")
        );
        when(emotionRecordRepository.countByUserId("user-1")).thenReturn(12L);
        when(userProfileRepository.findByUserId("user-1")).thenReturn(Optional.of(cachedProfile));

        Map<String, Object> profile = service.profileGenerate("user-1");

        assertThat(profile)
                .containsEntry("total_records", 10)
                .containsEntry("dominant_emotion", "开心")
                .containsEntry("summary", "缓存画像")
                .containsEntry("user_id", "user-1")
                .containsEntry("record_count", 10)
                .containsEntry("cached", true);
    }

    @Test
    void profileGenerateWithForceRegeneratesEvenWhenThresholdIsNotMet() {
        ChatPersistenceService service = service();
        UserProfile cachedProfile = new UserProfile(
                "profile-1",
                "user-1",
                "{\"total_records\":10,\"dominant_emotion\":\"开心\",\"summary\":\"缓存画像\"}",
                10,
                Instant.parse("2026-06-09T07:00:00Z"),
                Instant.parse("2026-06-09T07:00:00Z")
        );
        when(emotionRecordRepository.countByUserId("user-1")).thenReturn(12L);
        when(emotionRecordRepository.findByUserId("user-1")).thenReturn(List.of());
        when(userProfileRepository.findByUserId("user-1")).thenReturn(Optional.of(cachedProfile));
        when(agentClient.profileGenerate(anyMap())).thenReturn(Map.of(
                "total_records", 12,
                "dominant_emotion", "平静",
                "summary", "重新生成画像"
        ));
        when(userProfileRepository.save(cachedProfile)).thenReturn(cachedProfile);

        Map<String, Object> profile = service.profileGenerate("user-1", true);

        assertThat(profile)
                .containsEntry("total_records", 12)
                .containsEntry("dominant_emotion", "平静")
                .containsEntry("summary", "重新生成画像")
                .containsEntry("record_count", 12L)
                .containsEntry("cached", false);
        verify(agentClient).profileGenerate(anyMap());
    }

    private ChatPersistenceService service() {
        return new ChatPersistenceService(
                conversationRepository,
                chatMessageRepository,
                emotionRecordRepository,
                userProfileRepository,
                userRepository,
                agentClient,
                new ObjectMapper()
        );
    }
}
