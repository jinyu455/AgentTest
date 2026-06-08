package com.emoagent.backend.service;

import com.emoagent.backend.client.AgentClient;
import com.emoagent.backend.dto.ChatRequest;
import com.emoagent.backend.entity.ChatMessage;
import com.emoagent.backend.repository.ChatMessageRepository;
import com.emoagent.backend.repository.ConversationRepository;
import com.emoagent.backend.repository.EmotionRecordRepository;
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

    private ChatPersistenceService service() {
        return new ChatPersistenceService(
                conversationRepository,
                chatMessageRepository,
                emotionRecordRepository,
                agentClient,
                new ObjectMapper()
        );
    }
}
