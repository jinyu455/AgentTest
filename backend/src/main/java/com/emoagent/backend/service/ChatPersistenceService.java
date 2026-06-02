package com.emoagent.backend.service;

import com.emoagent.backend.dto.AnalyzeResponse;
import com.emoagent.backend.dto.ChatRequest;
import com.emoagent.backend.entity.ChatMessage;
import com.emoagent.backend.entity.Conversation;
import com.emoagent.backend.entity.EmotionRecord;
import com.emoagent.backend.repository.ChatMessageRepository;
import com.emoagent.backend.repository.ConversationRepository;
import com.emoagent.backend.repository.EmotionRecordRepository;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.ObjectMapper;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
public class ChatPersistenceService {
    private static final int TITLE_MAX_LENGTH = 30;

    private final ConversationRepository conversationRepository;
    private final ChatMessageRepository chatMessageRepository;
    private final EmotionRecordRepository emotionRecordRepository;
    private final ObjectMapper objectMapper;

    public ChatPersistenceService(
            ConversationRepository conversationRepository,
            ChatMessageRepository chatMessageRepository,
            EmotionRecordRepository emotionRecordRepository,
            ObjectMapper objectMapper
    ) {
        this.conversationRepository = conversationRepository;
        this.chatMessageRepository = chatMessageRepository;
        this.emotionRecordRepository = emotionRecordRepository;
        this.objectMapper = objectMapper;
    }

    @Transactional
    public ChatTurn startTurn(ChatRequest request) {
        Instant now = Instant.now();
        Conversation conversation = getOrCreateConversation(request, now);
        ChatMessage userMessage = new ChatMessage(
                UUID.randomUUID().toString(),
                conversation.getId(),
                "user",
                request.text(),
                now
        );
        chatMessageRepository.save(userMessage);
        return new ChatTurn(conversation.getId(), userMessage.getId());
    }

    @Transactional(readOnly = true)
    public List<Map<String, Object>> historyBeforeTurn(ChatTurn turn) {
        List<ChatMessage> messages = new ArrayList<>(
                chatMessageRepository.findTop20ByConversationIdAndIdNotOrderByCreatedAtDesc(
                        turn.conversationId(),
                        turn.userMessageId()
                )
        );
        Collections.reverse(messages);
        return messages.stream()
                .map(this::historyItem)
                .toList();
    }

    @Transactional(readOnly = true)
    public List<Map<String, Object>> conversationsForUser(String userId) {
        return conversationRepository.findTop20ByUserIdOrderByUpdatedAtDesc(userId).stream()
                .map(conversation -> {
                    Map<String, Object> item = new LinkedHashMap<>();
                    item.put("id", conversation.getId());
                    item.put("title", conversation.getTitle());
                    item.put("created_at", conversation.getCreatedAt().toString());
                    item.put("updated_at", conversation.getUpdatedAt().toString());
                    return item;
                })
                .toList();
    }

    @Transactional(readOnly = true)
    public List<Map<String, Object>> messagesForConversation(String conversationId, String userId) {
        conversationRepository.findByIdAndUserId(conversationId, userId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Conversation not found"));

        return chatMessageRepository.findByConversationIdOrderByCreatedAtAsc(conversationId).stream()
                .map(this::historyItem)
                .toList();
    }

    @Transactional
    public void saveEmotionRecord(ChatTurn turn, AnalyzeResponse analysisResult) {
        Map<String, Object> judgeResult = analysisResult.judgeResult();
        EmotionRecord record = new EmotionRecord(
                UUID.randomUUID().toString(),
                turn.conversationId(),
                turn.userMessageId(),
                stringValue(judgeResult.get("final_emotion")),
                stringValue(judgeResult.get("secondary_emotion")),
                integerValue(judgeResult.get("final_intensity")),
                doubleValue(judgeResult.get("final_confidence")),
                booleanValue(judgeResult.get("is_sarcasm")),
                booleanValue(judgeResult.get("is_mixed")),
                toJson(analysisResult),
                Instant.now()
        );
        emotionRecordRepository.save(record);
    }

    @Transactional
    public void saveAssistantMessage(String conversationId, Map<String, Object> chatResult) {
        ChatMessage assistantMessage = new ChatMessage(
                UUID.randomUUID().toString(),
                conversationId,
                "assistant",
                assistantContent(chatResult),
                Instant.now()
        );
        chatMessageRepository.save(assistantMessage);
        conversationRepository.findById(conversationId)
                .ifPresent(conversation -> {
                    conversation.touch(assistantMessage.getCreatedAt());
                    conversationRepository.save(conversation);
                });
    }

    private Conversation getOrCreateConversation(ChatRequest request, Instant now) {
        if (hasText(request.conversationId())) {
            return conversationRepository.findByIdAndUserId(request.conversationId(), request.userId())
                    .map(conversation -> {
                        conversation.touch(now);
                        return conversationRepository.save(conversation);
                    })
                    .orElseGet(() -> createConversationWithProvidedId(request, now));
        }

        return conversationRepository.save(new Conversation(
                UUID.randomUUID().toString(),
                request.userId(),
                buildTitle(request.text()),
                now,
                now
        ));
    }

    private Conversation createConversationWithProvidedId(ChatRequest request, Instant now) {
        if (conversationRepository.existsById(request.conversationId())) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Conversation does not belong to current user");
        }
        return conversationRepository.save(new Conversation(
                request.conversationId(),
                request.userId(),
                buildTitle(request.text()),
                now,
                now
        ));
    }

    private Map<String, Object> historyItem(ChatMessage message) {
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("role", message.getRole());
        item.put("content", message.getContent());
        item.put("created_at", message.getCreatedAt().toString());
        return item;
    }

    private String buildTitle(String text) {
        String normalized = text == null ? "新会话" : text.strip();
        if (normalized.isEmpty()) {
            return "新会话";
        }
        return normalized.length() > TITLE_MAX_LENGTH
                ? normalized.substring(0, TITLE_MAX_LENGTH)
                : normalized;
    }

    private String assistantContent(Map<String, Object> chatResult) {
        Object reply = chatResult.get("reply");
        if (reply != null && hasText(reply.toString())) {
            return reply.toString();
        }
        return toJson(chatResult);
    }

    private String toJson(Object value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JacksonException exception) {
            throw new IllegalStateException("Failed to serialize chat persistence payload", exception);
        }
    }

    private String stringValue(Object value) {
        return value == null ? null : value.toString();
    }

    private Integer integerValue(Object value) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value == null || !hasText(value.toString())) {
            return null;
        }
        return Integer.parseInt(value.toString());
    }

    private Double doubleValue(Object value) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value == null || !hasText(value.toString())) {
            return null;
        }
        return Double.parseDouble(value.toString());
    }

    private Boolean booleanValue(Object value) {
        if (value instanceof Boolean bool) {
            return bool;
        }
        if (value == null || !hasText(value.toString())) {
            return null;
        }
        return Boolean.parseBoolean(value.toString());
    }

    private boolean hasText(String value) {
        return value != null && !value.isBlank();
    }

    public record ChatTurn(String conversationId, String userMessageId) {
    }
}
