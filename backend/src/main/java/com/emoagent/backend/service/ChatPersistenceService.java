package com.emoagent.backend.service;

import com.emoagent.backend.client.AgentClient;
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
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
public class ChatPersistenceService {
    private static final int TITLE_MAX_LENGTH = 15;
    private static final String PROFILE_CONVERSATION_TITLE = "画像采样";

    private final ConversationRepository conversationRepository;
    private final ChatMessageRepository chatMessageRepository;
    private final EmotionRecordRepository emotionRecordRepository;
    private final AgentClient agentClient;
    private final ObjectMapper objectMapper;

    public ChatPersistenceService(
            ConversationRepository conversationRepository,
            ChatMessageRepository chatMessageRepository,
            EmotionRecordRepository emotionRecordRepository,
            AgentClient agentClient,
            ObjectMapper objectMapper) {
        this.conversationRepository = conversationRepository;
        this.chatMessageRepository = chatMessageRepository;
        this.emotionRecordRepository = emotionRecordRepository;
        this.agentClient = agentClient;
        this.objectMapper = objectMapper;
    }

    @Transactional
    public ChatTurn startTurn(ChatRequest request) {
        Instant now = Instant.now();
        Conversation conversation = getOrCreateConversation(request, now);// 创建或者找到一个对话
        // 创建一条用户消息
        ChatMessage userMessage = new ChatMessage(
                UUID.randomUUID().toString(),
                conversation.getId(),
                "user",
                request.text(),
                now);
        // 保存用户问题到数据库
        chatMessageRepository.save(userMessage);
        return new ChatTurn(conversation.getId(), userMessage.getId());
    }

    @Transactional(readOnly = true)
    // 查询最近20条历史消息
    public List<Map<String, Object>> historyBeforeTurn(ChatTurn turn) {
        List<ChatMessage> messages = new ArrayList<>(
                chatMessageRepository.findTop20ByConversationIdAndIdNotOrderByCreatedAtDesc(
                        turn.conversationId(),
                        turn.userMessageId()));
        // 把最早对话放在最前面，和用户聊天顺序一样
        Collections.reverse(messages);
        return messages.stream()
                .map(this::historyItem)
                .toList();
    }

    // 为前端提供用户的最近20条对话记录
    @Transactional(readOnly = true)
    public List<Map<String, Object>> conversationsForUser(String userId) {
        return conversationRepository.findTop20ByUserIdAndTitleNotOrderByUpdatedAtDesc(
                        userId,
                        PROFILE_CONVERSATION_TITLE).stream()
                .map(this::conversationToMap)
                .toList();
    }

    // admin 查看所有用户的最近20条对话记录
    @Transactional(readOnly = true)
    public List<Map<String, Object>> allConversations() {
        return conversationRepository.findTop20ByTitleNotOrderByUpdatedAtDesc(PROFILE_CONVERSATION_TITLE).stream()
                .map(this::conversationToMap)
                .toList();
    }

    private Map<String, Object> conversationToMap(Conversation conversation) {
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("id", conversation.getId());
        item.put("user_id", conversation.getUserId());
        item.put("title", conversation.getTitle());
        item.put("created_at", conversation.getCreatedAt().toString());
        item.put("updated_at", conversation.getUpdatedAt().toString());
        return item;
    }

    // 查询一个对话里面完整聊天记录
    @Transactional(readOnly = true)
    public List<Map<String, Object>> messagesForConversation(String conversationId, String userId) {
        conversationRepository.findByIdAndUserId(conversationId, userId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Conversation not found"));

        return chatMessageRepository.findByConversationIdOrderByCreatedAtAsc(conversationId).stream()
                .map(this::historyItem)
                .toList();
    }

    // 保存情绪记录
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
                Instant.now());
        emotionRecordRepository.save(record);
    }

    @Transactional
    public void saveStandaloneEmotionRecord(String userId, String text, AnalyzeResponse analysisResult) {
        Instant now = Instant.now();
        Conversation conversation = conversationRepository.findByUserIdAndTitle(userId, PROFILE_CONVERSATION_TITLE)
                .map(existing -> {
                    existing.touch(now);
                    return conversationRepository.save(existing);
                })
                .orElseGet(() -> conversationRepository.save(new Conversation(
                        UUID.randomUUID().toString(),
                        userId,
                        PROFILE_CONVERSATION_TITLE,
                        now,
                        now)));
        ChatMessage message = chatMessageRepository.save(new ChatMessage(
                UUID.randomUUID().toString(),
                conversation.getId(),
                "user",
                text,
                now));
        saveEmotionRecord(new ChatTurn(conversation.getId(), message.getId()), analysisResult);
    }

    // 保存ai回答结果
    @Transactional
    public void saveAssistantMessage(String conversationId, Map<String, Object> chatResult) {
        ChatMessage assistantMessage = new ChatMessage(
                UUID.randomUUID().toString(),
                conversationId,
                "assistant",
                assistantContent(chatResult),
                Instant.now());
        chatMessageRepository.save(assistantMessage);
        conversationRepository.findById(conversationId)
                .ifPresent(conversation -> {
                    conversation.touch(assistantMessage.getCreatedAt());
                    conversationRepository.save(conversation);
                });
    }

    // 获取或者创建对话
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
                now));
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
                now));
    }

    private Map<String, Object> historyItem(ChatMessage message) {
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("role", message.getRole());
        item.put("content", message.getContent());
        item.put("created_at", message.getCreatedAt().toString());
        return item;
    }

    // 取前15个字作为新会话标题
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

    // 一次性传递 会话 ID + 消息 ID
    public record ChatTurn(String conversationId, String userMessageId) {
    }

    /**
     * 查询指定用户（或所有用户）的情绪记录，构建 profile 请求载荷。
     * userId 为 null 时查询所有用户。
     */
    private Map<String, Object> buildProfilePayload(String userId) {
        List<EmotionRecord> records;
        if (userId == null) {
            // 查全部用户
            records = emotionRecordRepository.findAll();
        } else {
            records = emotionRecordRepository.findByUserId(userId);
        }

        List<Map<String, Object>> emotionRecords = records.stream().map(er -> {
            Map<String, Object> m = new HashMap<>();
            m.put("id", er.getId());
            m.put("conversation_id", er.getConversationId());
            m.put("message_id", er.getMessageId());
            m.put("final_emotion", er.getFinalEmotion());
            m.put("secondary_emotion", er.getSecondaryEmotion());
            m.put("final_intensity", er.getFinalIntensity());
            m.put("final_confidence", er.getFinalConfidence());
            m.put("is_sarcasm", er.getSarcasm());
            m.put("is_mixed", er.getMixed());
            m.put("raw_analysis_json", er.getRawAnalysisJson());
            m.put("created_at", er.getCreatedAt() != null ? er.getCreatedAt().toString() : "");
            return m;
        }).toList();

        Map<String, Object> payload = new HashMap<>();
        payload.put("user_id", userId);
        payload.put("emotion_records", emotionRecords);
        return payload;
    }

    // 获取用户画像统计数据
    @Transactional(readOnly = true)
    public Map<String, Object> profile(String userId) {
        return agentClient.profile(buildProfilePayload(userId));
    }

    // 生成用户画像（调用大模型）
    @Transactional
    public Map<String, Object> profileGenerate(String userId) {
        return agentClient.profileGenerate(buildProfilePayload(userId));
    }
}
