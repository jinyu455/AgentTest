package com.emoagent.backend.service;

import com.emoagent.backend.client.AgentClient;
import com.emoagent.backend.dto.AnalyzeResponse;
import com.emoagent.backend.dto.ChatRequest;
import com.emoagent.backend.entity.ChatMessage;
import com.emoagent.backend.entity.Conversation;
import com.emoagent.backend.entity.EmotionRecord;
import com.emoagent.backend.entity.User;
import com.emoagent.backend.entity.UserProfile;
import com.emoagent.backend.repository.ChatMessageRepository;
import com.emoagent.backend.repository.ConversationRepository;
import com.emoagent.backend.repository.EmotionRecordRepository;
import com.emoagent.backend.repository.UserProfileRepository;
import com.emoagent.backend.repository.UserRepository;
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
    private static final int PROFILE_UPDATE_THRESHOLD = 10;

    private final ConversationRepository conversationRepository;
    private final ChatMessageRepository chatMessageRepository;
    private final EmotionRecordRepository emotionRecordRepository;
    private final UserProfileRepository userProfileRepository;
    private final UserRepository userRepository;
    private final AgentClient agentClient;
    private final ObjectMapper objectMapper;

    public ChatPersistenceService(
            ConversationRepository conversationRepository,
            ChatMessageRepository chatMessageRepository,
            EmotionRecordRepository emotionRecordRepository,
            UserProfileRepository userProfileRepository,
            UserRepository userRepository,
            AgentClient agentClient,
            ObjectMapper objectMapper) {
        this.conversationRepository = conversationRepository;
        this.chatMessageRepository = chatMessageRepository;
        this.emotionRecordRepository = emotionRecordRepository;
        this.userProfileRepository = userProfileRepository;
        this.userRepository = userRepository;
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

    // 为前端提供用户的对话记录
    @Transactional(readOnly = true)
    public List<Map<String, Object>> conversationsForUser(String userId) {
        return conversationRepository.findByUserIdAndTitleNotOrderByUpdatedAtDesc(
                        userId,
                        PROFILE_CONVERSATION_TITLE).stream()
                .map(this::conversationToMap)
                .toList();
    }

    // admin 查看所有用户的对话记录
    @Transactional(readOnly = true)
    public List<Map<String, Object>> allConversations() {
        return conversationRepository.findByTitleNotOrderByUpdatedAtDesc(PROFILE_CONVERSATION_TITLE).stream()
                .map(this::conversationToMap)
                .toList();
    }

    private Map<String, Object> conversationToMap(Conversation conversation) {
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("id", conversation.getId());
        item.put("user_id", conversation.getUserId());
        item.put("username", userRepository.findById(conversation.getUserId()).map(User::getUsername).orElse(conversation.getUserId()));
        item.put("title", conversation.getTitle());
        item.put("created_at", conversation.getCreatedAt().toString());
        item.put("updated_at", conversation.getUpdatedAt().toString());
        return item;
    }

    // 查询一个对话里面完整聊天记录
    @Transactional(readOnly = true)
    public List<Map<String, Object>> messagesForConversation(String conversationId, String userId) {
        return messagesForConversation(conversationId, userId, false);
    }

    @Transactional(readOnly = true)
    public List<Map<String, Object>> messagesForConversation(String conversationId, String userId, boolean admin) {
        if (admin) {
            conversationRepository.findById(conversationId)
                    .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Conversation not found"));
            return chatMessageRepository.findByConversationIdOrderByCreatedAtAsc(conversationId).stream()
                    .map(this::historyItem)
                    .toList();
        }

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
        if (userId != null) {
            Map<String, Object> sampleConversation = latestConversationSample(userId);
            payload.put("sample_conversation", sampleConversation);
            payload.put("chat_history", sampleConversation.getOrDefault("messages", List.of()));
        } else {
            payload.put("chat_history", List.of());
        }
        return payload;
    }

    private Map<String, Object> latestConversationSample(String userId) {
        Map<String, Object> sample = new LinkedHashMap<>();
        List<Conversation> conversations =
                conversationRepository.findTop20ByUserIdAndTitleNotOrderByUpdatedAtDesc(userId, PROFILE_CONVERSATION_TITLE);

        for (Conversation conversation : conversations) {
            List<Map<String, Object>> messages = chatMessageRepository
                    .findByConversationIdOrderByCreatedAtAsc(conversation.getId()).stream()
                    .filter(message -> "user".equals(message.getRole()) || "assistant".equals(message.getRole()))
                    .limit(20)
                    .map(message -> {
                        Map<String, Object> item = new LinkedHashMap<>();
                        item.put("role", message.getRole());
                        item.put("content", message.getContent());
                        item.put("created_at", message.getCreatedAt() != null ? message.getCreatedAt().toString() : "");
                        return item;
                    })
                    .toList();

            if (!messages.isEmpty()) {
                sample.put("conversation_id", conversation.getId());
                sample.put("title", conversation.getTitle());
                sample.put("updated_at", conversation.getUpdatedAt() != null ? conversation.getUpdatedAt().toString() : "");
                sample.put("messages", messages);
                return sample;
            }
        }

        sample.put("messages", List.of());
        return sample;
    }

    // 获取用户画像统计数据
    @Transactional(readOnly = true)
    public Map<String, Object> profile(String userId) {
        return agentClient.profile(buildProfilePayload(userId));
    }

    // 生成用户画像（调用大模型），只有新增情绪记录满10条才重新生成
    @Transactional
    public Map<String, Object> profileGenerate(String userId) {
        return profileGenerate(userId, false);
    }

    // force=true 时手动重新生成；否则读取缓存，新增满10条才自动重新生成
    @Transactional
    public Map<String, Object> profileGenerate(String userId, boolean force) {
        // 获取当前情绪记录总数
        long currentCount = emotionRecordRepository.countByUserId(userId);
        if (currentCount == 0) {
            Map<String, Object> result = new HashMap<>();
            result.put("user_id", userId);
            result.put("profile", "");
            result.put("record_count", 0);
            result.put("cached", false);
            return result;
        }

        // 检查是否已有缓存的画像
        return userProfileRepository.findByUserId(userId)
                .map(existing -> {
                    // 计算新增记录数
                    long newRecordsCount = currentCount - existing.getRecordCount();
                    if (force || newRecordsCount >= PROFILE_UPDATE_THRESHOLD) {
                        // 手动强制生成，或新增满10条，重新生成
                        return regenerateProfile(userId, currentCount);
                    }
                    // 不满10条，返回缓存的画像
                    Map<String, Object> result = fromJsonMap(existing.getProfileData());
                    result.put("user_id", userId);
                    result.put("record_count", existing.getRecordCount());
                    result.put("cached", true);
                    return result;
                })
                .orElseGet(() -> {
                    // 没有缓存，首次生成
                    return regenerateProfile(userId, currentCount);
                });
    }

    private Map<String, Object> regenerateProfile(String userId, long recordCount) {
        Map<String, Object> payload = buildProfilePayload(userId);
        Map<String, Object> result = new HashMap<>(agentClient.profileGenerate(payload));

        // Python 返回的是完整画像结构，直接序列化存库
        String profileContent = toJson(result);

        Instant now = Instant.now();
        userProfileRepository.findByUserId(userId)
                .map(existing -> {
                    existing.setProfileData(profileContent);
                    existing.setRecordCount((int) recordCount);
                    existing.setUpdatedAt(now);
                    return userProfileRepository.save(existing);
                })
                .orElseGet(() -> userProfileRepository.save(new UserProfile(
                        UUID.randomUUID().toString(),
                        userId,
                        profileContent,
                        (int) recordCount,
                        now,
                        now)));

        result.put("user_id", userId);
        result.put("record_count", recordCount);
        result.put("cached", false);
        return result;
    }

    private Map<String, Object> fromJsonMap(String value) {
        try {
            Object parsed = objectMapper.readValue(value, Map.class);
            if (parsed instanceof Map<?, ?> map) {
                Map<String, Object> result = new HashMap<>();
                map.forEach((key, entryValue) -> result.put(String.valueOf(key), entryValue));
                return result;
            }
            throw new IllegalStateException("Cached user profile is not a JSON object");
        } catch (JacksonException exception) {
            throw new IllegalStateException("Failed to deserialize cached user profile", exception);
        }
    }
}
