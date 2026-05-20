const STORAGE_KEY = "emoagent.chat.state";
const API_KEY = "emoagent.api.base";
const DEFAULT_API_BASE = "http://localhost:8080";

const state = {
  apiBase: localStorage.getItem(API_KEY) || DEFAULT_API_BASE,
  userId: getOrCreateUserId(),
  conversationId: null,
  messages: [],
  busy: false,
};

const els = {
  apiBaseInput: document.querySelector("#apiBaseInput"),
  chatForm: document.querySelector("#chatForm"),
  confidenceValue: document.querySelector("#confidenceValue"),
  conversationIdLabel: document.querySelector("#conversationIdLabel"),
  conversationTitle: document.querySelector("#conversationTitle"),
  emotionIntensity: document.querySelector("#emotionIntensity"),
  emotionMeta: document.querySelector("#emotionMeta"),
  emotionMeter: document.querySelector("#emotionMeter"),
  messageInput: document.querySelector("#messageInput"),
  messageList: document.querySelector("#messageList"),
  mixedValue: document.querySelector("#mixedValue"),
  newChatButton: document.querySelector("#newChatButton"),
  primaryEmotion: document.querySelector("#primaryEmotion"),
  rawAnalysis: document.querySelector("#rawAnalysis"),
  rawToggle: document.querySelector("#rawToggle"),
  sarcasmValue: document.querySelector("#sarcasmValue"),
  saveApiButton: document.querySelector("#saveApiButton"),
  secondaryEmotion: document.querySelector("#secondaryEmotion"),
  sendButton: document.querySelector("#sendButton"),
  serviceStatus: document.querySelector("#serviceStatus"),
  statusDot: document.querySelector("#statusDot"),
};

init();

function init() {
  restoreState();
  els.apiBaseInput.value = state.apiBase;
  renderMessages();
  renderConversation();
  bindEvents();
  checkHealth();
}

function bindEvents() {
  els.chatForm.addEventListener("submit", handleSubmit);
  els.newChatButton.addEventListener("click", resetConversation);
  els.saveApiButton.addEventListener("click", saveApiBase);
  els.rawToggle.addEventListener("click", () => {
    els.rawAnalysis.hidden = !els.rawAnalysis.hidden;
  });
  els.messageInput.addEventListener("input", autoResizeInput);
  els.messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      els.chatForm.requestSubmit();
    }
  });
}

async function handleSubmit(event) {
  event.preventDefault();
  const text = els.messageInput.value.trim();
  if (!text || state.busy) {
    return;
  }

  state.messages.push({ role: "user", content: text });
  els.messageInput.value = "";
  autoResizeInput();
  setBusy(true);
  renderMessages(true);
  persistState();

  const loadingId = addLoadingMessage();

  try {
    const response = await fetch(`${state.apiBase}/api/emotion/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        user_id: state.userId,
        conversation_id: state.conversationId,
        metadata: { source: "frontend" },
      }),
    });

    if (!response.ok) {
      const detail = await readError(response);
      throw new Error(detail || `请求失败：${response.status}`);
    }

    const data = await response.json();
    state.conversationId = data.conversation_id || state.conversationId;
    removeLoadingMessage(loadingId);
    state.messages.push({
      role: "assistant",
      content: extractReply(data.chat_result),
      analysis: data.analysis_result,
    });
    updateInsights(data.analysis_result);
    renderConversation(text);
    renderMessages(true);
    persistState();
  } catch (error) {
    removeLoadingMessage(loadingId);
    state.messages.push({
      role: "assistant",
      content: `连接后端时遇到问题：${error.message}`,
      error: true,
    });
    setServiceStatus("连接失败", "offline");
    renderMessages(true);
  } finally {
    setBusy(false);
  }
}

function renderMessages(scrollToBottom = false) {
  const welcome = {
    role: "assistant",
    content: "你好，我会先理解你的情绪，再给出更贴近当下状态的回应。把想说的话发给我就好。",
  };
  const messages = state.messages.length ? state.messages : [welcome];
  els.messageList.innerHTML = messages.map(messageTemplate).join("");
  if (scrollToBottom) {
    els.messageList.scrollTop = els.messageList.scrollHeight;
  }
}

function messageTemplate(message) {
  const isUser = message.role === "user";
  const chips = message.analysis ? insightChips(message.analysis) : "";
  return `
    <article class="message ${isUser ? "user" : "assistant"}">
      <div class="avatar" aria-hidden="true">${isUser ? "你" : "E"}</div>
      <div class="bubble">
        <p>${escapeHtml(message.content)}</p>
        ${chips}
      </div>
    </article>
  `;
}

function insightChips(analysis) {
  const judge = analysis?.judge_result || {};
  const chips = [
    judge.final_emotion && `情绪 ${judge.final_emotion}`,
    toPercent(judge.final_confidence) && `置信 ${toPercent(judge.final_confidence)}`,
    judge.final_intensity != null && `强度 ${judge.final_intensity}`,
  ].filter(Boolean);

  if (!chips.length) {
    return "";
  }

  return `<div class="meta-line">${chips.map((chip) => `<span class="chip">${escapeHtml(chip)}</span>`).join("")}</div>`;
}

function addLoadingMessage() {
  const id = `loading-${Date.now()}`;
  els.messageList.insertAdjacentHTML(
    "beforeend",
    `
      <article class="message assistant loading" id="${id}">
        <div class="avatar" aria-hidden="true">E</div>
        <div class="bubble">
          <div class="typing" aria-label="正在回复">
            <span></span><span></span><span></span>
          </div>
        </div>
      </article>
    `
  );
  els.messageList.scrollTop = els.messageList.scrollHeight;
  return id;
}

function removeLoadingMessage(id) {
  document.querySelector(`#${id}`)?.remove();
}

function updateInsights(analysis) {
  const judge = analysis?.judge_result || {};
  const intensity = asNumber(judge.final_intensity);
  const confidence = asNumber(judge.final_confidence);
  const meterValue = intensity != null ? Math.min(100, Math.max(0, intensity * 10)) : 0;

  els.primaryEmotion.textContent = normalizeValue(judge.final_emotion, "未知");
  els.secondaryEmotion.textContent = normalizeValue(judge.secondary_emotion, "--");
  els.emotionIntensity.textContent = intensity != null ? String(intensity) : "--";
  els.confidenceValue.textContent = confidence != null ? toPercent(confidence) : "--";
  els.sarcasmValue.textContent = formatBoolean(judge.is_sarcasm);
  els.mixedValue.textContent = formatBoolean(judge.is_mixed);
  els.emotionMeter.style.width = `${meterValue}%`;
  els.emotionMeta.textContent =
    confidence != null ? `模型对本轮判断的置信度为 ${toPercent(confidence)}。` : "已收到分析结果。";
  els.rawAnalysis.textContent = JSON.stringify(analysis || {}, null, 2);
}

function renderConversation(lastText) {
  const title = state.messages.find((message) => message.role === "user")?.content || lastText;
  els.conversationTitle.textContent = title ? clamp(title, 24) : "开始一次情绪对话";
  els.conversationIdLabel.textContent = state.conversationId || "尚未创建";
}

async function checkHealth() {
  setServiceStatus("检查中", "");
  try {
    const response = await fetch(`${state.apiBase}/api/emotion/health`);
    if (!response.ok) {
      throw new Error(String(response.status));
    }
    setServiceStatus("已连接", "online");
  } catch {
    setServiceStatus("未连接", "offline");
  }
}

function setServiceStatus(text, statusClass) {
  els.serviceStatus.textContent = text;
  els.statusDot.className = `status-dot ${statusClass || ""}`.trim();
}

function setBusy(isBusy) {
  state.busy = isBusy;
  els.sendButton.disabled = isBusy;
  els.messageInput.disabled = isBusy;
}

function resetConversation() {
  state.conversationId = null;
  state.messages = [];
  els.rawAnalysis.textContent = "{}";
  updateInsights(null);
  renderConversation();
  renderMessages();
  persistState();
  els.messageInput.focus();
}

function saveApiBase() {
  state.apiBase = els.apiBaseInput.value.trim().replace(/\/$/, "") || DEFAULT_API_BASE;
  els.apiBaseInput.value = state.apiBase;
  localStorage.setItem(API_KEY, state.apiBase);
  checkHealth();
}

function restoreState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    state.conversationId = saved.conversationId || null;
    state.messages = Array.isArray(saved.messages) ? saved.messages : [];
  } catch {
    state.conversationId = null;
    state.messages = [];
  }
}

function persistState() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      conversationId: state.conversationId,
      messages: state.messages.slice(-40),
    })
  );
}

function getOrCreateUserId() {
  const key = "emoagent.user.id";
  const existing = localStorage.getItem(key);
  if (existing) {
    return existing;
  }
  const created = crypto.randomUUID ? crypto.randomUUID() : `user-${Date.now()}`;
  localStorage.setItem(key, created);
  return created;
}

function extractReply(chatResult) {
  if (!chatResult) {
    return "我收到了你的消息，但后端没有返回回复内容。";
  }
  if (typeof chatResult.reply === "string" && chatResult.reply.trim()) {
    return chatResult.reply.trim();
  }
  if (typeof chatResult.content === "string" && chatResult.content.trim()) {
    return chatResult.content.trim();
  }
  if (typeof chatResult.message === "string" && chatResult.message.trim()) {
    return chatResult.message.trim();
  }
  return JSON.stringify(chatResult, null, 2);
}

async function readError(response) {
  try {
    const data = await response.json();
    return data.message || data.detail || JSON.stringify(data);
  } catch {
    return response.statusText;
  }
}

function autoResizeInput() {
  els.messageInput.style.height = "auto";
  els.messageInput.style.height = `${Math.min(150, els.messageInput.scrollHeight)}px`;
}

function normalizeValue(value, fallback) {
  return value == null || value === "" ? fallback : String(value);
}

function formatBoolean(value) {
  if (value === true) {
    return "是";
  }
  if (value === false) {
    return "否";
  }
  return "--";
}

function asNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function toPercent(value) {
  const number = asNumber(value);
  if (number == null) {
    return "";
  }
  return `${Math.round(number * 100)}%`;
}

function clamp(text, length) {
  return text.length > length ? `${text.slice(0, length)}...` : text;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
