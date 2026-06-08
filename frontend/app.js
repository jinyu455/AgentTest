const STORAGE_KEY = "emoagent.chat.state";
const API_KEY = "emoagent.api.base";
const AUTH_KEY = "emoagent.auth";
const DEFAULT_API_BASE = "http://localhost:8080";

const state = {
  apiBase: localStorage.getItem(API_KEY) || DEFAULT_API_BASE,
  auth: loadAuth(),
  conversationId: null,
  conversations: [],
  conversationsStatus: "loading",
  messages: [],
  lastProfileSignal: null,
  profileAvatarSvg: "",
  pendingProfileGenerate: false,
  pendingProfileText: "",
  busy: false,
  profileBusy: false,
  captchaKey: "",
};

let els = {};

boot();

async function boot() {
  if (window.loadPartials) {
    await window.loadPartials();
  }
  els = collectElements();
  init();
}

function collectElements() {
  return {
    appShell: document.querySelector(".app-shell"),
    authMessage: document.querySelector("#authMessage"),
    authTabs: document.querySelectorAll("[data-auth-tab]"),
    authView: document.querySelector("#authView"),
    autoLogin: document.querySelector("#autoLogin"),
    apiBaseInput: document.querySelector("#apiBaseInput"),
    captchaCode: document.querySelector("#captchaCode"),
    captchaImage: document.querySelector("#captchaImage"),
    chatForm: document.querySelector("#chatForm"),
    confidenceValue: document.querySelector("#confidenceValue"),
    conversationIdLabel: document.querySelector("#conversationIdLabel"),
    conversationList: document.querySelector("#conversationList"),
    conversationTitle: document.querySelector("#conversationTitle"),
    currentUsername: document.querySelector("#currentUsername"),
    currentUserRole: document.querySelector("#currentUserRole"),
    emotionIntensity: document.querySelector("#emotionIntensity"),
    emotionMeta: document.querySelector("#emotionMeta"),
    emotionMeter: document.querySelector("#emotionMeter"),
    loginButton: document.querySelector("#loginButton"),
    loginForm: document.querySelector("#loginForm"),
    loginPassword: document.querySelector("#loginPassword"),
    loginUsername: document.querySelector("#loginUsername"),
    logoutButton: document.querySelector("#logoutButton"),
    messageInput: document.querySelector("#messageInput"),
    messageList: document.querySelector("#messageList"),
    mixedValue: document.querySelector("#mixedValue"),
    newChatButton: document.querySelector("#newChatButton"),
    primaryEmotion: document.querySelector("#primaryEmotion"),
    profileAvatar: document.querySelector("#profileAvatar"),
    profileEmotion: document.querySelector("#profileEmotion"),
    profileStatus: document.querySelector("#profileStatus"),
    profileSummary: document.querySelector("#profileSummary"),
    rawAnalysis: document.querySelector("#rawAnalysis"),
    rawToggle: document.querySelector("#rawToggle"),
    refreshCaptchaButton: document.querySelector("#refreshCaptchaButton"),
    registerButton: document.querySelector("#registerButton"),
    registerForm: document.querySelector("#registerForm"),
    registerPassword: document.querySelector("#registerPassword"),
    registerUsername: document.querySelector("#registerUsername"),
    sarcasmValue: document.querySelector("#sarcasmValue"),
    saveApiButton: document.querySelector("#saveApiButton"),
    secondaryEmotion: document.querySelector("#secondaryEmotion"),
    sendButton: document.querySelector("#sendButton"),
    serviceStatus: document.querySelector("#serviceStatus"),
    statusDot: document.querySelector("#statusDot"),
    traitList: document.querySelector("#traitList"),
    userMiniAvatar: document.querySelector("#userMiniAvatar"),
  };
}

function init() {
  restoreState();
  els.apiBaseInput.value = state.apiBase;
  bindEvents();
  renderRoute();
  checkHealth();
}

function bindEvents() {
  els.authTabs.forEach((tab) => {
    tab.addEventListener("click", () => switchAuthTab(tab.dataset.authTab));
  });
  els.loginForm.addEventListener("submit", handleLogin);
  els.registerForm.addEventListener("submit", handleRegister);
  els.refreshCaptchaButton.addEventListener("click", loadCaptcha);
  els.logoutButton.addEventListener("click", logout);
  els.chatForm.addEventListener("submit", handleSubmit);
  els.newChatButton.addEventListener("click", resetConversation);
  els.conversationList.addEventListener("click", handleConversationListClick);
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

function renderRoute() {
  const isAuthed = Boolean(state.auth?.token);
  els.authView.hidden = isAuthed;
  els.appShell.hidden = !isAuthed;

  if (!isAuthed) {
    switchAuthTab("login");
    loadCaptcha();
    return;
  }

  renderCurrentUser();
  renderMessages();
  renderConversation();
  renderConversationList();
  renderProfileAvatar(null);
  loadConversations();
  refreshProfileAvatar({ generate: false, reason: "initial" });
}

function switchAuthTab(tabName) {
  const isRegister = tabName === "register";
  els.loginForm.hidden = isRegister;
  els.registerForm.hidden = !isRegister;
  els.authTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.authTab === tabName));
  setAuthMessage("");
  if (isRegister && !state.captchaKey) {
    loadCaptcha();
  }
}

async function handleLogin(event) {
  event.preventDefault();
  setAuthBusy(true);
  setAuthMessage("正在登录...");
  try {
    const response = await apiFetch("/api/auth/login", {
      method: "POST",
      auth: false,
      body: JSON.stringify({
        username: els.loginUsername.value.trim(),
        password: els.loginPassword.value,
        auto_login: els.autoLogin.checked,
      }),
    });
    const data = await response.json();
    state.auth = {
      token: data.token,
      userId: data.user_id,
      username: data.username,
      role: data.role || "user",
    };
    localStorage.setItem(AUTH_KEY, JSON.stringify(state.auth));
    setAuthMessage("");
    renderRoute();
  } catch (error) {
    setAuthMessage(error.message || "登录失败", "error");
  } finally {
    setAuthBusy(false);
  }
}

async function handleRegister(event) {
  event.preventDefault();
  setAuthBusy(true);
  setAuthMessage("正在创建账号...");
  try {
    const response = await apiFetch("/api/auth/register", {
      method: "POST",
      auth: false,
      body: JSON.stringify({
        username: els.registerUsername.value.trim(),
        password: els.registerPassword.value,
        captcha_code: els.captchaCode.value.trim(),
        captcha_key: state.captchaKey,
      }),
    });
    const data = await response.json().catch(() => ({}));
    els.loginUsername.value = els.registerUsername.value.trim();
    els.loginPassword.value = "";
    els.registerForm.reset();
    state.captchaKey = "";
    switchAuthTab("login");
    setAuthMessage(data.message || "注册成功，请登录。", "success");
  } catch (error) {
    setAuthMessage(error.message || "注册失败", "error");
    await loadCaptcha();
  } finally {
    setAuthBusy(false);
  }
}

async function loadCaptcha() {
  try {
    const response = await apiFetch("/api/auth/captcha", { auth: false });
    const data = await response.json();
    state.captchaKey = data.captcha_key || "";
    const image = data.captcha_image || "";
    if (!image) {
      throw new Error("后端没有返回验证码图片");
    }
    els.captchaImage.src = image.startsWith("data:") ? image : `data:image/svg+xml;base64,${image}`;
    els.captchaImage.alt = "验证码";
  } catch (error) {
    state.captchaKey = "";
    els.captchaImage.removeAttribute("src");
    els.captchaImage.alt = "验证码加载失败";
    setAuthMessage(`验证码加载失败：${error.message}`, "error");
  }
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
    const response = await apiFetch("/api/emotion/chat", {
      method: "POST",
      body: JSON.stringify({
        text,
        user_id: state.auth.userId,
        conversation_id: state.conversationId,
        history: state.messages.slice(-20).map((message) => ({
          role: message.role,
          content: message.content,
        })),
        metadata: { source: "frontend" },
      }),
    });

    const data = await response.json();
    state.conversationId = data.conversation_id || state.conversationId;
    removeLoadingMessage(loadingId);
    state.messages.push({
      role: "assistant",
      content: extractReply(data.chat_result),
      analysis: data.analysis_result,
    });
    updateInsights(data.analysis_result);
    maybeRefreshProfileAfterAnalysis(data.analysis_result, text);
    renderConversation(text);
    renderMessages(true);
    persistState();
    await loadConversations();
  } catch (error) {
    removeLoadingMessage(loadingId);
    state.messages.push({
      role: "assistant",
      content: `连接后端时遇到问题：${error.message}`,
      error: true,
    });
    setServiceStatus("连接失败", "offline");
    renderMessages(true);
    persistState();
  } finally {
    setBusy(false);
  }
}

function renderCurrentUser() {
  const username = state.auth?.username || "未登录";
  const initial = username.slice(0, 1).toUpperCase();
  els.currentUsername.textContent = username;
  els.currentUserRole.textContent = state.auth?.role || "user";
  if (!state.profileAvatarSvg) {
    els.userMiniAvatar.textContent = initial || "你";
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
  const avatar = isUser ? (state.auth?.username || "你").slice(0, 1).toUpperCase() : "E";
  return `
    <article class="message ${isUser ? "user" : "assistant"}">
      <div class="avatar" aria-hidden="true">${escapeHtml(avatar || "你")}</div>
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

function renderProfileAvatar(profile) {
  const username = state.auth?.username || "你";
  const dominantEmotion = normalizeValue(profile?.dominant_emotion, "等待生成");
  const traits = Array.isArray(profile?.personality_traits) ? profile.personality_traits : [];
  const summary = profile?.summary || profile?.communication_style || "聊天中出现明显情绪变化时，会自动刷新你的画像。";
  const distribution = profile?.emotion_distribution || {};
  const seed = `${username}-${dominantEmotion}-${traits.join("|")}`;
  const colors = emotionColors(dominantEmotion);
  const density = Math.min(9, Math.max(3, Object.keys(distribution).length + traits.length + 3));
  const initials = traits.length ? traits.slice(0, 2).map((item) => String(item).slice(0, 1)).join("") : username.slice(0, 1);

  els.profileEmotion.textContent = dominantEmotion;
  els.profileSummary.textContent = summary;
  els.traitList.innerHTML = traits.map((trait) => `<span class="chip">${escapeHtml(trait)}</span>`).join("");
  const svg = avatarSvg(seed, colors, initials || "你", density);
  state.profileAvatarSvg = svg;
  els.profileAvatar.innerHTML = svg;
  els.userMiniAvatar.innerHTML = svg;
}

function maybeRefreshProfileAfterAnalysis(analysis, text) {
  const signal = profileSignalFromAnalysis(analysis);
  if (!signal) {
    return;
  }

  if (!state.lastProfileSignal) {
    state.lastProfileSignal = signal;
    refreshProfileAvatar({ generate: true, text, reason: "first-analysis" });
    return;
  }

  if (hasSignificantEmotionChange(state.lastProfileSignal, signal)) {
    state.lastProfileSignal = signal;
    refreshProfileAvatar({ generate: true, text, reason: "emotion-change" });
  } else {
    state.lastProfileSignal = signal;
  }
}

async function refreshProfileAvatar({ generate = false, text = "", reason = "" } = {}) {
  if (!state.auth?.token) {
    return;
  }
  if (state.profileBusy) {
    state.pendingProfileGenerate = state.pendingProfileGenerate || generate;
    if (generate && text) {
      state.pendingProfileText = text;
    }
    return;
  }

  setProfileBusy(true);
  setProfileStatus(generate ? "检测到情绪变化，正在分析并更新画像..." : "正在同步画像...");
  try {
    if (generate && text) {
      await analyzeForProfile(text, reason);
    }

    const profileResponse = await apiFetch("/api/emotion/profile", { method: "POST" });
    const profile = await profileResponse.json();
    renderProfileAvatar(profile);

    if (generate) {
      const generatedResponse = await apiFetch("/api/emotion/profile/generate", { method: "POST" });
      const generated = await generatedResponse.json();
      renderProfileAvatar(generated);
    }

    setProfileStatus(generate ? "头像已根据情绪变化更新。" : "", generate ? "success" : "");
  } catch (error) {
    setProfileStatus(error.message || "画像同步失败", "error");
  } finally {
    setProfileBusy(false);
    if (state.pendingProfileGenerate) {
      const pendingText = state.pendingProfileText;
      state.pendingProfileGenerate = false;
      state.pendingProfileText = "";
      refreshProfileAvatar({ generate: true, text: pendingText, reason: "pending-emotion-change" });
    }
  }
}

async function analyzeForProfile(text, reason) {
  const trimmed = (text || "").trim();
  if (!trimmed) {
    return null;
  }

  const response = await apiFetch("/api/emotion/analyze", {
    method: "POST",
    body: JSON.stringify({
      id: makeClientId("profile"),
      user_id: state.auth.userId,
      text: trimmed,
      source: "profile_avatar",
      created_at: new Date().toISOString(),
      metadata: {
        trigger: reason || "emotion-change",
        conversation_id: state.conversationId,
      },
    }),
  });
  return response.json();
}

function profileSignalFromAnalysis(analysis) {
  const judge = analysis?.judge_result || {};
  const emotion = normalizeValue(judge.final_emotion, "");
  const intensity = asNumber(judge.final_intensity);
  if (!emotion && intensity == null) {
    return null;
  }
  return {
    emotion,
    intensity: intensity ?? 0,
    sarcasm: Boolean(judge.is_sarcasm),
    mixed: Boolean(judge.is_mixed),
  };
}

function hasSignificantEmotionChange(previous, next) {
  if (previous.emotion && next.emotion && previous.emotion !== next.emotion) {
    return true;
  }
  if (Math.abs((previous.intensity ?? 0) - (next.intensity ?? 0)) >= 30) {
    return true;
  }
  return previous.sarcasm !== next.sarcasm || previous.mixed !== next.mixed;
}

function avatarSvg(seed, colors, label, density) {
  const points = Array.from({ length: density }, (_, index) => {
    const x = 14 + pseudo(seed, index) * 72;
    const y = 14 + pseudo(seed, index + 17) * 72;
    const r = 8 + pseudo(seed, index + 31) * 15;
    const color = index % 2 === 0 ? colors.accent : colors.warm;
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="${color}" opacity="0.38" />`;
  }).join("");

  return `
    <svg viewBox="0 0 100 100" role="img" aria-label="画像头像">
      <rect width="100" height="100" rx="26" fill="${colors.base}" />
      ${points}
      <path d="M20 72 C34 46, 62 90, 82 28" fill="none" stroke="rgba(255,255,255,.58)" stroke-width="8" stroke-linecap="round" />
      <text x="50" y="58" text-anchor="middle" fill="white" font-size="26" font-weight="900">${escapeHtml(label.slice(0, 2))}</text>
    </svg>
  `;
}

function emotionColors(emotion) {
  const value = String(emotion || "");
  if (value.includes("焦") || value.includes("紧张")) {
    return { base: "#436fb0", accent: "#f0bc42", warm: "#d85f49" };
  }
  if (value.includes("怒") || value.includes("烦")) {
    return { base: "#d85f49", accent: "#202027", warm: "#f0bc42" };
  }
  if (value.includes("喜") || value.includes("开心")) {
    return { base: "#137a74", accent: "#f0bc42", warm: "#ffffff" };
  }
  if (value.includes("悲") || value.includes("低落")) {
    return { base: "#6750a4", accent: "#436fb0", warm: "#fffaf2" };
  }
  return { base: "#137a74", accent: "#436fb0", warm: "#f0bc42" };
}

function pseudo(seed, offset) {
  let hash = 2166136261 + offset;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return ((hash >>> 0) % 1000) / 1000;
}

function renderConversation(lastText) {
  const title = state.messages.find((message) => message.role === "user")?.content || lastText;
  els.conversationTitle.textContent = title ? clamp(title, 24) : "开始一次情绪对话";
  els.conversationIdLabel.textContent = state.conversationId || "尚未创建";
}

function renderConversationList() {
  if (state.conversationsStatus === "loading") {
    els.conversationList.innerHTML = `<p class="empty-history">正在加载历史会话...</p>`;
    return;
  }

  if (state.conversationsStatus === "error") {
    els.conversationList.innerHTML = `<p class="empty-history">历史会话加载失败，请确认登录状态和后端服务。</p>`;
    return;
  }

  if (!state.conversations.length) {
    els.conversationList.innerHTML = `<p class="empty-history">暂无历史会话，发送第一条消息后会出现在这里。</p>`;
    return;
  }

  els.conversationList.innerHTML = state.conversations
    .map((conversation) => {
      const isActive = conversation.id === state.conversationId;
      const title = conversation.title || "未命名会话";
      const updatedAt = conversation.updated_at ? formatDate(conversation.updated_at) : "";
      return `
        <button class="history-item ${isActive ? "active" : ""}" type="button" data-conversation-id="${escapeHtml(conversation.id)}">
          <strong>${escapeHtml(clamp(title, 18))}</strong>
          <span>${escapeHtml(updatedAt)}</span>
        </button>
      `;
    })
    .join("");
}

async function handleConversationListClick(event) {
  const button = event.target.closest("[data-conversation-id]");
  if (!button || state.busy) {
    return;
  }

  const conversation = state.conversations.find((item) => item.id === button.dataset.conversationId);
  if (!conversation) {
    return;
  }

  await loadConversationMessages(conversation.id);
}

async function checkHealth() {
  setServiceStatus("检查中", "");
  try {
    const response = await apiFetch("/api/emotion/health", { auth: false });
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

function setProfileBusy(isBusy) {
  state.profileBusy = isBusy;
}

function setAuthBusy(isBusy) {
  els.loginButton.disabled = isBusy;
  els.registerButton.disabled = isBusy;
  els.refreshCaptchaButton.disabled = isBusy;
}

async function loadConversations() {
  if (!state.auth?.token) {
    return;
  }

  state.conversationsStatus = "loading";
  renderConversationList();
  try {
    const response = await apiFetch("/api/emotion/conversations");
    const conversations = await response.json();
    state.conversations = Array.isArray(conversations) ? conversations : [];
    state.conversationsStatus = "ready";
    renderConversationList();

    if (state.conversationId && !state.messages.length) {
      await loadConversationMessages(state.conversationId, false);
    }
  } catch {
    state.conversations = [];
    state.conversationsStatus = "error";
    renderConversationList();
  }
}

async function loadConversationMessages(conversationId, refreshList = true) {
  setBusy(true);
  try {
    const response = await apiFetch(`/api/emotion/conversations/${encodeURIComponent(conversationId)}/messages`);
    const messages = await response.json();
    state.conversationId = conversationId;
    state.messages = Array.isArray(messages)
      ? messages.map((message) => ({
          role: message.role,
          content: message.content,
        }))
      : [];
    updateInsights(null);
    renderConversation();
    renderMessages(true);
    persistState();
    if (refreshList) {
      renderConversationList();
    }
  } catch {
    state.messages = [];
    renderMessages();
    setServiceStatus("历史加载失败", "offline");
  } finally {
    setBusy(false);
  }
}

function resetConversation() {
  state.conversationId = null;
  state.messages = [];
  els.rawAnalysis.textContent = "{}";
  updateInsights(null);
  renderConversation();
  renderMessages();
  persistState();
  loadConversations();
  els.messageInput.focus();
}

function saveApiBase() {
  state.apiBase = els.apiBaseInput.value.trim().replace(/\/$/, "") || DEFAULT_API_BASE;
  els.apiBaseInput.value = state.apiBase;
  localStorage.setItem(API_KEY, state.apiBase);
  checkHealth();
  if (state.auth?.token) {
    loadConversations();
  }
}

async function apiFetch(path, options = {}) {
  const { auth = true, headers = {}, ...fetchOptions } = options;
  const requestHeaders = { ...headers };
  if (fetchOptions.body && !requestHeaders["Content-Type"]) {
    requestHeaders["Content-Type"] = "application/json";
  }
  if (auth && state.auth?.token) {
    requestHeaders.Authorization = `Bearer ${state.auth.token}`;
  }

  const response = await fetch(`${state.apiBase}${path}`, {
    ...fetchOptions,
    headers: requestHeaders,
  });

  if (response.status === 401 && auth) {
    logout("登录已失效，请重新登录。");
    throw new Error("登录已失效，请重新登录。");
  }

  if (!response.ok) {
    throw new Error(await readError(response));
  }

  return response;
}

function logout(message = "") {
  state.auth = null;
  state.conversationId = null;
  state.conversations = [];
  state.conversationsStatus = "loading";
  state.messages = [];
  state.lastProfileSignal = null;
  state.profileAvatarSvg = "";
  state.pendingProfileGenerate = false;
  state.pendingProfileText = "";
  localStorage.removeItem(AUTH_KEY);
  localStorage.removeItem(STORAGE_KEY);
  renderRoute();
  if (message) {
    setAuthMessage(message, "error");
  }
}

function loadAuth() {
  try {
    const auth = JSON.parse(localStorage.getItem(AUTH_KEY) || "null");
    return auth?.token ? auth : null;
  } catch {
    return null;
  }
}

function restoreState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    state.conversationId = saved.conversationId || null;
    state.messages = [];
    state.conversations = [];
    state.conversationsStatus = "loading";
    state.lastProfileSignal = null;
    state.profileAvatarSvg = "";
    state.pendingProfileGenerate = false;
    state.pendingProfileText = "";
  } catch {
    state.conversationId = null;
    state.messages = [];
    state.conversations = [];
    state.conversationsStatus = "loading";
    state.lastProfileSignal = null;
    state.profileAvatarSvg = "";
    state.pendingProfileGenerate = false;
    state.pendingProfileText = "";
  }
}

function persistState() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      conversationId: state.conversationId,
    })
  );
}

function setAuthMessage(message, type = "") {
  els.authMessage.textContent = message;
  els.authMessage.className = `auth-message ${type}`.trim();
}

function setProfileStatus(message, type = "") {
  els.profileStatus.textContent = message;
  els.profileStatus.className = `profile-status ${type}`.trim();
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
    return data.detail || data.message || data.error || JSON.stringify(data);
  } catch {
    return response.statusText;
  }
}

function makeClientId(prefix) {
  if (window.crypto?.randomUUID) {
    return `${prefix}_${window.crypto.randomUUID()}`;
  }
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
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

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
