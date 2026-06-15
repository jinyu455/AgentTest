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
  adminUsers: [],
  adminAllConversations: [],
  adminTargetUserId: "",
  adminSearchOpen: false,
  adminActiveTab: "style",
  busy: false,
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
    backToChatButton: document.querySelector("#backToChatButton"),
    captchaCode: document.querySelector("#captchaCode"),
    captchaImage: document.querySelector("#captchaImage"),
    chatForm: document.querySelector("#chatForm"),
    faceRatingInput: document.querySelector("#face-rating"),
    chatPanel: document.querySelector(".chat-panel"),
    confidenceValue: document.querySelector("#confidenceValue"),
    conversationIdLabel: document.querySelector("#conversationIdLabel"),
    conversationList: document.querySelector("#conversationList"),
    conversationTitle: document.querySelector("#conversationTitle"),
    currentUsername: document.querySelector("#currentUsername"),
    currentUserRole: document.querySelector("#currentUserRole"),
    emotionIntensity: document.querySelector("#emotionIntensity"),
    emotionMeta: document.querySelector("#emotionMeta"),
    emotionMeter: document.querySelector("#emotionMeter"),
    emotionComboChart: document.querySelector("#emotionComboChart"),
    emotionDistributionList: document.querySelector("#emotionDistributionList"),
    historyAvgConfidence: document.querySelector("#historyAvgConfidence"),
    historyAvgIntensity: document.querySelector("#historyAvgIntensity"),
    historyDominantEmotion: document.querySelector("#historyDominantEmotion"),
    historyPage: document.querySelector("#historyPage"),
    historyStatus: document.querySelector("#historyStatus"),
    historyTotalRecords: document.querySelector("#historyTotalRecords"),
    activityPatternList: document.querySelector("#activityPatternList"),
    adminActivityPatternList: document.querySelector("#adminActivityPatternList"),
    adminAvgIntensity: document.querySelector("#adminAvgIntensity"),
    adminComboChart: document.querySelector("#adminComboChart"),
    adminCommunicationStyle: document.querySelector("#adminCommunicationStyle"),
    adminDashboard: document.querySelector("#adminDashboard"),
    adminDominantEmotion: document.querySelector("#adminDominantEmotion"),
    adminEmotionDistributionList: document.querySelector("#adminEmotionDistributionList"),
    adminEmotionalPatterns: document.querySelector("#adminEmotionalPatterns"),
    adminMbti: document.querySelector("#adminMbti"),
    adminStatus: document.querySelector("#adminStatus"),
    adminTabs: document.querySelectorAll("[data-admin-tab]"),
    adminTabPanels: document.querySelectorAll("[data-admin-panel]"),
    adminSummary: document.querySelector("#adminSummary"),
    adminTargetLabel: document.querySelector("#adminTargetLabel"),
    adminTotalRecords: document.querySelector("#adminTotalRecords"),
    adminTraits: document.querySelector("#adminTraits"),
    adminUserSearchCard: document.querySelector("#adminUserSearchCard"),
    adminUserList: document.querySelector("#adminUserList"),
    adminUserSearch: document.querySelector("#adminUserSearch"),
    adminUserSearchButton: document.querySelector("#adminUserSearchButton"),
    adminMainContent: document.querySelector(".admin-main-content"),
    adminMainTitle: document.querySelector("#adminMainTitle"),
    adminSidebarTabs: document.querySelector(".admin-sidebar-tabs"),
    insightPanel: document.querySelector(".insight-panel"),
    loginButton: document.querySelector("#loginButton"),
    loginForm: document.querySelector("#loginForm"),
    loginPassword: document.querySelector("#loginPassword"),
    loginUsername: document.querySelector("#loginUsername"),
    logoutButton: document.querySelector("#logoutButton"),
    messageInput: document.querySelector("#messageInput"),
    messageList: document.querySelector("#messageList"),
    mixedValue: document.querySelector("#mixedValue"),
    newChatButton: document.querySelector("#newChatButton"),
    openHistoryButton: document.querySelector("#openHistoryButton"),
    primaryEmotion: document.querySelector("#primaryEmotion"),
    refreshHistoryButton: document.querySelector("#refreshHistoryButton"),
    refreshAdminButton: document.querySelector("#refreshAdminButton"),
    refreshCaptchaButton: document.querySelector("#refreshCaptchaButton"),
    registerButton: document.querySelector("#registerButton"),
    registerForm: document.querySelector("#registerForm"),
    registerPassword: document.querySelector("#registerPassword"),
    registerUsername: document.querySelector("#registerUsername"),
    sarcasmValue: document.querySelector("#sarcasmValue"),
    secondaryEmotion: document.querySelector("#secondaryEmotion"),
    sendButton: document.querySelector("#sendButton"),
    sidebar: document.querySelector(".sidebar"),
    userMiniAvatar: document.querySelector("#userMiniAvatar"),
    userInsightContent: document.querySelector(".user-insight-content"),
  };
}

function init() {
  restoreState();
  bindEvents();
  renderRoute();
}

function bindEvents() {
  els.authTabs.forEach((tab) => {
    tab.addEventListener("click", () => switchAuthTab(tab.dataset.authTab));
  });
  initFaceRating();
  els.loginForm.addEventListener("submit", handleLogin);
  els.registerForm.addEventListener("submit", handleRegister);
  els.refreshCaptchaButton.addEventListener("click", loadCaptcha);
  els.logoutButton.addEventListener("click", () => logout());
  els.adminUserList.addEventListener("click", handleAdminUserClick);
  els.adminTabs.forEach((tab) => {
    tab.addEventListener("click", () => switchAdminTab(tab.dataset.adminTab));
  });
  els.adminUserSearch.addEventListener("input", () => {
    state.adminSearchOpen = true;
    renderAdminUsers();
  });
  els.adminUserSearch.addEventListener("search", () => {
    state.adminSearchOpen = true;
    renderAdminUsers();
  });
  els.adminUserSearch.addEventListener("focus", () => {
    state.adminSearchOpen = true;
    renderAdminUsers();
  });
  els.adminUserSearch.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      selectFirstVisibleAdminUser();
    }
  });
  els.adminUserSearchButton.addEventListener("click", () => {
    state.adminSearchOpen = true;
    renderAdminUsers();
    selectFirstVisibleAdminUser();
  });
  document.addEventListener("click", (event) => {
    if (!els.adminUserSearchCard?.contains(event.target)) {
      state.adminSearchOpen = false;
      renderAdminUsers();
    }
  });
  els.chatForm.addEventListener("submit", handleSubmit);
  els.newChatButton.addEventListener("click", () => {
    if (isAdminMode()) {
      state.adminSearchOpen = true;
      renderAdminUsers();
      els.adminUserSearch.focus();
      return;
    }
    resetConversation();
  });
  els.conversationList.addEventListener("click", handleConversationListClick);
  els.openHistoryButton.addEventListener("click", openHistoryPage);
  els.backToChatButton.addEventListener("click", closeHistoryPage);
  els.refreshHistoryButton.addEventListener("click", loadEmotionHistory);
  els.refreshAdminButton.addEventListener("click", () => loadAdminDashboard({ force: true }));
  els.messageInput.addEventListener("input", autoResizeInput);
  els.messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      els.chatForm.requestSubmit();
    }
  });
}

function initFaceRating() {
  if (!els.faceRatingInput) {
    return;
  }
  els.faceRatingInput.closest("form")?.addEventListener("submit", (event) => event.preventDefault());
  const rating = new FaceRating(els.faceRatingInput);
  rating.update();
}

class FaceRating {
  constructor(input) {
    this.input = input;
    this.face = input.previousElementSibling;
    this.input.addEventListener("input", this.update.bind(this));
  }

  update(event) {
    const value = Number(event?.target?.value ?? this.input.value ?? this.input.defaultValue);
    const min = Number(this.input.min || 0);
    const max = Number(this.input.max || 100);
    const percentRaw = ((value - min) / (max - min)) * 100;
    const percent = Math.max(0, Math.min(100, Number(percentRaw.toFixed(2))));
    const rating = percent / 100;

    this.input.style.setProperty("--percent", `${percent}%`);
    this.input.style.setProperty("--input-hue", Math.round(120 * rating));
    this.face.style.setProperty("--rating", rating.toFixed(3));
    this.face.style.setProperty("--face-hue1", Math.round(120 * rating));
    this.face.style.setProperty("--face-hue2", Math.round((120 * rating + 330) % 360));
    this.face.style.setProperty("--eye-tilt", `${(-9 + rating * 18).toFixed(2)}deg`);
    this.face.style.setProperty("--eye-lift", `${(5 - rating * 10).toFixed(2)}px`);
    const duration = 1;
    const delay = -(duration * 0.99 * rating).toFixed(3);
    ["mouth-lower", "mouth-upper"].forEach((part) => {
      this.face.querySelector(`[data-${part}]`)?.style.setProperty("--mouth-delay", `${delay}s`);
    });

    const faces = ["低落表情", "有点低落", "平静表情", "有点开心", "开心表情"];
    let faceIndex = Math.floor((faces.length * percent) / 100);
    if (faceIndex === faces.length) {
      faceIndex -= 1;
    }
    this.face.setAttribute("aria-label", faces[faceIndex]);
  }
}

function isAdminMode() {
  return state.auth?.role === "admin";
}

function selectedAdminUser() {
  return state.adminUsers.find((user) => user.id === state.adminTargetUserId) || null;
}

function renderRoute() {
  const isAuthed = Boolean(state.auth?.token);
  const isAdmin = state.auth?.role === "admin";
  els.authView.hidden = isAuthed;
  els.appShell.hidden = !isAuthed;
  els.appShell.classList.toggle("admin-mode", isAuthed && isAdmin);
  closeHistoryPage({ loadChat: false });
  els.adminDashboard.hidden = true;

  if (!isAuthed) {
    switchAuthTab("login");
    loadCaptcha();
    return;
  }

  if (isAdmin) {
    els.sidebar.hidden = false;
    els.chatPanel.hidden = false;
    els.insightPanel.hidden = true;
    els.userInsightContent.hidden = true;
    els.adminMainContent.hidden = false;
    els.adminSidebarTabs.hidden = false;
    els.adminUserSearchCard.hidden = false;
    els.newChatButton.innerHTML = `
      <span aria-hidden="true">
        <svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7" /><path d="m16.5 16.5 4 4" /></svg>
      </span>
      搜索聊天
    `;
    renderCurrentUser();
    renderConversation();
    renderMessages();
    renderConversationList();
    switchAdminTab(state.adminActiveTab || "style");
    showAdminDashboardView(state.adminActiveTab || "style");
    loadAdminUsers();
    return;
  }

  els.sidebar.hidden = false;
  els.chatPanel.hidden = false;
  els.insightPanel.hidden = false;
  els.userInsightContent.hidden = false;
  els.adminMainContent.hidden = true;
  els.adminSidebarTabs.hidden = true;
  els.messageList.hidden = false;
  els.chatPanel.classList.remove("showing-admin-dashboard");
  els.adminUserSearchCard.hidden = true;
  els.newChatButton.innerHTML = `<span aria-hidden="true">+</span>新会话`;
  renderCurrentUser();
  renderMessages();
  renderConversation();
  renderConversationList();
  loadConversations();
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
    window.location.reload();
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
    renderMessages(true);
    persistState();
  } finally {
    setBusy(false);
  }
}

function renderCurrentUser() {
  const username = state.auth?.username || "未登录";
  els.currentUsername.textContent = username;
  els.currentUserRole.textContent = state.auth?.role || "user";
  els.userMiniAvatar.innerHTML = fixedUserAvatarMarkup(state.auth?.role);
}

function fixedUserAvatarMarkup(role = "user") {
  const isAdmin = role === "admin";
  const src = isAdmin ? "./image/admin_avatar.png" : "./image/user_avatar.png";
  const label = isAdmin ? "管理员头像" : "用户头像";
  return `<img src="${src}" alt="${label}" />`;
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
  const avatar = isUser ? fixedUserAvatarMarkup("user") : "E";
  return `
    <article class="message ${isUser ? "user" : "assistant"}">
      <div class="avatar" aria-hidden="true">${isUser ? avatar : escapeHtml(avatar)}</div>
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
}

function renderConversation(lastText) {
  const title = state.messages.find((message) => message.role === "user")?.content || lastText;
  if (isAdminMode()) {
    const user = selectedAdminUser();
    if (!user) {
      els.conversationTitle.textContent = "选择用户后查看聊天";
      return;
    }
    els.conversationTitle.textContent = title ? clamp(title, 28) : `正在查看 ${user.name} 的会话`;
    return;
  }
  els.conversationTitle.textContent = title ? clamp(title, 24) : "开始一次情绪对话";
  if (els.conversationIdLabel) {
    els.conversationIdLabel.textContent = "";
  }
}

function renderConversationList() {
  if (state.conversationsStatus === "loading") {
    els.conversationList.innerHTML = `<p class="empty-history">正在加载会话...</p>`;
    return;
  }

  if (state.conversationsStatus === "error") {
    els.conversationList.innerHTML = `<p class="empty-history">历史会话加载失败，请确认登录状态和网络连接。</p>`;
    return;
  }

  if (!state.conversations.length) {
    const emptyText = isAdminMode() ? "该用户暂无聊天记录。" : "暂无历史会话，发送第一条消息后会出现在这里。";
    els.conversationList.innerHTML = `<p class="empty-history">${emptyText}</p>`;
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

function openHistoryPage() {
  els.chatPanel.hidden = true;
  els.insightPanel.hidden = true;
  els.historyPage.hidden = false;
  loadEmotionHistory();
}

function closeHistoryPage(options = {}) {
  const { loadChat = true } = options;
  if (!els.historyPage) {
    return;
  }
  els.historyPage.hidden = true;
  els.chatPanel.hidden = false;
  els.insightPanel.hidden = false;
  if (loadChat) {
    renderMessages();
  }
}

async function loadEmotionHistory() {
  setHistoryStatus("正在读取历史情绪数据...");
  els.refreshHistoryButton.disabled = true;
  try {
    const response = await apiFetch("/api/emotion/profile", { method: "POST" });
    const profile = await response.json();
    renderEmotionHistory(profile);
    setHistoryStatus("已根据当前账号的历史情绪记录更新。", "success");
  } catch (error) {
    setHistoryStatus(error.message || "历史情绪数据加载失败", "error");
    renderEmotionHistory(null);
  } finally {
    els.refreshHistoryButton.disabled = false;
  }
}

function renderEmotionHistory(profile) {
  const total = asNumber(profile?.total_records) ?? 0;
  const avgIntensity = asNumber(profile?.avg_intensity);
  const avgConfidence = asNumber(profile?.avg_confidence);
  els.historyTotalRecords.textContent = String(total);
  els.historyDominantEmotion.textContent = normalizeValue(profile?.dominant_emotion, "--");
  els.historyAvgIntensity.textContent = avgIntensity != null ? avgIntensity.toFixed(1) : "--";
  els.historyAvgConfidence.textContent = avgConfidence != null ? toPercent(avgConfidence) : "--";
  renderComboChart(profile);
  renderKeyValueBars(els.emotionDistributionList, profile?.emotion_distribution, "暂无情绪分布");
  renderKeyValueBars(els.activityPatternList, profile?.activity_pattern, "暂无活跃时段");
}

async function loadAdminDashboard({ force = false } = {}) {
  if (!state.adminTargetUserId) {
    renderAdminDashboard(null);
    setAdminStatus("请选择一个用户后再生成画像。", "error");
    return;
  }
  setAdminStatus("正在生成用户画像...");
  els.refreshAdminButton.disabled = true;
  try {
    const forceParam = force ? "&force=true" : "";
    const response = await apiFetch(
      `/api/emotion/profile/generate?target_user_id=${encodeURIComponent(state.adminTargetUserId)}${forceParam}`,
      { method: "POST" }
    );
    const profile = await response.json();
    renderAdminDashboard(profile);
    setAdminStatus("");
  } catch (error) {
    renderAdminDashboard(null);
    setAdminStatus(error.message || "用户画像生成失败", "error");
  } finally {
    els.refreshAdminButton.disabled = false;
  }
}

async function loadAdminUsers() {
  setAdminStatus("正在读取用户列表...");
  els.adminUserList.innerHTML = `<p class="chart-empty">正在加载可见用户...</p>`;
  try {
    const response = await apiFetch("/api/emotion/conversations");
    const conversations = await response.json();
    state.adminAllConversations = Array.isArray(conversations) ? conversations : [];
    const users = Array.isArray(conversations) ? buildAdminUsers(conversations) : [];
    state.adminUsers = users;
    if (!users.length) {
      state.adminTargetUserId = "";
      state.conversations = [];
      state.conversationsStatus = "ready";
      renderAdminUsers();
      renderConversationList();
      renderAdminDashboard(null);
      setAdminStatus("暂无可生成画像的用户。需要普通用户先产生会话和情绪记录。", "error");
      return;
    }
    if (!state.adminTargetUserId || !users.some((user) => user.id === state.adminTargetUserId)) {
      state.adminTargetUserId = users[0].id;
    }
    renderAdminUsers();
    await loadAdminConversations();
    await loadAdminDashboard();
  } catch (error) {
    state.adminAllConversations = [];
    state.adminUsers = [];
    state.adminTargetUserId = "";
    state.conversations = [];
    state.conversationsStatus = "error";
    renderAdminUsers();
    renderConversationList();
    renderAdminDashboard(null);
    setAdminStatus(error.message || "用户列表加载失败", "error");
  }
}

function buildAdminUsers(conversations) {
  const users = new Map();
  conversations.forEach((conversation) => {
    const userId = conversation.user_id;
    if (!userId) {
      return;
    }
    const existing = users.get(userId) || {
      id: userId,
      name: conversation.username || conversation.user_name || userId,
      count: 0,
      latest: "",
      conversations: [],
    };
    existing.name = conversation.username || conversation.user_name || existing.name || userId;
    existing.count += 1;
    existing.conversations.push(conversation);
    if (!existing.latest || String(conversation.updated_at || "") > existing.latest) {
      existing.latest = conversation.updated_at || "";
    }
    users.set(userId, existing);
  });
  return [...users.values()].sort((a, b) => String(b.latest).localeCompare(String(a.latest)));
}

function renderAdminUsers() {
  els.adminUserSearchCard.classList.toggle("is-open", state.adminSearchOpen);
  if (!state.adminUsers.length) {
    els.adminTargetLabel.textContent = "暂无可选用户";
    els.adminUserList.innerHTML = `<p class="chart-empty">还没有普通用户会话。</p>`;
    return;
  }
  const current = state.adminUsers.find((user) => user.id === state.adminTargetUserId);
  els.adminTargetLabel.textContent = current ? current.name : "请选择一个用户";
  const visibleUsers = visibleAdminUsers();
  if (!visibleUsers.length) {
    els.adminUserList.innerHTML = `<p class="chart-empty">没有匹配的用户。</p>`;
    return;
  }
  els.adminUserList.innerHTML = visibleUsers
    .map((user) => `
      <button class="admin-user-item ${user.id === state.adminTargetUserId ? "active" : ""}" type="button" data-admin-user-id="${escapeHtml(user.id)}">
        <strong>${escapeHtml(user.name)}</strong>
        <span>${user.count} 个会话 · ${escapeHtml(formatDate(user.latest) || "未知时间")}</span>
      </button>
    `)
    .join("");
}

function visibleAdminUsers() {
  const query = els.adminUserSearch.value.trim().toLowerCase();
  if (!query) {
    return state.adminUsers.slice(0, 6);
  }
  return state.adminUsers.filter((user) => {
    const name = String(user.name || "").toLowerCase();
    const id = String(user.id || "").toLowerCase();
    return name.includes(query) || id.includes(query);
  });
}

async function selectFirstVisibleAdminUser() {
  const [user] = visibleAdminUsers();
  if (!user) {
    renderAdminUsers();
    return;
  }
  await selectAdminUser(user.id);
}

async function handleAdminUserClick(event) {
  const button = event.target.closest("[data-admin-user-id]");
  if (!button) {
    return;
  }
  await selectAdminUser(button.dataset.adminUserId);
}

async function selectAdminUser(userId) {
  const sameUser = userId === state.adminTargetUserId;
  state.adminTargetUserId = userId;
  state.adminSearchOpen = false;
  els.adminUserSearch.value = "";
  renderAdminUsers();
  if (sameUser) {
    return;
  }
  state.conversationId = null;
  state.messages = [];
  renderConversation();
  renderMessages();
  showAdminDashboardView(state.adminActiveTab || "style");
  await loadAdminConversations();
  await loadAdminDashboard();
}

async function loadAdminConversations() {
  if (!state.adminTargetUserId) {
    state.conversations = [];
    state.conversationsStatus = "ready";
    renderConversationList();
    return;
  }

  state.conversationsStatus = "loading";
  renderConversationList();
  try {
    const response = await apiFetch(`/api/emotion/conversations?target_user_id=${encodeURIComponent(state.adminTargetUserId)}`);
    const conversations = await response.json();
    state.conversations = Array.isArray(conversations) ? conversations : [];
    state.conversationsStatus = "ready";
    renderConversationList();
  } catch {
    state.conversations = [];
    state.conversationsStatus = "error";
    renderConversationList();
  }
}

function switchAdminTab(tabName = "style") {
  state.adminActiveTab = tabName;
  els.adminTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.adminTab === tabName));
  els.adminTabPanels.forEach((panel) => {
    panel.hidden = panel.dataset.adminPanel !== tabName;
  });
  if (els.adminMainTitle) {
    els.adminMainTitle.textContent = tabName === "trend" ? "情绪变化" : "用户画像";
  }
  if (isAdminMode()) {
    showAdminDashboardView(tabName);
  }
}

function showAdminDashboardView(tabName = "style") {
  if (!isAdminMode()) {
    return;
  }
  state.adminActiveTab = tabName;
  els.chatPanel.classList.add("showing-admin-dashboard");
  els.messageList.hidden = true;
  els.adminMainContent.hidden = false;
  els.adminTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.adminTab === tabName));
  els.adminTabPanels.forEach((panel) => {
    panel.hidden = panel.dataset.adminPanel !== tabName;
  });
  if (els.adminMainTitle) {
    els.adminMainTitle.textContent = tabName === "trend" ? "情绪变化" : "用户画像";
  }
}

function showAdminConversationView() {
  if (!isAdminMode()) {
    return;
  }
  els.adminMainContent.hidden = true;
  els.messageList.hidden = false;
  els.chatPanel.classList.remove("showing-admin-dashboard");
}

function renderAdminDashboard(profile) {
  const total = asNumber(profile?.total_records) ?? 0;
  const avgIntensity = asNumber(profile?.avg_intensity);
  els.adminTotalRecords.textContent = String(selectedAdminUser()?.count ?? total);
  els.adminDominantEmotion.textContent = normalizeValue(profile?.dominant_emotion, "--");
  els.adminAvgIntensity.textContent = avgIntensity != null ? avgIntensity.toFixed(1) : "--";
  els.adminMbti.textContent = normalizeValue(profile?.mbti, "--");
  els.adminSummary.textContent = profile?.summary || "暂无画像摘要";
  els.adminCommunicationStyle.textContent = profile?.communication_style || "--";
  els.adminEmotionalPatterns.textContent = profile?.emotional_patterns || "--";
  const traits = Array.isArray(profile?.personality_traits) ? profile.personality_traits : [];
  els.adminTraits.innerHTML = traits.length
    ? traits.map((trait) => `<span class="chip">${escapeHtml(trait)}</span>`).join("")
    : `<span class="chip">暂无特征</span>`;
  renderComboChart(profile, els.adminComboChart);
  renderKeyValueBars(els.adminEmotionDistributionList, profile?.emotion_distribution, "暂无情绪分布");
  renderKeyValueBars(els.adminActivityPatternList, profile?.activity_pattern, "暂无活跃时段");
}

function renderComboChart(profile, container = els.emotionComboChart) {
  const timeline = Array.isArray(profile?.timeline?.data_points) ? profile.timeline.data_points : [];
  const records = timeline
    .map((point) => ({
      createdAt: point.created_at || "",
      emotion: point.emotion || "",
      intensity: Math.max(0, Math.min(100, Number(point.intensity) || 0)),
    }))
    .filter((point) => point.createdAt || point.emotion || point.intensity > 0);

  if (!records.length) {
    container.classList.remove("is-scrollable-chart");
    container.innerHTML = `<p class="chart-empty">暂无可视化数据，完成几轮对话后这里会出现趋势图。</p>`;
    return;
  }

  const height = 500;
  const padding = { top: 62, right: 96, bottom: 86, left: 56 };
  const pointSpacing = 132;
  const barWidth = 42;
  const minVisiblePoints = 8;
  const plotSlots = Math.max(records.length, minVisiblePoints);
  const chartWidth = plotSlots * pointSpacing;
  const width = padding.left + chartWidth + padding.right;
  const shouldScroll = width > (container.clientWidth || 0);
  container.classList.toggle("is-scrollable-chart", shouldScroll);
  const svgWidthStyle = `width: ${width}px; min-width: ${width}px;`;
  const chartHeight = height - padding.top - padding.bottom;
  const baseline = padding.top + chartHeight;
  const points = records.map((record, index) => {
    const x = padding.left + pointSpacing * index + pointSpacing / 2;
    const y = baseline - (record.intensity / 100) * chartHeight;
    return { ...record, x, y };
  });

  const bars = points
    .map((point, index) => {
      const barHeight = baseline - point.y;
      const x = point.x - barWidth / 2;
      return `
        <rect class="chart-bar" x="${x.toFixed(1)}" y="${point.y.toFixed(1)}" width="${barWidth.toFixed(1)}" height="${barHeight.toFixed(1)}" rx="7">
          <title>${escapeHtml(recordTooltip(point, index))}</title>
        </rect>
      `;
    })
    .join("");

  const linePath = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(" ");
  const circles = points
    .map((point, index) => `
      <circle cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="5.5" fill="var(--accent)">
        <title>${escapeHtml(recordTooltip(point, index))}</title>
      </circle>
    `)
    .join("");
  const xLabels = points
    .map((point, index) => {
      const step = Math.max(1, Math.ceil(points.length / 6));
      const shouldShow = points.length <= 8 || index === 0 || index === points.length - 1 || index % step === 0;
      return shouldShow
        ? `<text x="${point.x.toFixed(1)}" y="${height - 20}" text-anchor="middle">${escapeHtml(formatShortDate(point.createdAt, index + 1))}</text>`
        : "";
    })
    .join("");
  const pointLabels = points
    .map((point) => `<text x="${point.x.toFixed(1)}" y="${Math.max(22, point.y - 12).toFixed(1)}" text-anchor="middle">${escapeHtml(point.emotion || String(point.intensity))}</text>`)
    .join("");
  const grid = [0, 50, 100]
    .map((value) => {
      const y = baseline - (value / 100) * chartHeight;
      return `
        <line class="grid-line" x1="${padding.left}" y1="${y.toFixed(1)}" x2="${width - padding.right}" y2="${y.toFixed(1)}" />
        <text x="${padding.left - 12}" y="${(y + 4).toFixed(1)}" text-anchor="end">${value}</text>
      `;
    })
    .join("");

  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" style="${svgWidthStyle}" role="img" aria-label="情绪强度趋势和强度分布">
      ${grid}
      <line x1="${padding.left}" y1="${baseline}" x2="${width - padding.right}" y2="${baseline}" />
      <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${baseline}" />
      <g class="chart-legend">
        <rect x="${padding.left}" y="12" width="18" height="10" rx="4" />
        <text x="${padding.left + 26}" y="21">柱：单次强度</text>
        <line x1="${padding.left + 132}" y1="17" x2="${padding.left + 158}" y2="17" />
        <text x="${padding.left + 166}" y="21">线：强度走势</text>
      </g>
      ${bars}
      <path d="${linePath}" fill="none" stroke="var(--coral)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
      ${circles}
      ${pointLabels}
      ${xLabels}
    </svg>
  `;
}

function renderKeyValueBars(container, source, emptyText) {
  const entries = Object.entries(source || {}).filter(([, value]) => Number(value) > 0);
  if (!entries.length) {
    container.innerHTML = `<p class="chart-empty">${emptyText}</p>`;
    return;
  }
  const maxValue = Math.max(...entries.map(([, value]) => Number(value) || 0), 1);
  container.innerHTML = entries
    .map(([label, value]) => {
      const width = Math.max(4, ((Number(value) || 0) / maxValue) * 100);
      const display = Number(value) <= 1 ? `${Math.round(Number(value) * 100)}%` : String(value);
      return `
        <div class="kv-bar">
          <span>${escapeHtml(label)}</span>
          <div><i style="width: ${width}%"></i></div>
          <strong>${escapeHtml(display)}</strong>
        </div>
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

function setBusy(isBusy) {
  state.busy = isBusy;
  if (els.sendButton) {
    els.sendButton.disabled = isBusy;
  }
  if (els.messageInput) {
    els.messageInput.disabled = isBusy || isAdminMode();
  }
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

    if (!state.messages.length) {
      const savedConversation = state.conversations.find((conversation) => conversation.id === state.conversationId);
      const conversationToOpen = savedConversation || state.conversations[0];
      if (conversationToOpen) {
        await loadConversationMessages(conversationToOpen.id, false);
      }
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
    const targetParam =
      isAdminMode() && state.adminTargetUserId ? `?target_user_id=${encodeURIComponent(state.adminTargetUserId)}` : "";
    const response = await apiFetch(`/api/emotion/conversations/${encodeURIComponent(conversationId)}/messages${targetParam}`);
    const messages = await response.json();
    state.conversationId = conversationId;
    state.messages = Array.isArray(messages)
      ? messages.map((message) => ({
          role: message.role,
          content: message.content,
        }))
      : [];
    if (!isAdminMode()) {
      updateInsights(null);
    }
    renderConversation();
    renderMessages(true);
    if (isAdminMode()) {
      showAdminConversationView();
    }
    if (!isAdminMode()) {
      persistState();
    }
    if (refreshList) {
      renderConversationList();
    }
  } catch {
    state.messages = [];
    renderMessages();
  } finally {
    setBusy(false);
  }
}

function resetConversation() {
  state.conversationId = null;
  state.messages = [];
  updateInsights(null);
  renderConversation();
  renderMessages();
  persistState();
  loadConversations();
  els.messageInput.focus();
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
  const authMessage = typeof message === "string" ? message : "";
  state.auth = null;
  state.conversationId = null;
  state.conversations = [];
  state.conversationsStatus = "loading";
  state.messages = [];
  state.lastProfileSignal = null;
  localStorage.removeItem(AUTH_KEY);
  renderRoute();
  if (authMessage) {
    setAuthMessage(authMessage, "error");
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
    const saved = JSON.parse(localStorage.getItem(currentStorageKey()) || "{}");
    state.conversationId = saved.conversationId || null;
    state.messages = [];
    state.conversations = [];
    state.conversationsStatus = "loading";
    state.lastProfileSignal = null;
  } catch {
    state.conversationId = null;
    state.messages = [];
    state.conversations = [];
    state.conversationsStatus = "loading";
    state.lastProfileSignal = null;
  }
}

function persistState() {
  localStorage.setItem(
    currentStorageKey(),
    JSON.stringify({
      conversationId: state.conversationId,
    })
  );
}

function currentStorageKey() {
  const userId = state.auth?.userId || "guest";
  return `${STORAGE_KEY}.${userId}`;
}

function setAuthMessage(message, type = "") {
  els.authMessage.textContent = message;
  els.authMessage.className = `auth-message ${type}`.trim();
}

function setHistoryStatus(message, type = "") {
  els.historyStatus.textContent = message;
  els.historyStatus.className = `history-status ${type}`.trim();
}

function setAdminStatus(message, type = "") {
  els.adminStatus.textContent = message;
  els.adminStatus.className = `history-status ${type}`.trim();
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

function formatShortDate(value, fallbackIndex) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return `#${fallbackIndex}`;
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}

function recordTooltip(point, index) {
  return `第 ${index + 1} 条｜${formatDate(point.createdAt) || "未知时间"}｜${point.emotion || "未知"}｜${point.intensity} 分`;
}

function shortId(value) {
  const text = String(value || "");
  return text.length > 10 ? `${text.slice(0, 8)}...` : text;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
