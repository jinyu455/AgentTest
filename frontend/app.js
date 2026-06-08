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
  adminTargetUserId: "",
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
    adminLogoutButton: document.querySelector("#adminLogoutButton"),
    adminMbti: document.querySelector("#adminMbti"),
    adminStatus: document.querySelector("#adminStatus"),
    adminSummary: document.querySelector("#adminSummary"),
    adminTargetLabel: document.querySelector("#adminTargetLabel"),
    adminTotalRecords: document.querySelector("#adminTotalRecords"),
    adminTraits: document.querySelector("#adminTraits"),
    adminUserList: document.querySelector("#adminUserList"),
    adminUserSearch: document.querySelector("#adminUserSearch"),
    adminUserSearchButton: document.querySelector("#adminUserSearchButton"),
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
    serviceStatus: document.querySelector("#serviceStatus"),
    statusDot: document.querySelector("#statusDot"),
    userMiniAvatar: document.querySelector("#userMiniAvatar"),
  };
}

function init() {
  restoreState();
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
  els.adminLogoutButton.addEventListener("click", logout);
  els.adminUserList.addEventListener("click", handleAdminUserClick);
  els.adminUserSearch.addEventListener("input", renderAdminUsers);
  els.adminUserSearch.addEventListener("search", renderAdminUsers);
  els.adminUserSearch.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      renderAdminUsers();
    }
  });
  els.adminUserSearchButton.addEventListener("click", renderAdminUsers);
  els.chatForm.addEventListener("submit", handleSubmit);
  els.newChatButton.addEventListener("click", resetConversation);
  els.conversationList.addEventListener("click", handleConversationListClick);
  els.openHistoryButton.addEventListener("click", openHistoryPage);
  els.backToChatButton.addEventListener("click", closeHistoryPage);
  els.refreshHistoryButton.addEventListener("click", loadEmotionHistory);
  els.refreshAdminButton.addEventListener("click", loadAdminDashboard);
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
  const isAdmin = state.auth?.role === "admin";
  els.authView.hidden = isAuthed;
  els.appShell.hidden = !isAuthed;
  closeHistoryPage({ loadChat: false });
  els.adminDashboard.hidden = true;

  if (!isAuthed) {
    switchAuthTab("login");
    loadCaptcha();
    return;
  }

  if (isAdmin) {
    els.chatPanel.hidden = true;
    els.insightPanel.hidden = true;
    document.querySelector(".sidebar").hidden = true;
    els.adminDashboard.hidden = false;
    loadAdminUsers();
    return;
  }

  document.querySelector(".sidebar").hidden = false;
  els.chatPanel.hidden = false;
  els.insightPanel.hidden = false;
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
    setAuthMessage("");
    restoreState();
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
  els.currentUsername.textContent = username;
  els.currentUserRole.textContent = state.auth?.role || "user";
  els.userMiniAvatar.innerHTML = fixedUserAvatarSvg(username);
}

function fixedUserAvatarSvg(username) {
  const label = (username || "你").slice(0, 1).toUpperCase() || "你";
  return `
    <svg viewBox="0 0 100 100" role="img" aria-label="用户头像">
      <rect width="100" height="100" rx="24" fill="#137a74" />
      <circle cx="28" cy="32" r="18" fill="#436fb0" opacity="0.55" />
      <circle cx="72" cy="28" r="12" fill="#f0bc42" opacity="0.72" />
      <circle cx="62" cy="72" r="22" fill="#fffaf2" opacity="0.22" />
      <text x="50" y="60" text-anchor="middle" fill="white" font-size="34" font-weight="900">${escapeHtml(label)}</text>
    </svg>
  `;
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
}

function renderConversation(lastText) {
  const title = state.messages.find((message) => message.role === "user")?.content || lastText;
  els.conversationTitle.textContent = title ? clamp(title, 24) : "开始一次情绪对话";
  if (els.conversationIdLabel) {
    els.conversationIdLabel.textContent = "";
  }
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

async function loadAdminDashboard() {
  if (!state.adminTargetUserId) {
    renderAdminDashboard(null);
    setAdminStatus("请选择一个用户后再生成画像。", "error");
    return;
  }
  setAdminStatus("正在生成用户画像...");
  els.refreshAdminButton.disabled = true;
  try {
    const response = await apiFetch(
      `/api/emotion/profile/generate?target_user_id=${encodeURIComponent(state.adminTargetUserId)}`,
      { method: "POST" }
    );
    const profile = await response.json();
    renderAdminDashboard(profile);
    setAdminStatus("已根据所选用户的历史情绪记录生成画像。", "success");
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
    const users = Array.isArray(conversations) ? buildAdminUsers(conversations) : [];
    state.adminUsers = users;
    if (!users.length) {
      state.adminTargetUserId = "";
      renderAdminUsers();
      renderAdminDashboard(null);
      setAdminStatus("暂无可生成画像的用户。需要普通用户先产生会话和情绪记录。", "error");
      return;
    }
    state.adminTargetUserId = users[0].id;
    renderAdminUsers();
    await loadAdminDashboard();
  } catch (error) {
    state.adminUsers = [];
    state.adminTargetUserId = "";
    renderAdminUsers();
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
    };
    existing.name = conversation.username || conversation.user_name || existing.name || userId;
    existing.count += 1;
    if (!existing.latest || String(conversation.updated_at || "") > existing.latest) {
      existing.latest = conversation.updated_at || "";
    }
    users.set(userId, existing);
  });
  return [...users.values()].sort((a, b) => String(b.latest).localeCompare(String(a.latest)));
}

function renderAdminUsers() {
  if (!state.adminUsers.length) {
    els.adminTargetLabel.textContent = "暂无可选用户";
    els.adminUserList.innerHTML = `<p class="chart-empty">还没有普通用户会话。</p>`;
    return;
  }
  const current = state.adminUsers.find((user) => user.id === state.adminTargetUserId);
  els.adminTargetLabel.textContent = current ? current.name : "请选择一个用户";
  const query = els.adminUserSearch.value.trim().toLowerCase();
  const visibleUsers = query
    ? state.adminUsers.filter((user) => user.name.toLowerCase().includes(query))
    : state.adminUsers;
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

async function handleAdminUserClick(event) {
  const button = event.target.closest("[data-admin-user-id]");
  if (!button || button.dataset.adminUserId === state.adminTargetUserId) {
    return;
  }
  state.adminTargetUserId = button.dataset.adminUserId;
  renderAdminUsers();
  await loadAdminDashboard();
}

function renderAdminDashboard(profile) {
  const total = asNumber(profile?.total_records) ?? 0;
  const avgIntensity = asNumber(profile?.avg_intensity);
  els.adminTotalRecords.textContent = String(total);
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

  const minWidth = 1180;
  const shouldScroll = records.length > 16;
  container.classList.toggle("is-scrollable-chart", shouldScroll);
  const width = shouldScroll ? Math.max(minWidth, records.length * 112 + 140) : minWidth;
  const svgWidthStyle = shouldScroll ? `width: ${width}px;` : `width: 100%; min-width: ${minWidth}px;`;
  const height = 420;
  const padding = { top: 48, right: 44, bottom: 72, left: 56 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const baseline = padding.top + chartHeight;
  const barSlot = chartWidth / records.length;
  const barWidth = Math.max(10, Math.min(42, barSlot * 0.48));
  const points = records.map((record, index) => {
    const x = padding.left + barSlot * index + barSlot / 2;
    const y = baseline - (record.intensity / 100) * chartHeight;
    return { ...record, x, y };
  });

  const bars = points
    .map((point, index) => {
      const barHeight = baseline - point.y;
      const x = point.x - barWidth / 2;
      return `
        <rect x="${x.toFixed(1)}" y="${point.y.toFixed(1)}" width="${barWidth.toFixed(1)}" height="${barHeight.toFixed(1)}" rx="7" fill="rgba(19, 122, 116, 0.18)">
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
    .map((point, index) => {
      const step = Math.max(1, Math.ceil(points.length / 8));
      if (points.length > 10 && index % step !== 0 && index !== points.length - 1) {
        return "";
      }
      return `<text x="${point.x.toFixed(1)}" y="${Math.max(22, point.y - 12).toFixed(1)}" text-anchor="middle">${escapeHtml(point.emotion || String(point.intensity))}</text>`;
    })
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
  state.auth = null;
  state.conversationId = null;
  state.conversations = [];
  state.conversationsStatus = "loading";
  state.messages = [];
  state.lastProfileSignal = null;
  localStorage.removeItem(AUTH_KEY);
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
