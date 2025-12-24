const BACKEND_URL = "http://localhost:8000/chat";

const chatContainer = document.getElementById("chat-container");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const historySection = document.getElementById("history-section");
const sidebar = document.getElementById("sidebar");
const toggleSidebarBtn = document.getElementById("toggle-sidebar");
const overlay = document.getElementById("overlay");

let chatHistory = [];
let currentChatId = null;

function loadChatHistory() {
    historySection.innerHTML = '<div class="history-label">Today</div>';

    chatHistory.forEach((chat, index) => {
        const item = document.createElement("div");
        item.className = "history-item";
        if (index === currentChatId) item.classList.add("active");

        item.innerHTML = `
            <span class="history-item-text">💬 ${chat.title}</span>
            <button class="delete-chat-btn">🗑️</button>
        `;

        item.onclick = () => loadChat(index);
        item.querySelector("button").onclick = (e) => {
            e.stopPropagation();
            deleteChat(index);
        };

        historySection.appendChild(item);
    });
}

function createNewChat() {
    const id = chatHistory.length;
    chatHistory.push({ id, title: "New Chat", messages: [] });
    currentChatId = id;

    chatContainer.innerHTML =
        `<div class="message bot">👋 Hello! I'm your Constitution AI Assistant.</div>`;

    loadChatHistory();
}

function loadChat(id) {
    currentChatId = id;
    chatContainer.innerHTML =
        `<div class="message bot">👋 Hello! I'm your Constitution AI Assistant.</div>`;

    chatHistory[id].messages.forEach(m =>
        addMessage(m.text, m.sender, false, false)
    );

    loadChatHistory();
}

function addMessage(text, sender, thinking = false, save = true) {
    const msg = document.createElement("div");
    msg.className = `message ${sender}`;
    if (thinking) msg.classList.add("thinking");
    msg.innerText = text;

    chatContainer.appendChild(msg);
    chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: "smooth" });

    if (save && currentChatId !== null) {
        chatHistory[currentChatId].messages.push({ text, sender });

        if (sender === "user" && chatHistory[currentChatId].title === "New Chat") {
            chatHistory[currentChatId].title = text.slice(0, 30) + "...";
            loadChatHistory();
        }
    }
    return msg;
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    if (currentChatId === null) createNewChat();

    addMessage(text, "user");
    userInput.value = "";

    const thinking = addMessage("⏳ Thinking...", "bot", true, false);

    try {
        const res = await fetch(BACKEND_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: text })
        });

        const data = await res.json();
        thinking.remove();
        addMessage(data.answer || "No response.", "bot");
    } catch {
        thinking.remove();
        addMessage("⚠️ Backend not reachable.", "bot");
    }
}

/* Sidebar mobile behavior */
toggleSidebarBtn.onclick = () => {
    sidebar.classList.toggle("open");
    overlay.classList.toggle("open");
};

overlay.onclick = () => {
    sidebar.classList.remove("open");
    overlay.classList.remove("open");
};

sendBtn.onclick = sendMessage;
newChatBtn.onclick = createNewChat;

userInput.addEventListener("keydown", e => {
    if (e.key === "Enter") sendMessage();
});

createNewChat();
userInput.focus();
