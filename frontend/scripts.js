const BACKEND_URL = "http://localhost:8080/api/chat";

const chatContainer = document.getElementById("chat-container");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const historySection = document.getElementById("history-section");
const sidebar = document.getElementById("sidebar");
const toggleSidebarBtn = document.getElementById("toggle-sidebar");

let chatHistory = [];
let currentChatId = null;

// Load chat history
function loadChatHistory() {
    historySection.innerHTML = '<div class="history-label">Today</div>';

    chatHistory.forEach((chat, index) => {
        const item = document.createElement("div");
        item.classList.add("history-item");
        if (index === currentChatId) item.classList.add("active");

        const textSpan = document.createElement("span");
        textSpan.classList.add("history-item-text");
        textSpan.innerHTML = `💬 ${chat.title}`;
        textSpan.onclick = () => loadChat(index);

        const deleteBtn = document.createElement("button");
        deleteBtn.classList.add("delete-chat-btn");
        deleteBtn.innerHTML = "🗑️";
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            deleteChat(index);
        };

        item.appendChild(textSpan);
        item.appendChild(deleteBtn);
        historySection.appendChild(item);
    });
}

function deleteChat(chatId) {
    if (chatHistory.length === 1) {
        alert("You must have at least one chat!");
        return;
    }

    if (confirm("Are you sure you want to delete this chat?")) {
        chatHistory.splice(chatId, 1);

        if (currentChatId === chatId) {
            currentChatId = Math.max(0, chatId - 1);
            loadChat(currentChatId);
        } else if (currentChatId > chatId) {
            currentChatId--;
        }

        loadChatHistory();
    }
}

function createNewChat() {
    const chatId = chatHistory.length;

    chatHistory.push({
        id: chatId,
        title: "New Chat",
        messages: []
    });

    currentChatId = chatId;

    chatContainer.innerHTML =
        '<div class="message bot">👋 Hello! I\'m your Constitution AI Assistant. Ask anything from the Pakistan Constitution PDF.</div>';

    loadChatHistory();
}

function loadChat(chatId) {
    currentChatId = chatId;
    const chat = chatHistory[chatId];

    chatContainer.innerHTML =
        '<div class="message bot">👋 Hello! I\'m your Constitution AI Assistant. Ask anything from the Pakistan Constitution PDF.</div>';

    chat.messages.forEach(msg => addMessage(msg.text, msg.sender, false, false));

    loadChatHistory();
}

function addMessage(text, sender, isThinking = false, save = true) {
    const msg = document.createElement("div");
    msg.classList.add("message", sender);
    if (isThinking) msg.classList.add("thinking");
    msg.innerText = text;

    chatContainer.appendChild(msg);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    if (save && currentChatId !== null) {
        chatHistory[currentChatId].messages.push({ text, sender });

        if (sender === "user" && chatHistory[currentChatId].title === "New Chat") {
            chatHistory[currentChatId].title =
                text.substring(0, 30) + (text.length > 30 ? "..." : "");
            loadChatHistory();
        }
    }

    return msg;
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (text === "") return;

    if (currentChatId === null) createNewChat();

    addMessage(text, "user");
    userInput.value = "";

    const thinkingMsg = addMessage("⏳ Thinking...", "bot", true, false);

    try {
        const response = await fetch(BACKEND_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },

            // ✅ FIXED — Must be "question", NOT "query"
            body: JSON.stringify({ question: text }),
        });

        const data = await response.json();

        thinkingMsg.remove();
        addMessage(data.answer || "No response available.", "bot");

    } catch (error) {
        thinkingMsg.remove();
        addMessage("⚠️ Error connecting to backend. Make sure server is running.", "bot");
        console.error("Backend error:", error);
    }
}

toggleSidebarBtn.addEventListener("click", () => {
    sidebar.classList.toggle("open");
});

document.addEventListener("click", (e) => {
    if (
        window.innerWidth <= 768 &&
        sidebar.classList.contains("open") &&
        !sidebar.contains(e.target) &&
        e.target !== toggleSidebarBtn
    ) {
        sidebar.classList.remove("open");
    }
});

newChatBtn.addEventListener("click", createNewChat);
sendBtn.addEventListener("click", sendMessage);

userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
});

createNewChat();
userInput.focus();
