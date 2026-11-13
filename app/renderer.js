const chat = document.getElementById("chat");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = form.querySelector("button");
const statusEl = document.getElementById("status");

let api = null;

function addMsg(text, cls, meta) {
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.textContent = text;
  if (meta) {
    const m = document.createElement("span");
    m.className = "meta";
    m.textContent = meta;
    div.appendChild(m);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

window.condrox.onBackendStatus((s) => {
  statusEl.className = s.state;
  if (s.state === "booting") statusEl.textContent = "booting brain…";
  if (s.state === "ready") {
    statusEl.textContent = "online";
    input.disabled = sendBtn.disabled = false;
    input.focus();
    addMsg("Condrox AI online. What would you like to talk about?", "bot");
  }
  if (s.state === "error") {
    statusEl.textContent = "boot failed";
    addMsg(`Backend failed to start: ${s.message}. ` +
      "Check that Python 3.11+ and requirements.txt are installed " +
      "(see README Quickstart).", "error");
  }
  if (s.state === "died") {
    statusEl.textContent = "brain stopped";
    input.disabled = sendBtn.disabled = true;
    addMsg(`Backend exited (code ${s.code}). Restart the app to reboot it.`, "error");
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || !api) return;
  input.value = "";
  addMsg(text, "user");
  input.disabled = sendBtn.disabled = true;
  try {
    const res = await fetch(`${api.apiUrl}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": api.apiKey,
      },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    const meta = [
      d.intent && `intent: ${d.intent}`,
      d.mode && `mode: ${d.mode}`,
      d.emotion && `emotion: ${d.emotion}`,
      d.confidence != null && `confidence: ${d.confidence.toFixed(2)}`,
      d.response_time_ms != null && `${d.response_time_ms.toFixed(0)} ms`,
    ].filter(Boolean).join(" · ");
    addMsg(d.response, "bot", meta);
  } catch (err) {
    addMsg(`Request failed: ${err.message}`, "error");
  } finally {
    input.disabled = sendBtn.disabled = false;
    input.focus();
  }
});

window.condrox.getConfig().then((c) => { api = c; });
