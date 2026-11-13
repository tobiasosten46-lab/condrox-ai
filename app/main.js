// Condrox AI Desktop — main process
// Boots the Condrox Python backend, waits for health, opens the chat window.
const { app, BrowserWindow, ipcMain } = require("electron");
const { spawn } = require("child_process");
const crypto = require("crypto");
const path = require("path");
const fs = require("fs");
const http = require("http");

const BACKEND_ROOT = path.resolve(__dirname, "..");
const PORT = parseInt(process.env.CONDROX_PORT || "8765", 10);
const API_KEY = crypto.randomBytes(24).toString("hex"); // per-session key
const BOOT_TIMEOUT_MS = 120000;

let backend = null;
let win = null;

function findPython() {
  if (process.env.CONDROX_PYTHON) return process.env.CONDROX_PYTHON;
  const candidates = process.platform === "win32"
    ? [path.join(BACKEND_ROOT, ".venv", "Scripts", "python.exe"), "python"]
    : [path.join(BACKEND_ROOT, ".venv", "bin", "python"), "python3", "python"];
  for (const c of candidates) {
    if (c.includes(path.sep) ? fs.existsSync(c) : true) return c;
  }
  return "python";
}

function startBackend() {
  const py = findPython();
  const env = {
    ...process.env,
    CONDROX_API_KEY: API_KEY,
    CONDROX_SKIP_KNOWLEDGE_LOAD: process.env.CONDROX_SKIP_KNOWLEDGE_LOAD || "1",
    PYTHONUNBUFFERED: "1",
  };
  backend = spawn(
    py,
    ["-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", String(PORT)],
    { cwd: BACKEND_ROOT, env, windowsHide: true }
  );
  backend.stdout.on("data", (d) => process.stdout.write(`[condrox] ${d}`));
  backend.stderr.on("data", (d) => process.stderr.write(`[condrox] ${d}`));
  backend.on("exit", (code) => {
    if (win && !win.isDestroyed()) {
      win.webContents.send("backend-status", { state: "died", code });
    }
  });
}

function waitForHealth(timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = http.get(
        { host: "127.0.0.1", port: PORT, path: "/health", timeout: 2000 },
        (res) => {
          res.resume();
          if (res.statusCode === 200) return resolve();
          retry();
        }
      );
      req.on("error", retry);
      req.on("timeout", () => { req.destroy(); retry(); });
    };
    const retry = () => {
      if (Date.now() > deadline) return reject(new Error("Backend boot timed out"));
      setTimeout(tick, 1000);
    };
    tick();
  });
}

function createWindow() {
  win = new BrowserWindow({
    width: 900,
    height: 700,
    minWidth: 640,
    minHeight: 500,
    title: "Condrox AI",
    backgroundColor: "#0e1116",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadFile(path.join(__dirname, "index.html"));
  win.setMenuBarVisibility(false);
}

ipcMain.handle("get-config", () => ({
  apiUrl: `http://127.0.0.1:${PORT}`,
  apiKey: API_KEY,
}));

app.whenReady().then(async () => {
  createWindow();
  win.webContents.once("did-finish-load", () => {
    win.webContents.send("backend-status", { state: "booting" });
  });
  startBackend();
  try {
    await waitForHealth(BOOT_TIMEOUT_MS);
    win.webContents.send("backend-status", { state: "ready" });
  } catch (err) {
    win.webContents.send("backend-status", { state: "error", message: String(err) });
  }
});

app.on("window-all-closed", () => {
  if (backend) backend.kill();
  app.quit();
});

app.on("before-quit", () => {
  if (backend) backend.kill();
});
