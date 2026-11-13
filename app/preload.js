const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("condrox", {
  getConfig: () => ipcRenderer.invoke("get-config"),
  onBackendStatus: (cb) =>
    ipcRenderer.on("backend-status", (_e, status) => cb(status)),
});
