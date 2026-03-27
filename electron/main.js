const {
  app,
  BrowserWindow,
  ipcMain,
  dialog,
  Tray,
  Menu,
  Notification,
  nativeImage
} = require('electron');

const path = require('node:path');
const fs = require('node:fs/promises');
const { spawn } = require('node:child_process');
const chokidar = require('chokidar');

const PROJECT_ROOT = path.join(__dirname, '..');
const WATCHED_EXTENSIONS = new Set(['.exe', '.dll']);

let mainWindow = null;
let tray = null;
let watcher = null;
let isQuitting = false;

const activeScans = new Set();
const scannedFingerprints = new Map();

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
}

app.on('second-instance', () => {
  showMainWindow();
});

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1100,
    minHeight: 760,
    show: false,
    backgroundColor: '#0b1020',
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.on('close', (event) => {
    if (!isQuitting) {
      event.preventDefault();
      mainWindow.hide();
    }
  });
}

function showMainWindow() {
  if (!mainWindow) return;
  if (mainWindow.isMinimized()) mainWindow.restore();
  mainWindow.show();
  mainWindow.focus();
}

function createTray() {
  const iconPath = path.join(PROJECT_ROOT, 'assets', 'tray.png');
  const trayIcon = nativeImage.createFromPath(iconPath);

  tray = new Tray(trayIcon);
  tray.setToolTip('ExeVision');

  refreshTrayMenu();

  tray.on('click', () => {
    if (!mainWindow) return;
    if (mainWindow.isVisible()) {
      mainWindow.hide();
    } else {
      showMainWindow();
    }
  });
}

function refreshTrayMenu() {
  if (!tray) return;

  const menu = Menu.buildFromTemplate([
    { label: 'Open ExeVision', click: showMainWindow },
    {
      label: watcher ? 'Watching Downloads' : 'Watcher not running',
      enabled: false
    },
    { type: 'separator' },
    {
      label: 'Quit',
      click: () => {
        isQuitting = true;
        app.quit();
      }
    }
  ]);

  tray.setContextMenu(menu);
}

function getPythonLaunchConfig() {
  const bridgeScript = path.join(PROJECT_ROOT, 'app', 'electron_bridge.py');

  if (process.env.PYTHON_PATH) {
    return {
      command: process.env.PYTHON_PATH,
      args: [bridgeScript]
    };
  }

  if (process.platform === 'win32') {
    return {
      command: 'py',
      args: ['-3', bridgeScript]
    };
  }

  return {
    command: 'python3',
    args: [bridgeScript]
  };
}

function runAnalysis(filePath) {
  return new Promise((resolve, reject) => {
    const { command, args } = getPythonLaunchConfig();

    const child = spawn(command, [...args, filePath], {
      cwd: PROJECT_ROOT,
      windowsHide: true
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    child.on('error', (err) => {
      reject(`Failed to start Python process: ${err.message}`);
    });

    child.on('close', (code) => {
      if (code !== 0) {
        reject(stderr || `Python process exited with code ${code}`);
        return;
      }

      try {
        const parsed = JSON.parse(stdout);

        if (!parsed.ok) {
          reject(parsed.error || 'Unknown analysis error');
          return;
        }

        resolve(parsed.result);
      } catch (err) {
        reject(`Could not parse analyzer output: ${err.message}\n\nRaw output:\n${stdout}`);
      }
    });
  });
}

function isWatchedExecutable(filePath) {
  return WATCHED_EXTENSIONS.has(path.extname(filePath).toLowerCase());
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function ensureStableFile(filePath, checks = 3, delayMs = 1500) {
  let lastSignature = null;
  let stableCount = 0;

  while (stableCount < checks) {
    const stat = await fs.stat(filePath);

    if (!stat.isFile()) {
      throw new Error('Target is not a file.');
    }

    const signature = `${stat.size}:${Math.trunc(stat.mtimeMs)}`;

    if (stat.size > 0 && signature === lastSignature) {
      stableCount += 1;
    } else {
      lastSignature = signature;
      stableCount = 1;
    }

    await sleep(delayMs);
  }

  return lastSignature;
}

async function autoAnalyzeFile(filePath) {
  if (!isWatchedExecutable(filePath)) return;
  if (activeScans.has(filePath)) return;

  activeScans.add(filePath);

  try {
    const fingerprint = await ensureStableFile(filePath);

    if (scannedFingerprints.get(filePath) === fingerprint) {
      return;
    }

    const result = await runAnalysis(filePath);
    scannedFingerprints.set(filePath, fingerprint);

    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('autoscan:complete', { filePath, result });
    }

    if (Notification.isSupported()) {
      new Notification({
        title: 'ExeVision auto-scan complete',
        body: `${path.basename(filePath)} → ${result.score_info?.label || 'Done'} (${result.score_info?.score ?? '--'}/100)`
      }).show();
    }
  } catch (error) {
    const message = String(error);

    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('autoscan:error', { filePath, error: message });
    }

    if (Notification.isSupported()) {
      new Notification({
        title: 'ExeVision auto-scan failed',
        body: `${path.basename(filePath)} → ${message}`
      }).show();
    }
  } finally {
    activeScans.delete(filePath);
  }
}

function startDownloadsWatcher() {
  const downloadsDir = app.getPath('downloads');

  watcher = chokidar.watch(downloadsDir, {
    persistent: true,
    ignoreInitial: true,
    depth: 0,
    awaitWriteFinish: {
      stabilityThreshold: 5000,
      pollInterval: 500
    }
  });

  watcher.on('add', (filePath) => {
    void autoAnalyzeFile(filePath);
  });

  watcher.on('change', (filePath) => {
    void autoAnalyzeFile(filePath);
  });

  watcher.on('error', (error) => {
    console.error('Watcher error:', error);
  });

  refreshTrayMenu();
}

ipcMain.handle('dialog:pickFile', async () => {
  const result = await dialog.showOpenDialog({
    title: 'Choose file to analyze',
    properties: ['openFile'],
    filters: [
      { name: 'Executables', extensions: ['exe', 'dll'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });

  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }

  return result.filePaths[0];
});

ipcMain.handle('analysis:run', async (_event, filePath) => {
  return await runAnalysis(filePath);
});

app.whenReady().then(() => {
  createWindow();
  createTray();
  startDownloadsWatcher();

  if (app.isPackaged) {
    app.setLoginItemSettings({
      openAtLogin: true
    });
  }

  app.on('activate', () => {
    showMainWindow();
  });
});

app.on('before-quit', async () => {
  isQuitting = true;
  if (watcher) {
    await watcher.close();
  }
});

app.on('window-all-closed', () => {
  // Intentionally do nothing.
  // The app stays alive in the tray and keeps watching Downloads.
});
