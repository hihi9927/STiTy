const { ipcRenderer } = require('electron');

const DOM = {
  serverIndicatorMini: document.getElementById('serverIndicatorMini'),
  recordingIndicatorMini: document.getElementById('recordingIndicatorMini'),
  closeSettings: document.getElementById('closeSettings'),
  serverUrlInput: document.getElementById('serverUrlInput'),
  langSelect: document.getElementById('langSelect'),
  polishCheckbox: document.getElementById('polishCheckbox'),
  modeTranslateOnly: document.getElementById('modeTranslateOnly'),
  modeTranscriptOnly: document.getElementById('modeTranscriptOnly'),
  modeBoth: document.getElementById('modeBoth'),
  opacitySlider: document.getElementById('opacitySlider'),
  opacityValue: document.getElementById('opacityValue'),
  autoOpacityCheckbox: document.getElementById('autoOpacityCheckbox'),
  textSizeSlider: document.getElementById('textSizeSlider'),
  textSizeValue: document.getElementById('textSizeValue'),
  translationLog: document.getElementById('translationLog'),
  closeAppBtn: document.getElementById('closeAppBtn')
};

// 메인 창으로부터 상태 업데이트 수신
ipcRenderer.on('update-server-status', (event, connected) => {
  DOM.serverIndicatorMini.classList.toggle('connected', connected);
});

ipcRenderer.on('update-recording-status', (event, recording) => {
  DOM.recordingIndicatorMini.classList.toggle('active', recording);
});

ipcRenderer.on('update-translation-log', (event, history) => {
  updateLogDisplay(history);
});

function updateLogDisplay(history) {
  if (history.length === 0) {
    DOM.translationLog.innerHTML = '<div class="log-empty">아직 번역 기록이 없습니다</div>';
    return;
  }

  DOM.translationLog.innerHTML = history.map(entry => `
    <div class="log-entry">
      <div class="log-time">${entry.time}</div>
      <div class="log-text">${entry.text.replace(/\n/g, '<br>')}</div>
    </div>
  `).join('');
}

function loadSettings() {
  const savedUrl = localStorage.getItem('serverUrl');
  if (savedUrl) {
    DOM.serverUrlInput.value = savedUrl;
  }

  const savedLang = localStorage.getItem('langPreference');
  if (savedLang) {
    DOM.langSelect.value = savedLang;
  }

  const savedPolish = localStorage.getItem('polishEnabled');
  if (savedPolish !== null) {
    DOM.polishCheckbox.checked = savedPolish === 'true';
  }

  const savedDisplayMode = localStorage.getItem('displayMode') || 'both';
  if (savedDisplayMode === 'translateOnly') {
    DOM.modeTranslateOnly.checked = true;
  } else if (savedDisplayMode === 'transcriptOnly') {
    DOM.modeTranscriptOnly.checked = true;
  } else {
    DOM.modeBoth.checked = true;
  }

  const savedOpacity = localStorage.getItem('panelOpacity');
  if (savedOpacity) {
    DOM.opacitySlider.value = savedOpacity;
    DOM.opacityValue.textContent = savedOpacity + '%';
  }

  const savedAutoOpacity = localStorage.getItem('autoAdjustOpacity');
  if (savedAutoOpacity !== null) {
    DOM.autoOpacityCheckbox.checked = savedAutoOpacity === 'true';
  }

  const savedTextSize = localStorage.getItem('textSize');
  if (savedTextSize) {
    DOM.textSizeSlider.value = savedTextSize;
    DOM.textSizeValue.textContent = savedTextSize + '%';
  }
}

function setupEventListeners() {
  // Space 키 이벤트 - 설정 창에서도 녹음 토글 가능
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !e.target.matches('input, select, textarea')) {
      e.preventDefault();
      if (e.repeat) return; // 연속 입력 방지

      // 메인 창으로 녹음 토글 요청 전달
      ipcRenderer.send('toggle-recording');
    }
  });

  // 설정 창 닫기
  DOM.closeSettings.addEventListener('click', () => {
    ipcRenderer.send('close-settings-window');
  });

  // 투명도 변경
  DOM.opacitySlider.addEventListener('input', (e) => {
    const value = e.target.value;
    DOM.opacityValue.textContent = value + '%';
    localStorage.setItem('panelOpacity', value);
    ipcRenderer.send('opacity-changed', value);
  });

  // 자동 투명도 조정 변경
  DOM.autoOpacityCheckbox.addEventListener('change', (e) => {
    localStorage.setItem('autoAdjustOpacity', e.target.checked);
    ipcRenderer.send('auto-opacity-changed', e.target.checked);
  });

  // 텍스트 크기 변경
  DOM.textSizeSlider.addEventListener('input', (e) => {
    const value = e.target.value;
    DOM.textSizeValue.textContent = value + '%';
    localStorage.setItem('textSize', value);
    ipcRenderer.send('text-size-changed', value);
  });

  // 서버 URL 변경
  DOM.serverUrlInput.addEventListener('change', (e) => {
    const value = e.target.value.trim();
    localStorage.setItem('serverUrl', value);
    ipcRenderer.send('server-url-changed', value);
  });

  // 언어 설정 변경
  DOM.langSelect.addEventListener('change', (e) => {
    localStorage.setItem('langPreference', e.target.value);
    ipcRenderer.send('lang-changed', e.target.value);
  });

  // 다듬기 설정 변경
  DOM.polishCheckbox.addEventListener('change', (e) => {
    localStorage.setItem('polishEnabled', e.target.checked);
    ipcRenderer.send('polish-changed', e.target.checked);
  });

  // 표시 모드 변경
  const displayModeHandler = (e) => {
    const mode = e.target.value;
    localStorage.setItem('displayMode', mode);
    ipcRenderer.send('display-mode-changed', mode);
  };

  DOM.modeTranslateOnly.addEventListener('change', displayModeHandler);
  DOM.modeTranscriptOnly.addEventListener('change', displayModeHandler);
  DOM.modeBoth.addEventListener('change', displayModeHandler);

  // 앱 종료
  DOM.closeAppBtn.addEventListener('click', () => {
    ipcRenderer.send('close-app');
  });
}

function init() {
  loadSettings();
  setupEventListeners();

  // 메인 창에 초기 상태 요청
  ipcRenderer.send('request-initial-state');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
