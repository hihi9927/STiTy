const { ipcRenderer } = require('electron');

const DOM = {
  serverIndicatorMini: document.getElementById('serverIndicatorMini'),
  recordingIndicatorMini: document.getElementById('recordingIndicatorMini'),
  closeSettings: document.getElementById('closeSettings'),
  serverUrlInput: document.getElementById('serverUrlInput'),
  modeTranslateOnly: document.getElementById('modeTranslateOnly'),
  modeTranscriptOnly: document.getElementById('modeTranscriptOnly'),
  modeBoth: document.getElementById('modeBoth'),
  langHintAuto: document.getElementById('langHintAuto'),
  langHintKo: document.getElementById('langHintKo'),
  langHintEn: document.getElementById('langHintEn'),
  opacitySlider: document.getElementById('opacitySlider'),
  opacityValue: document.getElementById('opacityValue'),
  textSizeSlider: document.getElementById('textSizeSlider'),
  textSizeValue: document.getElementById('textSizeValue'),
  textColorWhite: document.getElementById('textColorWhite'),
  textColorBlack: document.getElementById('textColorBlack'),
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

  const savedTextSize = localStorage.getItem('textSize');
  if (savedTextSize) {
    DOM.textSizeSlider.value = savedTextSize;
    DOM.textSizeValue.textContent = savedTextSize + '%';
  }

  const savedTextColor = localStorage.getItem('textColor') || 'white';
  if (savedTextColor === 'white') {
    DOM.textColorWhite.checked = true;
  } else {
    DOM.textColorBlack.checked = true;
  }

  const savedLanguageHint = localStorage.getItem('languageHint') || 'auto';
  if (savedLanguageHint === 'auto') {
    DOM.langHintAuto.checked = true;
  } else if (savedLanguageHint === 'ko') {
    DOM.langHintKo.checked = true;
  } else {
    DOM.langHintEn.checked = true;
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

  // 텍스트 색깔 변경
  const textColorHandler = (e) => {
    const color = e.target.value;
    localStorage.setItem('textColor', color);
    ipcRenderer.send('text-color-changed', color);
  };

  DOM.textColorWhite.addEventListener('change', textColorHandler);
  DOM.textColorBlack.addEventListener('change', textColorHandler);

  // 표시 모드 변경
  const displayModeHandler = (e) => {
    const mode = e.target.value;
    localStorage.setItem('displayMode', mode);
    ipcRenderer.send('display-mode-changed', mode);
  };

  DOM.modeTranslateOnly.addEventListener('change', displayModeHandler);
  DOM.modeTranscriptOnly.addEventListener('change', displayModeHandler);
  DOM.modeBoth.addEventListener('change', displayModeHandler);

  // 언어 힌트 변경
  const languageHintHandler = (e) => {
    const lang = e.target.value;
    localStorage.setItem('languageHint', lang);
    ipcRenderer.send('lang-changed', lang);
  };

  DOM.langHintAuto.addEventListener('change', languageHintHandler);
  DOM.langHintKo.addEventListener('change', languageHintHandler);
  DOM.langHintEn.addEventListener('change', languageHintHandler);

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
