const DOM = {
  currTextOriginal: document.getElementById('currTextOriginal'),
  currTextTranslated: document.getElementById('currTextTranslated'),
  current: document.getElementById('current'),
  mainPanel: document.getElementById('mainPanel')
};

const state = {
  tcpClient: null,
  mediaRecorder: null,
  mediaStream: null,
  isRecording: false,
  isServerConnected: false,
  translationHistory: [],
  recordingEnabled: false,
  SERVER_HOST: '127.0.0.1',
  SERVER_PORT: 43007,
  currentOriginal: '',
  currentTranslated: '',
  isConnecting: false,
  isBlackFullscreen: false,
  displayMode: 'both', // 'translateOnly', 'transcriptOnly', 'both'
  backgroundBrightness: undefined,
  autoAdjustOpacity: false
};

function updateServerStatus(connected) {
  state.isServerConnected = connected;
  console.log('🔌 서버 상태:', connected ? '연결됨' : '연결 끊김');

  // 설정 창으로 상태 전달
  if (window.require) {
    try {
      const { ipcRenderer } = window.require('electron');
      ipcRenderer.send('status-update', 'server-status', connected);
    } catch(e) {}
  }
}

function updateRecordingStatus(recording) {
  state.isRecording = recording;
  console.log('🎤 녹음 상태:', recording ? '활성' : '비활성');

  // 그라데이션 줄 색상 변경
  const ruleElement = document.querySelector('.rule');
  if (ruleElement) {
    if (recording) {
      ruleElement.classList.remove('paused');
    } else {
      ruleElement.classList.add('paused');
    }
  }

  // 설정 창으로 상태 전달
  if (window.require) {
    try {
      const { ipcRenderer } = window.require('electron');
      ipcRenderer.send('status-update', 'recording-status', recording);
    } catch(e) {}
  }
}

function loadOpacity() {
  const saved = localStorage.getItem('panelOpacity');
  updateOpacity(saved || 100);
}

function updateOpacity(value, autoAdjust = false) {
  const transparency = value / 100;
  let bgOpacity = 0.95 * (1 - transparency);

  // 자동 조정 모드일 때 배경 밝기에 따라 보정
  if (autoAdjust && state.backgroundBrightness !== undefined) {
    const brightness = state.backgroundBrightness;

    // 밝기 기준: 0-255
    // 0-80: 매우 어두움 -> 투명도 높임 (0.3-0.5)
    // 80-150: 중간 어두움 -> 기본 투명도 (0.5-0.75)
    // 150-200: 밝음 -> 불투명도 높임 (0.75-0.9)
    // 200-255: 매우 밝음 -> 매우 불투명 (0.9-0.98)

    if (brightness < 80) {
      // 매우 어두운 배경: 투명하게
      bgOpacity = Math.max(0.3, bgOpacity * 0.5);
    } else if (brightness < 150) {
      // 중간 어두운 배경: 약간 투명하게
      bgOpacity = Math.max(0.5, bgOpacity * 0.8);
    } else if (brightness < 200) {
      // 밝은 배경: 불투명하게
      bgOpacity = Math.min(0.95, bgOpacity + 0.15);
    } else {
      // 매우 밝은 배경: 매우 불투명하게
      bgOpacity = Math.min(0.98, bgOpacity + 0.3);
    }

    console.log(`🎨 투명도 조정: 밝기=${brightness.toFixed(0)} -> 불투명도=${bgOpacity.toFixed(2)}`);
  }

  DOM.mainPanel.style.background = `rgba(20,20,20,${bgOpacity})`;
  localStorage.setItem('panelOpacity', value);
}

function loadTextSize() {
  const saved = localStorage.getItem('textSize');
  updateTextSize(saved || 100);
}

function updateTextSize(value) {
  const scale = value / 100;
  const originalSize = 20; // 원본 크기
  const translatedOriginalSize = 16; // 번역 텍스트 원본 크기

  DOM.currTextOriginal.style.fontSize = `${originalSize * scale}px`;
  DOM.currTextTranslated.style.fontSize = `${translatedOriginalSize * scale}px`;
  localStorage.setItem('textSize', value);
}

function showResult(original, translated) {
  const orig = (original ?? '').trim();
  const trans = (translated ?? '').trim();

  if (!orig && !trans) return;

  DOM.current.classList.add('typing');

  // 표시 모드에 따라 텍스트 표시
  if (state.displayMode === 'translateOnly') {
    // 번역만
    DOM.currTextOriginal.textContent = trans || orig;
    DOM.currTextTranslated.textContent = '';
  } else if (state.displayMode === 'transcriptOnly') {
    // 전사만
    DOM.currTextOriginal.textContent = orig;
    DOM.currTextTranslated.textContent = '';
  } else {
    // 번역+전사 (기본)
    DOM.currTextOriginal.textContent = orig;
    DOM.currTextTranslated.textContent = trans;
  }

  state.currentOriginal = orig;
  state.currentTranslated = trans;

  const logText = orig + (trans ? '\n' + trans : '');
  addToLog(logText);
}

function addToLog(text) {
  const now = new Date();
  const timeStr = now.toLocaleTimeString('ko-KR', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });

  state.translationHistory.unshift({ time: timeStr, text });
  if (state.translationHistory.length > 50) {
    state.translationHistory.pop();
  }

  // 설정 창으로 로그 전달
  if (window.require) {
    try {
      const { ipcRenderer } = window.require('electron');
      ipcRenderer.send('status-update', 'translation-log', state.translationHistory);
    } catch(e) {}
  }
}

function openSettings() {
  if (!window.require) return;

  try {
    const { ipcRenderer } = window.require('electron');
    ipcRenderer.send('open-settings-window');
  } catch(e) {
    console.log('설정 창 열기 실패', e);
  }
}

async function connectTCPSocket() {
  if (state.isConnecting) {
    console.log('⚠️ 이미 연결 시도 중');
    return;
  }

  if (state.tcpClient && !state.tcpClient.destroyed) {
    console.log('✅ 이미 연결되어 있음');
    return;
  }

  state.isConnecting = true;

  try {
    if (!window.require) {
      console.error('❌ Node.js require가 필요합니다');
      state.isConnecting = false;
      return;
    }

    const net = window.require('net');

    console.log(`🔌 TCP 연결 시도: ${state.SERVER_HOST}:${state.SERVER_PORT}`);

    state.tcpClient = new net.Socket();

    // TCP 소켓 옵션 설정
    state.tcpClient.setNoDelay(true); // Nagle 알고리즘 비활성화 (낮은 지연시간)
    state.tcpClient.setKeepAlive(true, 1000); // Keep-alive 설정

    state.tcpClient.connect(state.SERVER_PORT, state.SERVER_HOST, () => {
      console.log('✅ TCP 연결 성공');
      state.isConnecting = false;
      updateServerStatus(true);
    });

    let buffer = '';

    state.tcpClient.on('data', (data) => {
      try {
        // TCP로부터 받은 데이터를 문자열로 변환
        buffer += data.toString('utf-8');

        // 줄바꿈으로 구분된 메시지 처리
        const lines = buffer.split('\n');
        buffer = lines.pop(); // 마지막 불완전한 줄은 버퍼에 남김

        for (const line of lines) {
          if (!line.trim()) continue;

          // 서버 출력 형식: "시작시간(ms) 종료시간(ms) 텍스트"
          console.log('📥 서버 메시지:', line);

          const parts = line.trim().split(' ');
          if (parts.length >= 3) {
            const startMs = parseInt(parts[0]);
            const endMs = parseInt(parts[1]);
            const text = parts.slice(2).join(' ');

            if (text) {
              console.log('🟢 결과:', text);
              showResult(text, '');
            }
          }
        }
      } catch (e) {
        console.error('❌ 메시지 파싱 오류:', e, data);
      }
    });

    state.tcpClient.on('error', (error) => {
      console.error('❌ TCP 오류:', error);
      state.isConnecting = false;
      updateServerStatus(false);
    });

    state.tcpClient.on('close', () => {
      console.log('🔌 TCP 연결 종료');
      state.isConnecting = false;
      updateServerStatus(false);
      state.tcpClient = null;
    });

  } catch (error) {
    console.error('❌ 연결 실패:', error);
    state.isConnecting = false;
    updateServerStatus(false);
  }
}

async function initAudioStream() {
  if (state.mediaStream) {
    console.log('✅ 오디오 스트림 재사용');
    return;
  }
  
  try {
    console.log('🎤 마이크 권한 요청 중...');
    state.mediaStream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        channelCount: 1,
        sampleRate: 48000,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      } 
    });
    console.log('✅ 마이크 접근 허용');
  } catch (error) {
    console.error('❌ 마이크 접근 거부:', error);
    alert('마이크 접근이 거부되었습니다.\n브라우저 설정에서 마이크 권한을 허용해주세요.');
    throw error;
  }
}

async function startRecording() {
  if (state.isRecording) {
    console.log('⚠️ 이미 녹음 중');
    return;
  }

  try {
    // TCP 연결 확인 및 재연결
    await connectTCPSocket();

    // TCP 연결이 열릴 때까지 대기
    let retries = 0;
    while ((!state.tcpClient || state.tcpClient.destroyed) && retries < 10) {
      console.log(`⏳ TCP 연결 대기 중... (${retries + 1}/10)`);
      await new Promise(resolve => setTimeout(resolve, 300));
      retries++;
    }

    if (!state.tcpClient || state.tcpClient.destroyed) {
      console.error('❌ TCP 연결 타임아웃');
      alert('서버 연결에 실패했습니다. 서버 주소를 확인해주세요.');
      return;
    }

    await initAudioStream();

    // AudioContext를 사용하여 PCM 데이터로 변환
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioContext({ sampleRate: 16000 });

    const source = audioContext.createMediaStreamSource(state.mediaStream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);

    let chunkCount = 0;

    processor.onaudioprocess = (e) => {
      if (!state.isRecording || !state.tcpClient || state.tcpClient.destroyed) {
        return;
      }

      const inputData = e.inputBuffer.getChannelData(0);

      // Float32Array를 Int16Array로 변환 (PCM_16 형식, Little Endian)
      const int16Data = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        const s = Math.max(-1, Math.min(1, inputData[i]));
        int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }

      // Buffer로 변환하여 TCP 소켓으로 전송
      const buffer = Buffer.from(int16Data.buffer);

      try {
        state.tcpClient.write(buffer);

        // 10초마다 한 번씩만 로그 출력 (16kHz, 4096샘플 => 약 0.256초마다 호출되므로 약 39번에 한번)
        chunkCount++;
        if (chunkCount % 40 === 0) {
          console.log('📤 오디오 전송 중... (청크:', chunkCount, ', 크기:', buffer.length, 'bytes)');
        }
      } catch (error) {
        console.error('❌오디오 전송 실패:', error);
      }
    };

    source.connect(processor);
    processor.connect(audioContext.destination);

    // AudioContext 정리를 위해 저장
    state.audioContext = audioContext;
    state.audioProcessor = processor;
    state.audioSource = source;

    state.isRecording = true;
    updateRecordingStatus(true);
    console.log('🎙️ 녹음 시작');

  } catch(error){
    console.error('❌ 녹음 시작 에러:', error);
    state.isRecording = false;
    updateRecordingStatus(false);
  }
}

function stopRecording() {
  console.log('⏹️ 녹음 중지 요청');

  // AudioContext 정리
  if (state.audioProcessor) {
    state.audioProcessor.disconnect();
    state.audioProcessor = null;
  }

  if (state.audioSource) {
    state.audioSource.disconnect();
    state.audioSource = null;
  }

  if (state.audioContext) {
    state.audioContext.close();
    state.audioContext = null;
  }

  state.isRecording = false;
  updateRecordingStatus(false);
  console.log('✅ 녹음 중지 완료');
}

function cleanupRecording() {
  console.log('🧹 리소스 정리 시작');

  // AudioContext 정리
  if (state.audioProcessor) {
    state.audioProcessor.disconnect();
    state.audioProcessor = null;
  }

  if (state.audioSource) {
    state.audioSource.disconnect();
    state.audioSource = null;
  }

  if (state.audioContext && state.audioContext.state !== 'closed') {
    state.audioContext.close();
    state.audioContext = null;
  }

  // MediaStream 정리
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach(track => {
      track.stop();
      console.log('🛑 오디오 트랙 중지:', track.label);
    });
    state.mediaStream = null;
  }

  // TCP 소켓 정리
  if (state.tcpClient) {
    if (!state.tcpClient.destroyed) {
      try {
        state.tcpClient.destroy();
      } catch (e) {
        console.error('⚠️ TCP 소켓 종료 실패:', e);
      }
    }
    state.tcpClient = null;
  }

  state.isRecording = false;
  state.recordingEnabled = false;
  console.log('✅ 리소스 정리 완료');
}

function loadSettings() {
  const savedHost = localStorage.getItem('serverHost');
  if (savedHost) {
    state.SERVER_HOST = savedHost;
    console.log('📂 서버 호스트 불러옴:', savedHost);
  }

  const savedPort = localStorage.getItem('serverPort');
  if (savedPort) {
    state.SERVER_PORT = parseInt(savedPort);
    console.log('📂 서버 포트 불러옴:', savedPort);
  }

  const savedDisplayMode = localStorage.getItem('displayMode');
  if (savedDisplayMode) {
    state.displayMode = savedDisplayMode;
    console.log('📂 표시 모드 불러옴:', savedDisplayMode);
  }
}

function setupEventListeners() {
  // Space 키를 토글 방식으로 변경
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !e.target.matches('input, select')) {
      e.preventDefault();

      // 이미 눌린 상태면 무시 (연속 입력 방지)
      if (e.repeat) return;

      console.log('⌨️ Space 키 누름 - 토글');

      if (state.isRecording) {
        // 녹음 중이면 중지
        console.log('⏸️ 녹음 중지');
        stopRecording();
      } else {
        // 녹음 중이 아니면 시작
        console.log('▶️ 녹음 시작');
        startRecording();
      }
    }
  });
}

function setupElectronIntegration() {
  if (!window.require) return;

  try {
    const { ipcRenderer } = window.require('electron');
    let isOverInteractive = false;
    let dragTimer = null;
    let mouseDownPos = null;
    let isDraggingMode = false;
    let isSettingsOpen = false; // 설정창 열림 상태 추적
    const LONG_PRESS_DURATION = 300;

    // 배경 밝기 감지 및 자동 투명도 조정
    async function detectBackgroundBrightness() {
      if (!state.autoAdjustOpacity) {
        return;
      }

      try {
        const { desktopCapturer, screen } = require('electron');

        // 현재 화면의 실제 크기 가져오기
        const primaryDisplay = screen.getPrimaryDisplay();
        const scaleFactor = primaryDisplay.scaleFactor || 1;
        const screenSize = primaryDisplay.size;

        console.log(`🖥️ 화면 정보: ${screenSize.width}x${screenSize.height}, scaleFactor: ${scaleFactor}`);

        // 화면을 실제 해상도로 캡처
        const sources = await desktopCapturer.getSources({
          types: ['screen'],
          thumbnailSize: {
            width: Math.floor(screenSize.width * scaleFactor),
            height: Math.floor(screenSize.height * scaleFactor)
          }
        });

        if (sources.length === 0) {
          console.warn('❌ 화면 캡처 소스가 없음');
          return;
        }

        const thumbnail = sources[0].thumbnail;
        const imgSize = thumbnail.getSize();
        console.log(`📸 캡처된 이미지 크기: ${imgSize.width}x${imgSize.height}`);

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();

        img.onload = async () => {
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);

          // 패널의 화면상 위치 가져오기 (윈도우 좌표)
          const rect = DOM.mainPanel.getBoundingClientRect();

          // 윈도우의 절대 위치 가져오기
          const windowBounds = await ipcRenderer.invoke('get-window-bounds');

          // 패널의 화면상 절대 위치 계산
          const panelScreenX = windowBounds.x + rect.left;
          const panelScreenY = windowBounds.y + rect.top;

          console.log(`📍 패널 화면 위치: x=${panelScreenX.toFixed(0)}, y=${panelScreenY.toFixed(0)}, w=${rect.width.toFixed(0)}, h=${rect.height.toFixed(0)}`);
          console.log(`📍 윈도우 위치: x=${windowBounds.x}, y=${windowBounds.y}`);

          // 이미지 좌표로 변환 (scaleFactor 고려)
          const imgX = Math.floor((panelScreenX / screenSize.width) * img.width);
          const imgY = Math.floor((panelScreenY / screenSize.height) * img.height);
          const imgW = Math.floor((rect.width / screenSize.width) * img.width);
          const imgH = Math.floor((rect.height / screenSize.height) * img.height);

          console.log(`🎯 이미지 샘플링: x=${imgX}, y=${imgY}, w=${imgW}, h=${imgH} (이미지: ${img.width}x${img.height})`);

          // 유효성 검사
          const safeX = Math.max(0, Math.min(imgX, img.width - 1));
          const safeY = Math.max(0, Math.min(imgY, img.height - 1));
          const safeW = Math.max(10, Math.min(imgW, img.width - safeX));
          const safeH = Math.max(10, Math.min(imgH, img.height - safeY));

          if (safeW <= 0 || safeH <= 0) {
            console.warn('⚠️ 샘플링 영역이 유효하지 않음');
            return;
          }

          console.log(`✅ 안전 샘플링: x=${safeX}, y=${safeY}, w=${safeW}, h=${safeH}`);

          const imageData = ctx.getImageData(safeX, safeY, safeW, safeH);
          const pixels = imageData.data;

          // 평균 밝기 계산
          let totalBrightness = 0;
          const pixelCount = pixels.length / 4;

          for (let i = 0; i < pixels.length; i += 4) {
            const r = pixels[i];
            const g = pixels[i + 1];
            const b = pixels[i + 2];
            const brightness = (r * 0.299 + g * 0.587 + b * 0.114);
            totalBrightness += brightness;
          }

          const avgBrightness = totalBrightness / pixelCount;
          state.backgroundBrightness = avgBrightness;

          console.log(`💡 배경 밝기: ${avgBrightness.toFixed(2)} / 255 (${((avgBrightness/255)*100).toFixed(0)}%)`);

          // 투명도 자동 조정
          const currentOpacity = localStorage.getItem('panelOpacity') || 100;
          updateOpacity(currentOpacity, true);
        };

        img.src = thumbnail.toDataURL();
      } catch (error) {
        console.error('❌ 배경 밝기 감지 실패:', error);
        console.error(error.stack);
      }
    }

    // 자동 투명도가 활성화되어 있으면 즉시 실행
    if (state.autoAdjustOpacity) {
      console.log('🚀 초기 배경 밝기 감지 시작...');
      setTimeout(() => detectBackgroundBrightness(), 1000);
    }

    // 주기적으로 배경 밝기 체크 (3초마다)
    setInterval(() => {
      if (state.autoAdjustOpacity) {
        detectBackgroundBrightness();
      }
    }, 3000);

    document.addEventListener('mousemove', (e) => {
      // 설정창이 열려 있으면 메인 창은 모든 마우스 이벤트 무시
      if (isSettingsOpen) {
        if (isOverInteractive) {
          isOverInteractive = false;
          ipcRenderer.send('set-ignore-mouse-events', true, { forward: true });
        }
        return;
      }

      if (mouseDownPos) {
        if (isDraggingMode) {
          const deltaX = e.screenX - mouseDownPos.screenX;
          const deltaY = e.screenY - mouseDownPos.screenY;
          ipcRenderer.send('move-window', deltaX, deltaY);
          mouseDownPos.screenX = e.screenX;
          mouseDownPos.screenY = e.screenY;
          return;
        }
        if (!isOverInteractive) {
          isOverInteractive = true;
          ipcRenderer.send('set-ignore-mouse-events', false);
        }
        return;
      }

      // 패널 주변에 마진을 두어 더 넓은 클릭 영역 제공
      const rect = DOM.mainPanel.getBoundingClientRect();
      const margin = 30; // 패널 주변 30px까지 클릭 가능
      const isOverPanel = (
        e.clientX >= rect.left - margin &&
        e.clientX <= rect.right + margin &&
        e.clientY >= rect.top - margin &&
        e.clientY <= rect.bottom + margin
      );

      if (isOverPanel !== isOverInteractive) {
        isOverInteractive = isOverPanel;
        ipcRenderer.send('set-ignore-mouse-events', !isOverInteractive, { forward: true });
      }
    });

    DOM.mainPanel.addEventListener('mousedown', (e) => {
      // 설정창이 열려 있으면 메인 패널 클릭 무시
      if (isSettingsOpen) return;

      if (e.button !== 0) return;
      mouseDownPos = { screenX: e.screenX, screenY: e.screenY };
      isDraggingMode = false;

      // 즉시 커서를 grabbing으로 변경
      DOM.mainPanel.style.cursor = 'grabbing';

      if (!isOverInteractive) {
        isOverInteractive = true;
        ipcRenderer.send('set-ignore-mouse-events', false);
      }
      dragTimer = setTimeout(() => {
        if (mouseDownPos) {
          console.log('드래그 모드 시작');
          isDraggingMode = true;
        }
        dragTimer = null;
      }, LONG_PRESS_DURATION);
    });

    document.addEventListener('mouseup', (e) => {
      if (e.button !== 0 || !mouseDownPos) return;
      const wasDragging = isDraggingMode;
      mouseDownPos = null;
      isDraggingMode = false;
      DOM.mainPanel.style.cursor = 'grab';

      if (dragTimer) {
        clearTimeout(dragTimer);
        dragTimer = null;

        console.log('짧은 클릭 -> 설정 열기');
        isSettingsOpen = true; // 설정창 열림 상태 설정
        openSettings();

      } else if (wasDragging) {
        console.log('드래그 종료');
      }

      setTimeout(() => {
        // 마우스업 후에도 패널 주변 영역 확인
        const rect = DOM.mainPanel.getBoundingClientRect();
        const margin = 30;
        const isOverPanelNow = (
          e.clientX >= rect.left - margin &&
          e.clientX <= rect.right + margin &&
          e.clientY >= rect.top - margin &&
          e.clientY <= rect.bottom + margin
        );
        ipcRenderer.send('set-ignore-mouse-events', !isOverPanelNow, { forward: true });
        isOverInteractive = isOverPanelNow;
      }, 10);
    });

    // 설정 창으로부터 변경 사항 수신
    ipcRenderer.on('opacity-changed', (_event, value) => {
      updateOpacity(value);
    });

    ipcRenderer.on('text-size-changed', (_event, value) => {
      updateTextSize(value);
    });

    ipcRenderer.on('server-host-changed', (_event, value) => {
      state.SERVER_HOST = value;
      console.log('💾 서버 호스트 변경됨:', value);
    });

    ipcRenderer.on('server-port-changed', (_event, value) => {
      state.SERVER_PORT = parseInt(value);
      console.log('💾 서버 포트 변경됨:', value);
    });

    ipcRenderer.on('display-mode-changed', (_event, value) => {
      state.displayMode = value;
      console.log('💾 표시 모드 변경됨:', value);
      // 현재 표시된 텍스트를 새로운 모드로 다시 표시
      if (state.currentOriginal || state.currentTranslated) {
        showResult(state.currentOriginal, state.currentTranslated);
      }
    });

    // 설정 창에서 초기 상태 요청 시 응답
    ipcRenderer.on('request-state-for-settings', () => {
      ipcRenderer.send('send-state-to-settings', {
        isServerConnected: state.isServerConnected,
        isRecording: state.isRecording,
        translationHistory: state.translationHistory
      });
    });

    // 설정 창에서 녹음 토글 요청
    ipcRenderer.on('toggle-recording', () => {
      if (state.isRecording) {
        stopRecording();
      } else {
        startRecording();
      }
    });

    // 설정 창이 닫힐 때 이벤트 수신
    ipcRenderer.on('settings-window-closed', () => {
      console.log('설정 창 닫힘 - 메인 창 마우스 이벤트 복원');
      isSettingsOpen = false;
      // 마우스 이벤트 처리 재개
      ipcRenderer.send('set-ignore-mouse-events', true, { forward: true });
    });

    // 자동 투명도 조정 토글
    ipcRenderer.on('auto-opacity-changed', (_event, enabled) => {
      state.autoAdjustOpacity = enabled;
      console.log('💾 자동 투명도 조정:', enabled ? '활성화' : '비활성화');
      if (enabled) {
        detectBackgroundBrightness();
      } else {
        // 비활성화 시 원래 투명도로 복원
        const currentOpacity = localStorage.getItem('panelOpacity') || 100;
        updateOpacity(currentOpacity, false);
      }
    });

    // 윈도우가 닫히기 전에 정리
    window.addEventListener('beforeunload', () => {
      console.log('🧹 윈도우 종료 전 정리 시작');
      // 녹음 정리
      cleanupRecording();
      // 모든 IPC 리스너 제거
      ipcRenderer.removeAllListeners();
    });

  } catch(e) {
    console.log('Electron IPC 사용 불가', e);
  }
}

async function init() {
  console.log('🚀 앱 초기화 시작');
  loadOpacity();
  loadTextSize();
  loadSettings();

  // 자동 투명도 설정 불러오기
  const savedAutoOpacity = localStorage.getItem('autoOpacity');
  if (savedAutoOpacity === 'true') {
    state.autoAdjustOpacity = true;
    console.log('💾 자동 투명도 조정 활성화됨');
  }

  updateServerStatus(false);
  updateRecordingStatus(false);
  setupEventListeners();
  setupElectronIntegration();

  // 앱 시작 시 자동으로 녹음 시작
  console.log('🎬 자동 녹음 시작...');
  await startRecording();

  console.log('✅ 초기화 완료 - Space 키로 녹음 시작/중지를 토글하세요');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}