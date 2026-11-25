'use strict';
/*
 * 층간소음 기록기
 *  - 측정 경로: 마이크 원본(OS 노이즈 억제·자동 이득 OFF) → FFT → A/Z 가중 레벨 (증폭 없음)
 *  - 청취/녹음 경로: 원본 → 저주파 쉘프/피크 부스트 → (저역 통과) → 증폭 → 컴프레서 → 리미터
 *  - 녹음: 원본 스트림 + 증폭 스트림을 동시에 MediaRecorder로 저장
 *  - 진동: DeviceMotion 가속도(중력 제거) 피크
 */

const $ = id => document.getElementById(id);
const LIMITS = { day: { leq: 39, lmax: 57 }, night: { leq: 34, lmax: 52 } };
const FFT_N = 8192;
const TICK_MS = 100;

const state = {
  ctx: null, stream: null, analyser: null, timeBuf: null, nodes: {},
  running: false, startTime: 0,
  frames: [],          // 현재 초에 쌓인 프레임 {a, z, lf}(에너지)
  seconds: [],         // 초 단위 로그
  minuteBlocks: [],    // 완료된 1분 블록
  sessEnergy: 0, sessCount: 0, sessMax: -Infinity,
  exceedMinutes: 0,
  vibPeak: 0, vibSecPeak: 0, vibBaseline: null, lastVib: { x: 0, y: 0, z: 0 }, vibAvailable: false,
  events: [], lastEventTime: 0,
  recorders: [], recStart: 0, recording: false, files: [],
  wakeLock: null,
};

/* ---------- 설정 ---------- */
const cfg = {
  get cal() { return parseFloat($('cal').value) || 0; },
  get trig() { return +$('trig').value; },
};
function loadSettings() {
  try {
    const s = JSON.parse(localStorage.getItem('fn-settings') || '{}');
    for (const k of ['cal', 'boost', 'gain', 'trig']) if (s[k] != null) $(k).value = s[k];
    for (const k of ['lfOnly', 'calDone']) if (s[k] != null) $(k).checked = s[k];
  } catch (e) { /* 저장소 사용 불가 시 기본값 */ }
}
function saveSettings() {
  try {
    localStorage.setItem('fn-settings', JSON.stringify({
      cal: $('cal').value, boost: $('boost').value, gain: $('gain').value, trig: $('trig').value,
      lfOnly: $('lfOnly').checked, calDone: $('calDone').checked,
    }));
  } catch (e) { /* ignore */ }
}

/* ---------- FFT & 가중치 ---------- */
let fftRe, fftIm, hann, hannU, rev, wA, wLF, wZ, binHz;
function initFFT(sampleRate) {
  const N = FFT_N;
  fftRe = new Float64Array(N); fftIm = new Float64Array(N);
  hann = new Float64Array(N); let u = 0;
  for (let i = 0; i < N; i++) { hann[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1)); u += hann[i] * hann[i]; }
  hannU = u / N;
  rev = new Uint32Array(N); const bits = Math.log2(N);
  for (let i = 0; i < N; i++) { let r = 0; for (let b = 0; b < bits; b++) r = (r << 1) | ((i >> b) & 1); rev[i] = r; }
  binHz = sampleRate / N;
  wA = new Float64Array(N / 2); wLF = new Float64Array(N / 2); wZ = new Float64Array(N / 2);
  for (let k = 1; k < N / 2; k++) {
    const f = k * binHz, f2 = f * f;
    const ra = (12194 ** 2 * f2 * f2) /
      ((f2 + 20.6 ** 2) * Math.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2));
    const aDb = 20 * Math.log10(ra) + 2.0;
    wA[k] = Math.pow(10, aDb / 10);               // 파워 가중
    wZ[k] = (f >= 10 && f <= 20000) ? 1 : 0;
    wLF[k] = (f >= 20 && f <= 200) ? 1 : 0;
  }
}
function fft() {
  const N = FFT_N, re = fftRe, im = fftIm;
  for (let i = 0; i < N; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let size = 2; size <= N; size <<= 1) {
    const half = size >> 1, ang = -2 * Math.PI / size;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let s = 0; s < N; s += size) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const a = s + k, b = a + half;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}
const toDb = e => 10 * Math.log10(Math.max(e, 1e-20)) + cfg.cal;
const fmt = v => (v == null || !isFinite(v)) ? '--' : v.toFixed(1);

/* ---------- 오디오 시작 ---------- */
async function start() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false, noiseSuppression: false, autoGainControl: false,
        channelCount: 1,
        // 일부 브라우저 전용 플래그 (무시되어도 무방)
        googEchoCancellation: false, googNoiseSuppression: false, googAutoGainControl: false, googHighpassFilter: false,
      },
    });
  } catch (e) {
    alert('마이크 권한이 필요합니다: ' + e.message);
    return;
  }
  const ctx = state.ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
  await ctx.resume();
  initFFT(ctx.sampleRate);

  const src = ctx.createMediaStreamSource(state.stream);

  // 측정 경로 (무가공)
  const an = state.analyser = ctx.createAnalyser();
  an.fftSize = FFT_N; an.smoothingTimeConstant = 0;
  state.timeBuf = new Float32Array(FFT_N);
  src.connect(an);

  // 청취/녹음 경로 (저주파 증폭)
  const n = state.nodes;
  n.hp = ctx.createBiquadFilter(); n.hp.type = 'highpass'; n.hp.frequency.value = 18; n.hp.Q.value = 0.7;
  n.shelf = ctx.createBiquadFilter(); n.shelf.type = 'lowshelf'; n.shelf.frequency.value = 200;
  n.peak = ctx.createBiquadFilter(); n.peak.type = 'peaking'; n.peak.frequency.value = 60; n.peak.Q.value = 0.9;
  n.lp = ctx.createBiquadFilter(); n.lp.type = 'lowpass'; n.lp.Q.value = 0.7;
  n.lp2 = ctx.createBiquadFilter(); n.lp2.type = 'lowpass'; n.lp2.Q.value = 0.7;
  n.gain = ctx.createGain();
  n.comp = ctx.createDynamicsCompressor();
  n.comp.threshold.value = -42; n.comp.knee.value = 12; n.comp.ratio.value = 6; n.comp.attack.value = 0.003; n.comp.release.value = 0.3;
  n.makeup = ctx.createGain(); n.makeup.gain.value = 2.0;
  n.limit = ctx.createDynamicsCompressor();
  n.limit.threshold.value = -2; n.limit.knee.value = 0; n.limit.ratio.value = 20; n.limit.attack.value = 0.001; n.limit.release.value = 0.1;
  n.dest = ctx.createMediaStreamDestination();
  n.monitor = ctx.createGain(); n.monitor.gain.value = 0;
  src.connect(n.hp); n.hp.connect(n.shelf); n.shelf.connect(n.peak); n.peak.connect(n.lp); n.lp.connect(n.lp2);
  n.lp2.connect(n.gain); n.gain.connect(n.comp); n.comp.connect(n.makeup); n.makeup.connect(n.limit);
  n.limit.connect(n.dest); n.limit.connect(n.monitor); n.monitor.connect(ctx.destination);
  applyEnhance();

  await startMotion();
  requestWakeLock();

  state.running = true; state.startTime = Date.now();
  state.frames = []; state.seconds = []; state.minuteBlocks = [];
  state.sessEnergy = 0; state.sessCount = 0; state.sessMax = -Infinity; state.exceedMinutes = 0;
  state.events = []; $('events').innerHTML = '';
  state.timer = setInterval(tick, TICK_MS);
  requestAnimationFrame(drawSpec);

  $('btnStart').disabled = true; $('btnRec').disabled = false; $('btnStop').disabled = false;
  $('btnCsv').disabled = false; $('btnReport').disabled = false;
  setStatus('측정 중');
}

function applyEnhance() {
  const n = state.nodes; if (!n.shelf) return;
  const boost = +$('boost').value, gain = +$('gain').value;
  $('tBoost').textContent = boost; $('tGain').textContent = gain;
  n.shelf.gain.value = boost;
  n.peak.gain.value = boost * 0.5;
  const lf = $('lfOnly').checked;
  n.lp.frequency.value = lf ? 500 : 16000;
  n.lp2.frequency.value = lf ? 500 : 16000;
  n.gain.gain.value = Math.pow(10, gain / 20);
  n.monitor.gain.value = $('monitor').checked ? 1 : 0;
  saveSettings();
}

/* ---------- 측정 루프 ---------- */
let lastSpecCol = null;
function tick() {
  const an = state.analyser, buf = state.timeBuf;
  an.getFloatTimeDomainData(buf);
  for (let i = 0; i < FFT_N; i++) { fftRe[i] = buf[i] * hann[i]; fftIm[i] = 0; }
  fft();
  let eA = 0, eZ = 0, eLF = 0;
  const half = FFT_N / 2;
  const col = new Float32Array(Math.min(half, Math.ceil(500 / binHz)));
  for (let k = 1; k < half; k++) {
    const p = fftRe[k] * fftRe[k] + fftIm[k] * fftIm[k];
    eA += p * wA[k]; eZ += p * wZ[k]; eLF += p * wLF[k];
    if (k < col.length) col[k] = p;
  }
  // Parseval: 단측 스펙트럼 → 평균제곱
  const norm = 2 / (FFT_N * FFT_N * hannU);
  eA *= norm; eZ *= norm; eLF *= norm;
  lastSpecCol = { col, norm };

  state.frames.push({ a: eA, z: eZ, lf: eLF });
  const la = toDb(eA), lf = toDb(eLF);
  $('vLA').textContent = fmt(la);
  $('meterBar').style.width = Math.max(0, Math.min(100, (la - 20) / 60 * 100)) + '%';
  $('vLF').textContent = fmt(lf);

  // 사건 감지
  const now = Date.now();
  const vibHit = state.vibAvailable && state.vibPeak > vibThreshold();
  if ((la >= cfg.trig || vibHit) && now - state.lastEventTime > 2000) {
    state.lastEventTime = now;
    // 1초 동안 피크를 모아 기록
    const evt = { t: now, lmax: la, lf, vib: state.vibPeak, memo: '' };
    state.pendingEvent = evt;
    setTimeout(() => { addEvent(evt); state.pendingEvent = null; }, 1000);
  }
  if (state.pendingEvent) {
    state.pendingEvent.lmax = Math.max(state.pendingEvent.lmax, la);
    state.pendingEvent.lf = Math.max(state.pendingEvent.lf, lf);
    state.pendingEvent.vib = Math.max(state.pendingEvent.vib, state.vibPeak);
  }
  state.vibSecPeak = Math.max(state.vibSecPeak, state.vibPeak);
  $('vVib').textContent = state.vibAvailable ? state.vibPeak.toFixed(0) : '--';
  state.vibPeak *= 0.85;

  // 1초 집계
  const elapsed = Math.floor((now - state.startTime) / 1000);
  if (elapsed > state.seconds.length) closeSecond(now);
  updatePeriod();
  const s = elapsed % 60;
  $('vDur').textContent = `${Math.floor(elapsed / 60)}:${String(s).padStart(2, '0')}`;
}

function closeSecond(now) {
  const fr = state.frames; state.frames = [];
  if (!fr.length) return;
  const mean = key => fr.reduce((s, f) => s + f[key], 0) / fr.length;
  const maxA = Math.max(...fr.map(f => f.a));
  const sec = {
    t: now, eA: mean('a'), eZ: mean('z'), eLF: mean('lf'), maxA,
    vib: state.vibSecPeak, period: periodOf(new Date(now)),
  };
  state.vibSecPeak = 0;
  state.seconds.push(sec);
  state.sessEnergy += sec.eA; state.sessCount++;
  state.sessMax = Math.max(state.sessMax, maxA);

  const last60 = state.seconds.slice(-60);
  const leq = toDb(last60.reduce((s, x) => s + x.eA, 0) / last60.length);
  const lmax = toDb(Math.max(...last60.map(x => x.maxA)));
  const lim = LIMITS[sec.period];
  setVal('vLeq', leq, lim.leq); setVal('vLmax', lmax, lim.lmax);
  $('limLeq').textContent = `기준 ${lim.leq} dB(A)`; $('limLmax').textContent = `기준 ${lim.lmax} dB(A)`;
  $('vSess').textContent = `${fmt(toDb(state.sessEnergy / state.sessCount))} / ${fmt(toDb(state.sessMax))}`;

  if (state.seconds.length % 60 === 0) {
    const blk = { t: now, leq, lmax, period: sec.period,
      over: leq > lim.leq || lmax > lim.lmax };
    state.minuteBlocks.push(blk);
    if (blk.over) state.exceedMinutes++;
    $('vExceed').textContent = state.exceedMinutes;
  }
}
function setVal(id, v, lim) {
  const el = $(id); el.textContent = fmt(v);
  el.classList.toggle('over', v > lim); el.classList.toggle('near', v <= lim && v > lim - 3);
}
function periodOf(d) { const h = d.getHours(); return (h >= 6 && h < 22) ? 'day' : 'night'; }
function updatePeriod() {
  const p = periodOf(new Date());
  $('vPeriod').textContent = p === 'day' ? '주간' : '야간';
  $('vPeriodNote').textContent = p === 'day' ? '06–22시: Leq 39 / Lmax 57' : '22–06시: Leq 34 / Lmax 52';
}

/* ---------- 스펙트로그램 ---------- */
function drawSpec() {
  if (!state.running) return;
  const cv = $('spec'), g = cv.getContext('2d');
  const W = cv.clientWidth * devicePixelRatio | 0, H = cv.height;
  if (cv.width !== W) cv.width = W;
  if (lastSpecCol) {
    const { col, norm } = lastSpecCol; lastSpecCol = null;
    g.drawImage(cv, -2, 0);
    const n = col.length;
    for (let y = 0; y < H; y++) {
      const k = Math.max(1, Math.round((1 - y / H) * (n - 1)));
      const db = toDb(col[k] * norm);
      const v = Math.max(0, Math.min(1, (db - 0) / 60));
      g.fillStyle = heat(v);
      g.fillRect(W - 2, y, 2, 1);
    }
  }
  requestAnimationFrame(drawSpec);
}
function heat(v) {
  const r = Math.round(255 * Math.min(1, v * 1.8));
  const gg = Math.round(255 * Math.max(0, Math.min(1, v * 1.8 - 0.6)));
  const b = Math.round(120 * Math.max(0, 0.6 - v));
  return `rgb(${r},${gg},${b})`;
}

/* ---------- 진동 ---------- */
function vibThreshold() { return 15; } // mg (바닥에 놓았을 때 발망치·물건 낙하 수준)
async function startMotion() {
  if (!('DeviceMotionEvent' in window)) { $('vibNote').textContent = '이 기기는 진동 센서를 지원하지 않음'; return; }
  try {
    if (typeof DeviceMotionEvent.requestPermission === 'function') {
      const r = await DeviceMotionEvent.requestPermission();
      if (r !== 'granted') { $('vibNote').textContent = '진동 센서 권한 거부됨'; return; }
    }
  } catch (e) { /* ignore */ }
  window.addEventListener('devicemotion', onMotion);
}
function onMotion(e) {
  const a = e.accelerationIncludingGravity; if (!a || a.x == null) return;
  // 간단한 고역 통과로 중력·기울기 제거 → 순간 충격만
  const lp = state.lastVib, alpha = 0.9;
  lp.x = alpha * lp.x + (1 - alpha) * a.x; lp.y = alpha * lp.y + (1 - alpha) * a.y; lp.z = alpha * lp.z + (1 - alpha) * a.z;
  const dx = a.x - lp.x, dy = a.y - lp.y, dz = a.z - lp.z;
  const mg = Math.sqrt(dx * dx + dy * dy + dz * dz) / 9.80665 * 1000;
  if (!state.vibAvailable) { state.vibAvailable = true; $('vibNote').textContent = 'mg (1/1000 g) · 바닥에 놓고 측정'; }
  state.vibWarm = (state.vibWarm || 0) + 1;
  if (state.vibWarm < 60) return; // 초기 필터 안정화
  state.vibPeak = Math.max(state.vibPeak, mg);
}

/* ---------- 사건 기록 ---------- */
function addEvent(evt) {
  state.events.push(evt);
  const recPos = state.recording ? hms((evt.t - state.recStart) / 1000) : '-';
  evt.recPos = recPos;
  const tr = document.createElement('tr');
  const lim = LIMITS[periodOf(new Date(evt.t))];
  tr.innerHTML = `<td>${new Date(evt.t).toLocaleTimeString('ko-KR')}</td>
    <td class="${evt.lmax > lim.lmax ? 'over' : ''}">${fmt(evt.lmax)}</td>
    <td>${fmt(evt.lf)}</td><td>${state.vibAvailable ? evt.vib.toFixed(0) : '-'}</td><td>${recPos}</td>`;
  if (evt.memo) tr.title = evt.memo;
  $('events').prepend(tr);
  if (evt.memo) {
    const m = document.createElement('tr');
    m.innerHTML = `<td colspan="5" class="note">↳ ${escapeHtml(evt.memo)}</td>`;
    tr.after(m);
  }
}
function escapeHtml(s) { return s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
function hms(sec) { sec = Math.max(0, Math.floor(sec)); const h = Math.floor(sec / 3600), m = Math.floor(sec / 60) % 60, s = sec % 60; return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`; }

/* ---------- 녹음 ---------- */
function pickMime() {
  const c = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/ogg;codecs=opus', 'audio/webm'];
  return c.find(m => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || '';
}
function startRec() {
  const mime = pickMime();
  const opts = mime ? { mimeType: mime, audioBitsPerSecond: 256000 } : {};
  const make = (stream, label) => {
    const r = new MediaRecorder(stream, opts), chunks = [];
    r.ondataavailable = e => e.data.size && chunks.push(e.data);
    r.onstop = () => saveRecording(new Blob(chunks, { type: r.mimeType || mime }), label);
    r.start(1000);
    return r;
  };
  state.recStart = Date.now();
  state.recorders = [make(state.stream, '원본'), make(state.nodes.dest.stream, '저주파증폭')];
  state.recording = true;
  $('btnRec').textContent = '■ 녹음 정지'; setStatus('녹음 중', true);
}
function stopRec() {
  state.recorders.forEach(r => r.state !== 'inactive' && r.stop());
  state.recorders = []; state.recording = false;
  $('btnRec').textContent = '● 녹음 시작'; setStatus(state.running ? '측정 중' : '대기');
}
async function saveRecording(blob, label) {
  const ext = (blob.type.includes('mp4') ? 'm4a' : blob.type.includes('ogg') ? 'ogg' : 'webm');
  const name = `층간소음_${stamp(new Date(state.recStart))}_${label}.${ext}`;
  const hash = await sha256(blob);
  state.files.push({ name, hash, size: blob.size, label, start: state.recStart, end: Date.now() });
  addFileLink(blob, name, `SHA-256 ${hash.slice(0, 16)}… · ${(blob.size / 1048576).toFixed(1)}MB`);
}
function addFileLink(blob, name, meta) {
  const box = $('files'); const ph = $('filesEmpty'); if (ph) ph.remove();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name;
  a.innerHTML = `⬇ ${escapeHtml(name)}<br><span class="note">${meta}</span>`;
  box.appendChild(a);
}
async function sha256(blob) {
  if (!crypto.subtle) return '(지원 안 됨)';
  const d = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer());
  return [...new Uint8Array(d)].map(b => b.toString(16).padStart(2, '0')).join('');
}
function stamp(d) {
  const p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

/* ---------- 내보내기 ---------- */
function exportCsv() {
  const rows = [['시각', '시간대', 'LAeq_1s(dBA)', 'LAmax_1s(dBA)', 'LZeq_1s(dB)', 'L_20-200Hz(dB)', '진동피크(mg)']];
  for (const s of state.seconds) rows.push([
    new Date(s.t).toLocaleString('ko-KR'), s.period === 'day' ? '주간' : '야간',
    fmt(toDb(s.eA)), fmt(toDb(s.maxA)), fmt(toDb(s.eZ)), fmt(toDb(s.eLF)), state.vibAvailable ? s.vib.toFixed(1) : '',
  ]);
  rows.push([]); rows.push(['[1분 단위 평가]', '시간대', 'Leq_1min', 'Lmax', '기준 초과']);
  for (const b of state.minuteBlocks) rows.push([new Date(b.t).toLocaleString('ko-KR'), b.period === 'day' ? '주간' : '야간', fmt(b.leq), fmt(b.lmax), b.over ? '초과' : '']);
  rows.push([]); rows.push(['[사건]', 'Lmax', '저주파', '진동(mg)', '녹음위치', '메모']);
  for (const e of state.events) rows.push([new Date(e.t).toLocaleString('ko-KR'), fmt(e.lmax), fmt(e.lf), e.vib.toFixed(0), e.recPos, e.memo]);
  const csv = '﻿' + rows.map(r => r.map(v => `"${String(v ?? '').replace(/"/g, '""')}"`).join(',')).join('\r\n');
  download(new Blob([csv], { type: 'text/csv' }), `층간소음_로그_${stamp(new Date(state.startTime))}.csv`);
}
function exportReport() {
  const nB = state.minuteBlocks.length;
  const lmaxOverByHour = {};
  for (const s of state.seconds) {
    if (toDb(s.maxA) > LIMITS[s.period].lmax) {
      const h = new Date(s.t); h.setMinutes(0, 0, 0);
      lmaxOverByHour[h.toLocaleString('ko-KR')] = (lmaxOverByHour[h.toLocaleString('ko-KR')] || 0) + 1;
    }
  }
  const lines = [
    '층간소음 자가 측정 기록 (참고용 — 공인 측정 아님)',
    '='.repeat(40),
    `측정 기간: ${new Date(state.startTime).toLocaleString('ko-KR')} ~ ${new Date().toLocaleString('ko-KR')}`,
    `측정 기기: ${navigator.userAgent}`,
    `보정값: ${cfg.cal} (${$('calDone').checked ? '기준기와 비교 보정함' : '미보정 — 수치는 참고용'})`,
    `마이크 처리: 에코 제거/노이즈 억제/자동 이득 해제 요청`,
    '',
    `전체 등가소음도 Leq: ${fmt(toDb(state.sessEnergy / state.sessCount))} dB(A)`,
    `전체 최고소음도 Lmax: ${fmt(toDb(state.sessMax))} dB(A)`,
    `평가한 1분 구간: ${nB}개, 기준 초과 구간: ${state.exceedMinutes}개`,
    '',
    '[시간대별 Lmax 기준 초과 (초 단위 횟수)]',
    ...Object.entries(lmaxOverByHour).map(([h, c]) => `  ${h} 시대: ${c}회`),
    '',
    '[기준 초과 1분 구간]',
    ...state.minuteBlocks.filter(b => b.over).map(b => `  ${new Date(b.t).toLocaleString('ko-KR')}  Leq ${fmt(b.leq)} / Lmax ${fmt(b.lmax)} (${b.period === 'day' ? '주간' : '야간'})`),
    '',
    `[사건 기록 ${state.events.length}건]`,
    ...state.events.map(e => `  ${new Date(e.t).toLocaleString('ko-KR')}  Lmax ${fmt(e.lmax)}  저주파 ${fmt(e.lf)}  진동 ${e.vib.toFixed(0)}mg  녹음 ${e.recPos}  ${e.memo || ''}`),
    '',
    '[녹음 파일 및 무결성 해시 (SHA-256)]',
    ...state.files.map(f => `  ${f.name}\n    ${f.hash}\n    ${new Date(f.start).toLocaleString('ko-KR')} ~ ${new Date(f.end).toLocaleString('ko-KR')}`),
    '',
    '※ 기준: 공동주택 층간소음의 범위와 기준에 관한 규칙(2023.1.2 시행) 직접충격소음',
    '   1분 Leq 주간 39 / 야간 34 dB(A), Lmax 주간 57 / 야간 52 dB(A)',
    '※ "저주파증폭" 파일은 청취 편의를 위해 가공된 것이며 소음도 수치 판단에는 "원본" 파일을 사용해야 합니다.',
  ];
  download(new Blob(['﻿' + lines.join('\r\n')], { type: 'text/plain' }), `층간소음_보고서_${stamp(new Date(state.startTime))}.txt`);
}
function download(blob, name) {
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
}

/* ---------- 기타 ---------- */
async function requestWakeLock() {
  try { if ('wakeLock' in navigator) state.wakeLock = await navigator.wakeLock.request('screen'); } catch (e) { /* ignore */ }
}
document.addEventListener('visibilitychange', () => { if (state.running && document.visibilityState === 'visible') requestWakeLock(); });
function setStatus(t, live) { const s = $('status'); s.textContent = t; s.classList.toggle('live', !!live); }
function stopAll() {
  if (state.recording) stopRec();
  state.running = false; clearInterval(state.timer);
  window.removeEventListener('devicemotion', onMotion);
  state.stream && state.stream.getTracks().forEach(t => t.stop());
  state.ctx && state.ctx.close();
  try { state.wakeLock && state.wakeLock.release(); } catch (e) { /* ignore */ }
  $('btnStart').disabled = false; $('btnRec').disabled = true; $('btnStop').disabled = true;
  setStatus('정지');
}
function updateCalBadge() {
  const b = $('calBadge'); const done = $('calDone').checked;
  b.textContent = done ? '보정됨' : '미보정 · 참고값'; b.style.background = done ? '#1f3d2e' : '';
  $('tTrig').textContent = $('trig').value;
  saveSettings();
}

loadSettings();
$('btnStart').onclick = start;
$('btnStop').onclick = stopAll;
$('btnRec').onclick = () => state.recording ? stopRec() : startRec();
$('btnCsv').onclick = exportCsv;
$('btnReport').onclick = exportReport;
$('btnMark').onclick = () => {
  if (!state.running) return;
  const la = state.seconds.length ? toDb(state.seconds[state.seconds.length - 1].maxA) : NaN;
  addEvent({ t: Date.now(), lmax: la, lf: state.seconds.length ? toDb(state.seconds[state.seconds.length - 1].eLF) : NaN, vib: state.vibPeak, memo: $('memo').value || '수동 기록' });
  $('memo').value = '';
};
['boost', 'gain', 'lfOnly', 'monitor'].forEach(id => $(id).addEventListener('input', applyEnhance));
['cal', 'calDone', 'trig'].forEach(id => $(id).addEventListener('input', updateCalBadge));
$('calUp').onclick = () => { $('cal').value = cfg.cal + 1; updateCalBadge(); };
$('calDown').onclick = () => { $('cal').value = cfg.cal - 1; updateCalBadge(); };
updateCalBadge(); applyEnhance(); updatePeriod();
if ('serviceWorker' in navigator && location.protocol === 'https:') navigator.serviceWorker.register('sw.js').catch(() => {});
