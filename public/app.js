/* Hyperion — cliente de navegador. Sin dependencias; las conversaciones
   se guardan únicamente en localStorage. */

'use strict';

const $ = (selector) => document.querySelector(selector);

const MAX_SESSIONS = 30;
const MAX_CONTENT_BYTES = 490_000; // margen bajo el límite de 512 KB del servidor
const STORAGE = {
  sessions: 'hyperion.sessions',
  active: 'hyperion.active',
  style: 'hyperion.style',
};

/* ---------------------------------------------------------------------------
   Estado con carga defensiva: localStorage corrupto nunca debe romper la app.
   --------------------------------------------------------------------------- */

function loadSessions() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE.sessions) || '[]');
    if (!Array.isArray(raw)) return [];
    return raw
      .filter((s) => s && typeof s.id === 'string' && Array.isArray(s.messages))
      .map((s) => ({
        id: s.id,
        title: typeof s.title === 'string' && s.title.trim() ? s.title : 'Nueva conversación',
        created: Number(s.created) || Date.now(),
        messages: s.messages
          .filter((m) => m && typeof m.text === 'string')
          .map((m) => ({
            role: m.role === 'user' ? 'user' : 'assistant',
            text: m.text,
            time: Number(m.time) || Date.now(),
            meta: m.meta && typeof m.meta === 'object' ? m.meta : null,
          })),
      }));
  } catch {
    return [];
  }
}

const state = {
  sessions: loadSessions(),
  active: localStorage.getItem(STORAGE.active) || null,
  style: localStorage.getItem(STORAGE.style) || 'balanced',
  busy: false,
  learning: false,
};

if (!['concise', 'balanced', 'technical'].includes(state.style)) state.style = 'balanced';
if (!state.sessions.some((s) => s.id === state.active)) state.active = state.sessions[0]?.id || null;

let forceScroll = false;

function save() {
  try {
    localStorage.setItem(STORAGE.sessions, JSON.stringify(state.sessions.slice(0, MAX_SESSIONS)));
    if (state.active) localStorage.setItem(STORAGE.active, state.active);
    else localStorage.removeItem(STORAGE.active);
  } catch {
    /* almacenamiento no disponible: la app sigue funcionando en memoria */
  }
}

/* crypto.randomUUID solo existe en contextos seguros; fallback para LAN/http. */
function uid() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function activeSession() {
  return state.sessions.find((s) => s.id === state.active);
}

/* ---------------------------------------------------------------------------
   Utilidades de renderizado
   --------------------------------------------------------------------------- */

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function markdown(text) {
  const safe = escapeHtml(text);
  return safe
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .split(/\n{2,}/)
    .map((p) => `<p>${p.replace(/\n/g, '<br>')}</p>`)
    .join('');
}

function formatWhen(timestamp) {
  const minutes = (Date.now() - timestamp) / 60_000;
  if (minutes < 1) return 'ahora';
  if (minutes < 60) return `${Math.floor(minutes)} min`;
  if (minutes < 60 * 24) return `${Math.floor(minutes / 60)} h`;
  return new Date(timestamp).toLocaleDateString([], { day: 'numeric', month: 'short' });
}

function renderSessions() {
  $('#sessions').innerHTML = state.sessions.map((session) => `
    <div class="session ${session.id === state.active ? 'active' : ''}">
      <button class="session-open" data-id="${session.id}" title="${escapeHtml(session.title)}">
        <span class="session-title">${escapeHtml(session.title)}</span>
        <span class="session-when">${formatWhen(session.created)}</span>
      </button>
      <button class="session-delete" data-delete="${session.id}" aria-label="Eliminar sesión" title="Eliminar sesión">×</button>
    </div>`).join('');
}

function metaChips(meta) {
  const chips = [];
  if (typeof meta.confidence === 'number' && Number.isFinite(meta.confidence)) {
    chips.push(`${Math.round(meta.confidence * 100)}% confianza`);
  }
  if (Number.isFinite(meta.latency_ms) && meta.latency_ms > 0) chips.push(`${meta.latency_ms} ms`);
  if (meta.intent) chips.push(escapeHtml(String(meta.intent)));
  const sources = Array.isArray(meta.sources) ? meta.sources.filter((s) => typeof s === 'string') : [];
  let sourceChip = '';
  if (sources.length) {
    const label = sources.slice(0, 2).map(escapeHtml).join(' · ') + (sources.length > 2 ? ` +${sources.length - 2}` : '');
    sourceChip = `<span class="chip chip-src">${label}</span>`;
  }
  return `<div class="meta">${chips.map((c) => `<span class="chip">${c}</span>`).join('')}${sourceChip}</div>`;
}

function renderMessage(message) {
  const assistant = message.role === 'assistant';
  const isError = Boolean(assistant && message.meta && message.meta.intent === 'error');
  const head = `
    <div class="message-head">
      <strong>${assistant ? 'Hyperion' : 'Tú'}</strong>
      <time>${new Date(message.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</time>
    </div>`;
  const meta = assistant && message.meta && !isError ? metaChips(message.meta) : '';
  return `
    <article class="message ${assistant ? 'assistant' : 'user'}${isError ? ' error' : ''}">
      <div class="avatar" aria-hidden="true">${assistant ? 'H' : 'Tú'}</div>
      <div class="message-main">${head}<div class="message-body">${markdown(message.text)}</div>${meta}</div>
    </article>`;
}

function renderMessages() {
  const session = activeSession();
  const messages = session?.messages || [];
  $('#hero').hidden = messages.length > 0;
  const container = $('#conversation');
  const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 90;
  $('#messages').innerHTML = messages.map(renderMessage).join('');
  if (forceScroll || nearBottom) container.scrollTop = container.scrollHeight;
  forceScroll = false;
}

function render() {
  renderSessions();
  renderMessages();
}

/* ---------------------------------------------------------------------------
   Sesiones
   --------------------------------------------------------------------------- */

function createSession() {
  const session = { id: uid(), title: 'Nueva conversación', created: Date.now(), messages: [] };
  state.sessions.unshift(session);
  state.active = session.id;
  forceScroll = true;
  save();
  render();
  $('#prompt').focus();
}

function openSession(id) {
  if (state.active !== id) {
    state.active = id;
    forceScroll = true;
    save();
    render();
  }
  closeSidebar();
}

function deleteSession(id) {
  state.sessions = state.sessions.filter((s) => s.id !== id);
  if (state.active === id) state.active = state.sessions[0]?.id || null;
  save();
  if (!state.sessions.length) createSession();
  else render();
}

/* ---------------------------------------------------------------------------
   Envío de mensajes
   --------------------------------------------------------------------------- */

function setBusy(busy) {
  state.busy = busy;
  $('#send').disabled = busy || !$('#prompt').value.trim();
}

async function send(text) {
  text = String(text ?? '').trim();
  if (state.busy || !text) return;
  if (!activeSession()) createSession();

  const session = activeSession();
  state.busy = true;
  setBusy(true);

  session.messages.push({ role: 'user', text, time: Date.now() });
  if (session.messages.length === 1) session.title = text.slice(0, 34);
  forceScroll = true;
  save();
  render();

  const holder = document.createElement('article');
  holder.className = 'message assistant';
  holder.innerHTML = `
    <div class="avatar" aria-hidden="true">H</div>
    <div class="message-main">
      <div class="message-head"><strong>Hyperion</strong></div>
      <div class="thinking"><span class="dots"><i></i><i></i><i></i></span> Procesando localmente…</div>
    </div>`;
  $('#messages').append(holder);
  $('#conversation').scrollTop = $('#conversation').scrollHeight;

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, style: state.style }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Error local');
    session.messages.push({ role: 'assistant', text: data.answer, time: Date.now(), meta: data });
  } catch (error) {
    session.messages.push({
      role: 'assistant',
      text: `**Sin respuesta del núcleo local.** ${error.message}`,
      time: Date.now(),
      meta: { confidence: 0, latency_ms: 0, intent: 'error' },
    });
  } finally {
    setBusy(false);
    save();
    render();
    if (document.activeElement === document.body) $('#prompt').focus();
  }
}

/* ---------------------------------------------------------------------------
   Barra lateral móvil
   --------------------------------------------------------------------------- */

function openSidebar() {
  document.body.classList.add('sidebar-open');
  $('#menu').setAttribute('aria-expanded', 'true');
}

function closeSidebar() {
  document.body.classList.remove('sidebar-open');
  $('#menu').setAttribute('aria-expanded', 'false');
}

/* ---------------------------------------------------------------------------
   Estado del servidor
   --------------------------------------------------------------------------- */

async function refreshStatus() {
  const status = $('#header-status');
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('estado no válido');
    const data = await response.json();
    status.classList.remove('offline');
    $('#header-status-text').textContent = 'Núcleo en línea';
    if (data.version) $('#model-version').textContent = `v${data.version}`;
    $('#knowledge-count').textContent = `${data.documents} núcleos · ${data.learned} aprendidos`;
  } catch {
    status.classList.add('offline');
    $('#header-status-text').textContent = 'Núcleo sin respuesta';
  }
}

function setLearnStatus(text, kind) {
  const el = $('#learn-status');
  el.textContent = text;
  el.className = 'learn-status' + (kind ? ` ${kind}` : '');
}

/* ---------------------------------------------------------------------------
   Eventos
   --------------------------------------------------------------------------- */

const promptField = $('#prompt');

function autogrow() {
  promptField.style.height = 'auto';
  promptField.style.height = `${Math.min(promptField.scrollHeight, 180)}px`;
}

$('#chat-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const text = promptField.value;
  promptField.value = '';
  promptField.style.height = '';
  $('#counter').textContent = '0 / 4000';
  $('#send').disabled = true;
  send(text);
});

promptField.addEventListener('input', () => {
  autogrow();
  $('#counter').textContent = `${promptField.value.length} / 4000`;
  $('#send').disabled = state.busy || !promptField.value.trim();
});

promptField.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    $('#chat-form').requestSubmit();
  }
});

$('#new-chat').addEventListener('click', () => {
  createSession();
  closeSidebar();
});

$('#sessions').addEventListener('click', (event) => {
  const deleteButton = event.target.closest('[data-delete]');
  if (deleteButton) {
    deleteSession(deleteButton.dataset.delete);
    return;
  }
  const openButton = event.target.closest('.session-open[data-id]');
  if (openButton) openSession(openButton.dataset.id);
});

document.querySelectorAll('[data-prompt]').forEach((button) => {
  button.addEventListener('click', () => send(button.dataset.prompt));
});

$('#menu').addEventListener('click', () => {
  if (document.body.classList.contains('sidebar-open')) closeSidebar();
  else openSidebar();
});

$('#backdrop').addEventListener('click', closeSidebar);

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closeSidebar();
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    createSession();
  }
});

/* Diálogo: ampliar conocimiento */

$('#knowledge-button').addEventListener('click', () => {
  setLearnStatus('', '');
  $('#knowledge-dialog').showModal();
});

document.querySelectorAll('[data-close]').forEach((button) => {
  button.addEventListener('click', () => button.closest('dialog')?.close());
});

$('#knowledge-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (state.learning) return;

  const title = $('#document-title').value;
  const content = $('#document-content').value;
  const bytes = new TextEncoder().encode(content).length;
  if (bytes > MAX_CONTENT_BYTES) {
    setLearnStatus(`El contenido pesa ~${Math.round(bytes / 1000)} KB y supera el límite de ${Math.round(MAX_CONTENT_BYTES / 1000)} KB. Divídelo en partes más pequeñas.`, 'error');
    return;
  }

  state.learning = true;
  const submitButton = event.target.querySelector('.done');
  submitButton.disabled = true;
  setLearnStatus('Vectorizando e indexando en memoria…', '');

  try {
    const response = await fetch('/api/learn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, content }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Error local');
    setLearnStatus(`✓ ${data.chunks} fragmentos integrados · ${data.documents} núcleos activos`, 'ok');
    $('#document-title').value = '';
    $('#document-content').value = '';
    await refreshStatus();
  } catch (error) {
    setLearnStatus(`Error: ${error.message}`, 'error');
  } finally {
    state.learning = false;
    submitButton.disabled = false;
  }
});

$('#reset-memory').addEventListener('click', async () => {
  if (!window.confirm('¿Eliminar toda la memoria aprendida? Esta acción no se puede deshacer.')) return;
  try {
    await fetch('/api/reset', { method: 'POST' });
    setLearnStatus('Memoria aprendida eliminada.', 'ok');
  } catch {
    setLearnStatus('No se pudo contactar con el núcleo local.', 'error');
  }
  refreshStatus();
});

/* Diálogo: configuración */

$('#settings-button').addEventListener('click', () => {
  $('#style').value = state.style;
  $('#settings').showModal();
});

$('#settings').addEventListener('close', () => {
  state.style = $('#style').value;
  try {
    localStorage.setItem(STORAGE.style, state.style);
  } catch {
    /* sin almacenamiento: la preferencia vive solo en memoria */
  }
});

/* ---------------------------------------------------------------------------
   Arranque
   --------------------------------------------------------------------------- */

const isApple = /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent);
$('#new-chat-kbd').textContent = isApple ? '⌘ K' : 'Ctrl K';

if (state.sessions.length) render();
else createSession();

refreshStatus();
setInterval(refreshStatus, 60_000);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) refreshStatus();
});
