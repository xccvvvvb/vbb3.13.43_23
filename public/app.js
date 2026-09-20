/* Hyperion browser client. Dependency-free; stores conversation only in localStorage. */
const $ = (selector) => document.querySelector(selector);
const state = {
  sessions: JSON.parse(localStorage.getItem('hyperion.sessions') || '[]'),
  active: null,
  style: localStorage.getItem('hyperion.style') || 'balanced',
  busy: false
};

function save() {
  localStorage.setItem('hyperion.sessions', JSON.stringify(state.sessions.slice(0, 20)));
}

function newSession() {
  const session = { id: crypto.randomUUID(), title: 'Nueva conversación', created: Date.now(), messages: [] };
  state.sessions.unshift(session);
  state.active = session.id;
  save(); render();
  $('#prompt').focus();
}

function activeSession() { return state.sessions.find(s => s.id === state.active); }

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function markdown(text) {
  let safe = escapeHtml(text);
  safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');
  safe = safe.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  return safe.split(/\n\n+/).map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
}

function renderSessions() {
  $('#sessions').innerHTML = state.sessions.map(session =>
    `<button class="${session.id === state.active ? 'active' : ''}" data-id="${session.id}">◇ &nbsp;${escapeHtml(session.title)}</button>`
  ).join('');
}

function renderMessages() {
  const session = activeSession();
  const messages = session?.messages || [];
  $('#hero').hidden = messages.length > 0;
  $('#messages').innerHTML = messages.map((message) => {
    const assistant = message.role === 'assistant';
    const meta = assistant && message.meta
      ? `<div class="meta">CONFIANZA ${Math.round(message.meta.confidence * 100)}% · ${message.meta.latency_ms} MS · ${escapeHtml(message.meta.intent.toUpperCase())}</div>` : '';
    return `<article class="message ${assistant ? 'assistant' : 'user'}">
      <div class="avatar">${assistant ? 'H' : 'TÚ'}</div>
      <div><div class="message-head"><strong>${assistant ? 'HYPERION' : 'OPERADOR'}</strong><time>${new Date(message.time).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}</time></div>
      <div class="message-body">${markdown(message.text)}</div>${meta}</div></article>`;
  }).join('');
  requestAnimationFrame(() => { $('#conversation').scrollTop = $('#conversation').scrollHeight; });
}

function render() { renderSessions(); renderMessages(); }

async function send(text) {
  if (state.busy || !text.trim()) return;
  if (!activeSession()) newSession();
  state.busy = true;
  $('#send').disabled = true;
  const session = activeSession();
  session.messages.push({ role: 'user', text: text.trim(), time: Date.now() });
  if (session.messages.length === 1) session.title = text.trim().slice(0, 34);
  save(); render();
  const holder = document.createElement('article');
  holder.className = 'message assistant';
  holder.innerHTML = '<div class="avatar">H</div><div><div class="message-head"><strong>HYPERION</strong><span class="metric">PROCESANDO LOCALMENTE</span></div><div class="thinking"><i></i><i></i><i></i></div></div>';
  $('#messages').append(holder);
  $('#conversation').scrollTop = $('#conversation').scrollHeight;
  try {
    const response = await fetch('/api/chat', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({message:text, style:state.style}) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Error local');
    session.messages.push({ role: 'assistant', text: data.answer, time: Date.now(), meta: data });
  } catch (error) {
    session.messages.push({ role: 'assistant', text: `**Fallo del núcleo local:** ${error.message}`, time: Date.now(), meta: {confidence:0,latency_ms:0,intent:'error'} });
  } finally {
    state.busy = false; $('#send').disabled = false; save(); render(); $('#prompt').focus();
  }
}

$('#chat-form').addEventListener('submit', event => {
  event.preventDefault(); const field = $('#prompt'); const text = field.value; field.value = ''; field.style.height = ''; $('#counter').textContent = '0 / 4000'; send(text);
});
$('#prompt').addEventListener('input', event => {
  event.target.style.height = 'auto'; event.target.style.height = `${Math.min(event.target.scrollHeight, 130)}px`;
  $('#counter').textContent = `${event.target.value.length} / 4000`;
});
$('#prompt').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); $('#chat-form').requestSubmit(); } });
$('#new-chat').addEventListener('click', newSession);
$('#sessions').addEventListener('click', event => { const button = event.target.closest('[data-id]'); if (button) { state.active = button.dataset.id; render(); } });
document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => send(button.dataset.prompt)));
$('#knowledge-button').addEventListener('click', () => $('#knowledge-dialog').showModal());
$('[data-close]').addEventListener('click', () => $('#knowledge-dialog').close());
$('#knowledge-form').addEventListener('submit', async event => {
  event.preventDefault();
  const status = $('#learn-status');
  status.textContent = 'VECTORZANDO E INDEXANDO EN MEMORIA…';
  try {
    const response = await fetch('/api/learn', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({title:$('#document-title').value, content:$('#document-content').value})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    status.textContent = `✓ ${data.chunks} FRAGMENTOS INTEGRADOS · ${data.documents} NÚCLEOS ACTIVOS`;
    $('#document-content').value = '';
    await refreshStatus();
  } catch (error) { status.textContent = `ERROR: ${error.message}`; }
});
$('#reset-memory').addEventListener('click', async () => {
  await fetch('/api/reset', {method:'POST'});
  $('#learn-status').textContent = 'MEMORIA APRENDIDA ELIMINADA';
  refreshStatus();
});
$('#settings-button').addEventListener('click', () => { $('#style').value = state.style; $('#settings').showModal(); });
$('#settings').addEventListener('close', () => { state.style = $('#style').value; localStorage.setItem('hyperion.style', state.style); });
$('#menu').addEventListener('click', () => $('#sidebar').classList.toggle('open'));
document.addEventListener('keydown', event => { if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); newSession(); } });

async function refreshStatus() {
  try {
    const response = await fetch('/api/status');
    const data = await response.json();
    $('#knowledge-count').textContent = `${data.documents} NÚCLEOS · ${data.learned} APRENDIDOS`;
  } catch (_) { $('.header-status').innerHTML = '<span>●</span> NÚCLEO SIN RESPUESTA'; }
}

if (!state.sessions.length) newSession(); else { state.active = state.sessions[0].id; render(); }
refreshStatus();
