/* ============================================================
   VNAAP main.js — 共用互動邏輯 + 主題引擎
   ============================================================ */

/* ── 主題引擎 ─────────────────────────────────────────────── *
 * 深色/淺色主題儲存於 localStorage('vnaap-theme')。
 * base.html <head> 內已有一段內嵌 script 在畫面繪製前套用主題，
 * 避免「先深後淺」的畫面閃爍（FOUC）。這裡只負責切換按鈕與
 * 廣播 'vnaap:themechange' 事件，供圖表等元件即時重繪配色。
 * ------------------------------------------------------------ */
const VNAAP_THEME_KEY = 'vnaap-theme';

function vnaapGetTheme() {
  return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
}

function vnaapSetTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  try { localStorage.setItem(VNAAP_THEME_KEY, theme); } catch (e) {}
  document.querySelectorAll('.theme-toggle .tt-icon-sun').forEach(el => {
    el.style.opacity = theme === 'light' ? '1' : '.4';
  });
  document.querySelectorAll('.theme-toggle .tt-icon-moon').forEach(el => {
    el.style.opacity = theme === 'dark' ? '1' : '.4';
  });
  window.dispatchEvent(new CustomEvent('vnaap:themechange', { detail: { theme } }));
}

function vnaapToggleTheme() {
  vnaapSetTheme(vnaapGetTheme() === 'dark' ? 'light' : 'dark');
}

/* 讀取目前主題下實際渲染出的 CSS 變數值，供 Chart.js / 手刻 SVG 使用 */
function vnaapThemeColors() {
  const cs = getComputedStyle(document.documentElement);
  const v = (name) => cs.getPropertyValue(name).trim();
  return {
    text: v('--text-secondary') || '#94a3b8',
    textMuted: v('--text-muted') || '#64748b',
    grid: vnaapGetTheme() === 'light' ? 'rgba(15,23,42,0.08)' : 'rgba(255,255,255,0.06)',
    border: v('--border') || '#1f2531',
    accent: v('--accent') || '#22c55e',
    green: v('--green') || '#22c55e',
    blue: v('--blue') || '#3b82f6',
    red: v('--red') || '#ef4444',
    orange: v('--orange') || '#f97316',
    yellow: v('--yellow') || '#eab308',
    purple: v('--purple') || '#a855f7',
    cardBg: v('--bg-card') || '#12151c',
  };
}

/* 讓所有已建立的 Chart.js 圖表套用當下主題配色的通用邏輯。
   個別頁面的圖表在建立時常把格線/文字顏色寫死成深色主題的值，
   這裡統一在「主題切換當下」與「頁面載入後」都執行一次，
   確保無論使用者是切換主題、或直接以淺色主題重新整理頁面，
   圖表格線與文字都不會變成看不清楚的顏色。
   個別頁面若想精準控制，可自行監聽 'vnaap:themechange' 事件。 */
function vnaapRecolorCharts() {
  if (typeof Chart === 'undefined' || !Chart.instances) return;
  const c = vnaapThemeColors();
  Object.values(Chart.instances).forEach(inst => {
    try {
      const opts = inst.options;
      if (opts.scales) {
        Object.values(opts.scales).forEach(scale => {
          if (scale.ticks) scale.ticks.color = c.textMuted;
          if (scale.grid) scale.grid.color = c.grid;
          if (scale.pointLabels) scale.pointLabels.color = c.text;
          if (scale.angleLines) scale.angleLines.color = c.grid;
        });
      }
      if (opts.plugins && opts.plugins.legend && opts.plugins.legend.labels) {
        opts.plugins.legend.labels.color = c.text;
      }
      inst.update('none');
    } catch (e) { /* 個別圖表若有自訂結構就略過，不影響其他圖表 */ }
  });
}
window.addEventListener('vnaap:themechange', vnaapRecolorCharts);

/* ── VNAAP 共用工具命名空間 ───────────────────────────────── */
const VNAAP = {
  getCookie(name) {
    const match = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
    return match ? decodeURIComponent(match[2]) : '';
  },

  async api(url, options = {}) {
    const opts = Object.assign({
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
    }, options);
    opts.headers = Object.assign({
      'Content-Type': 'application/json',
      'X-CSRFToken': VNAAP.getCookie('csrftoken'),
    }, options.headers || {});

    const res = await fetch(url, opts);
    let data = null;
    try { data = await res.json(); } catch (e) { /* 無 JSON 內容也沒關係 */ }
    if (!res.ok) {
      const err = new Error((data && (data.error || data.detail)) || res.statusText);
      err.status = res.status; err.data = data;
      throw err;
    }
    return data;
  },

  formatBytes(bytes) {
    if (bytes === 0 || bytes === undefined || bytes === null) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
    return (bytes / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1) + ' ' + units[i];
  },

  initDropzone(el, onFiles) {
    if (!el) return;
    ['dragenter', 'dragover'].forEach(ev => el.addEventListener(ev, e => {
      e.preventDefault(); e.stopPropagation(); el.classList.add('active');
    }));
    ['dragleave', 'drop'].forEach(ev => el.addEventListener(ev, e => {
      e.preventDefault(); e.stopPropagation();
      if (ev === 'dragleave') el.classList.remove('active');
    }));
    el.addEventListener('drop', e => {
      el.classList.remove('active');
      const files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length) onFiles(files);
    });
  },
};
window.VNAAP = VNAAP;

/* ── Toast 通知 ───────────────────────────────────────────── */
function showToast(message, type = 'info', duration = 4200) {
  let wrap = document.getElementById('toastWrap');
  if (!wrap) {
    wrap = document.createElement('div');
    wrap.id = 'toastWrap';
    wrap.className = 'toast-wrap';
    document.body.appendChild(wrap);
  }
  const el = document.createElement('div');
  el.className = 'toast toast-' + type;
  el.innerHTML = `<span></span><button class="toast-close">&times;</button>`;
  el.querySelector('span').textContent = message;
  el.querySelector('.toast-close').onclick = () => el.remove();
  wrap.appendChild(el);
  if (duration > 0) setTimeout(() => { if (el.parentElement) el.remove(); }, duration);
  return el;
}
window.showToast = showToast;

/* 頁面載入時已存在的（Django messages 渲染出的）toast 也自動淡出 */
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.toast-wrap .toast').forEach(el => {
    setTimeout(() => { if (el.parentElement) el.style.opacity = '0'; }, 4200);
    setTimeout(() => { if (el.parentElement) el.remove(); }, 4600);
  });
});

/* ── 確認對話框（Promise 版，取代原生 confirm()） ─────────── */
function confirmAction(message, title = '請確認', variant = 'primary') {
  return new Promise(resolve => {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay open';
    overlay.innerHTML = `
      <div class="modal">
        <div class="modal-header">
          <div class="modal-title">${title}</div>
          <button class="modal-close" type="button">&times;</button>
        </div>
        <div class="modal-body"><p style="font-size:.85rem;color:var(--text-secondary);line-height:1.6">${message}</p></div>
        <div class="modal-footer">
          <button class="btn-ghost" type="button" data-act="cancel">取消</button>
          <button class="${variant === 'danger' ? 'btn-danger' : 'btn-primary'}" type="button" data-act="ok">確認</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    const cleanup = (val) => { overlay.remove(); resolve(val); };
    overlay.querySelector('[data-act="ok"]').onclick = () => cleanup(true);
    overlay.querySelector('[data-act="cancel"]').onclick = () => cleanup(false);
    overlay.querySelector('.modal-close').onclick = () => cleanup(false);
    overlay.addEventListener('click', e => { if (e.target === overlay) cleanup(false); });
  });
}
window.confirmAction = confirmAction;

/* ── Sidebar（手機版）─────────────────────────────────────── */
function openSidebar() {
  document.querySelector('.sidebar')?.classList.add('open');
  document.querySelector('.sidebar-overlay')?.classList.add('open');
}
function closeSidebar() {
  document.querySelector('.sidebar')?.classList.remove('open');
  document.querySelector('.sidebar-overlay')?.classList.remove('open');
}
window.closeSidebar = closeSidebar;

/* ── Tabs（[data-tabs] 容器內的 .tab-item / .tab-pane）────── */
document.addEventListener('click', (e) => {
  const item = e.target.closest('[data-tabs] .tab-item');
  if (!item) return;
  const root = item.closest('[data-tabs]');
  const target = item.getAttribute('data-tab');
  root.querySelectorAll(':scope > .tab-nav > .tab-item, .tab-nav .tab-item').forEach(t => t.classList.remove('active'));
  item.classList.add('active');
  root.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id === target));
});

/* ── 可排序表格（[data-sortable] + th[data-sort]） ────────── */
document.addEventListener('click', (e) => {
  const th = e.target.closest('table[data-sortable] th[data-sort]');
  if (!th) return;
  const table = th.closest('table');
  const tbody = table.querySelector('tbody');
  const idx = Array.from(th.parentElement.children).indexOf(th);
  const type = th.getAttribute('data-sort');
  const asc = !th.classList.contains('sort-asc');
  table.querySelectorAll('th[data-sort]').forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
  th.classList.add(asc ? 'sort-asc' : 'sort-desc');

  const rows = Array.from(tbody.querySelectorAll('tr'));
  rows.sort((r1, r2) => {
    let a = r1.children[idx]?.innerText.trim() || '';
    let b = r2.children[idx]?.innerText.trim() || '';
    if (type === 'number') { a = parseFloat(a.replace(/[^0-9.-]/g, '')) || 0; b = parseFloat(b.replace(/[^0-9.-]/g, '')) || 0; }
    else if (type === 'date') { a = new Date(a).getTime() || 0; b = new Date(b).getTime() || 0; }
    if (a < b) return asc ? -1 : 1;
    if (a > b) return asc ? 1 : -1;
    return 0;
  });
  rows.forEach(r => tbody.appendChild(r));
});

/* ── 數字滾動動畫（[data-animate-count]） ─────────────────── */
function animateCountEl(el) {
  const target = parseFloat(el.getAttribute('data-animate-count')) || 0;
  const duration = 900;
  const start = performance.now();
  function tick(now) {
    const p = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - p, 3);
    const val = target * eased;
    el.textContent = Number.isInteger(target) ? Math.round(val).toLocaleString() : val.toFixed(1);
    if (p < 1) requestAnimationFrame(tick);
    else el.textContent = target.toLocaleString();
  }
  requestAnimationFrame(tick);
}
function initCountAnimations() {
  const els = document.querySelectorAll('[data-animate-count]');
  if (!els.length) return;
  const io = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) { animateCountEl(entry.target); io.unobserve(entry.target); }
    });
  }, { threshold: 0.3 });
  els.forEach(el => io.observe(el));
}

/* ── 標頭時鐘 ──────────────────────────────────────────────── */
function tickClock() {
  const el = document.getElementById('headerClock');
  if (!el) return;
  el.textContent = new Date().toLocaleTimeString('zh-TW', { hour12: false });
}

/* ── 初始化 ───────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  tickClock();
  setInterval(tickClock, 1000);
  initCountAnimations();
  // 頁面內的圖表通常在此事件前就已建立完成（inline script 先於 DOMContentLoaded 執行），
  // 這裡補上一次配色校正，修正「以淺色主題重新整理頁面」時圖表格線寫死深色的問題。
  requestAnimationFrame(vnaapRecolorCharts);

  document.getElementById('hamburgerBtn')?.addEventListener('click', () => {
    const sb = document.querySelector('.sidebar');
    sb && sb.classList.contains('open') ? closeSidebar() : openSidebar();
  });

  document.querySelectorAll('.theme-toggle').forEach(btn => {
    btn.addEventListener('click', vnaapToggleTheme);
  });
});
