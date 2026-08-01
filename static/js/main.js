// ============================================================
// main.js — VNAAP 全域工具函式
// 視覺化網路攻擊自動化分析平台
// ============================================================

(function () {
  'use strict';

  // ── 1. CSRF Token 處理 ────────────────────────────────────
  /**
   * 從 cookie 中取得指定名稱的值
   * Django 的 CSRF token 存放在名為 csrftoken 的 cookie 中
   */
  function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
      const cookies = document.cookie.split(';');
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.substring(0, name.length + 1) === (name + '=')) {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
  }

  // 全域 CSRF token
  const csrfToken = getCookie('csrftoken');

  // ── 2. API Fetch 封裝 ─────────────────────────────────────
  /**
   * 統一的 API 請求封裝
   * 自動附加 CSRF token 和 JSON headers
   * 401 回應自動跳轉登入頁
   */
  async function api(url, options = {}) {
    const defaults = {
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrfToken,
        'X-Requested-With': 'XMLHttpRequest',
      },
      credentials: 'same-origin',
    };

    const config = {
      ...defaults,
      ...options,
      headers: { ...defaults.headers, ...(options.headers || {}) },
    };

    // FormData 時移除 Content-Type（讓瀏覽器自動設定 boundary）
    if (config.body instanceof FormData) {
      delete config.headers['Content-Type'];
    }

    try {
      const response = await fetch(url, config);

      // 401 → 跳轉登入頁
      if (response.status === 401) {
        window.location.href = '/accounts/login/';
        return null;
      }

      // 非 2xx 回應
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw { status: response.status, data: errorData };
      }

      // 204 No Content
      if (response.status === 204) return null;

      return await response.json();
    } catch (err) {
      if (err.status) throw err;
      console.error('[API] 網路錯誤:', err);
      throw { status: 0, data: { detail: '網路連線失敗' } };
    }
  }

  // 快捷方法
  api.get = (url) => api(url, { method: 'GET' });
  api.post = (url, data) => api(url, { method: 'POST', body: JSON.stringify(data) });
  api.put = (url, data) => api(url, { method: 'PUT', body: JSON.stringify(data) });
  api.delete = (url) => api(url, { method: 'DELETE' });
  api.upload = (url, formData) => api(url, { method: 'POST', body: formData });

  // ── 3. Toast 通知系統 ─────────────────────────────────────
  /**
   * 顯示 toast 通知
   * @param {string} message - 訊息內容
   * @param {string} type    - 類型: success | error | warning | info
   * @param {number} duration - 自動消失時間（毫秒）
   */
  function showToast(message, type = 'info', duration = 4000) {
    // 確保容器存在
    let wrap = document.getElementById('toastWrap');
    if (!wrap) {
      wrap = document.createElement('div');
      wrap.id = 'toastWrap';
      wrap.className = 'toast-wrap';
      document.body.appendChild(wrap);
    }

    // 建立 toast 元素
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
      <span>${message}</span>
      <button class="toast-close" onclick="this.parentElement.remove()">&times;</button>
    `;
    wrap.appendChild(toast);

    // 自動消失
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(20px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, duration);

    // 點擊關閉
    toast.addEventListener('click', () => toast.remove());
  }

  // ── 4. Sidebar 響應式控制 ──────────────────────────────────
  function toggleSidebar() {
    document.body.classList.toggle('sidebar-open');
  }

  function closeSidebar() {
    document.body.classList.remove('sidebar-open');
  }

  // ── 5. 時鐘更新 ────────────────────────────────────────────
  function startClock() {
    const el = document.getElementById('headerClock');
    if (!el) return;

    function tick() {
      const now = new Date();
      el.textContent = now.toLocaleTimeString('zh-TW', { hour12: false }) + ' UTC+8';
      requestAnimationFrame(() => setTimeout(tick, 1000));
    }
    tick();
  }

  // ── 6. Django Messages Toast 自動消失 ──────────────────────
  function initDjangoMessages() {
    const wrap = document.getElementById('toastWrap');
    if (!wrap) return;
    setTimeout(() => {
      wrap.style.opacity = '0';
      wrap.style.transition = 'opacity 0.5s ease';
      setTimeout(() => wrap.remove(), 500);
    }, 4000);
  }

  // ── 7. 確認對話框 ──────────────────────────────────────────
  /**
   * 顯示自定義確認對話框，取代 browser confirm
   * @param {string} message - 確認訊息
   * @param {string} confirmText - 確認按鈕文字
   * @param {string} type - 按鈕類型: danger | primary
   * @returns {Promise<boolean>}
   */
  function confirmAction(message, confirmText = '確認', type = 'danger') {
    return new Promise((resolve) => {
      const overlay = document.createElement('div');
      overlay.className = 'modal-overlay active';
      overlay.innerHTML = `
        <div class="modal">
          <div class="modal-header">
            <div class="modal-title">確認操作</div>
            <button class="btn-icon modal-close">&times;</button>
          </div>
          <div class="modal-body">
            <p style="color:var(--text-secondary);font-size:0.88rem;">${message}</p>
          </div>
          <div class="modal-footer">
            <button class="btn-ghost" data-action="cancel">取消</button>
            <button class="btn-${type}" data-action="confirm">${confirmText}</button>
          </div>
        </div>
      `;
      document.body.appendChild(overlay);

      function cleanup(result) {
        overlay.remove();
        resolve(result);
      }

      overlay.querySelector('[data-action="cancel"]').onclick = () => cleanup(false);
      overlay.querySelector('[data-action="confirm"]').onclick = () => cleanup(true);
      overlay.querySelector('.modal-close').onclick = () => cleanup(false);
      overlay.addEventListener('click', (e) => {
        if (e.target === overlay) cleanup(false);
      });
    });
  }

  // ── 8. 表格排序 ────────────────────────────────────────────
  function initSortableTable(table) {
    if (!table) return;
    const headers = table.querySelectorAll('th[data-sort]');

    headers.forEach((th, colIndex) => {
      th.style.cursor = 'pointer';
      th.style.userSelect = 'none';
      th.addEventListener('click', () => {
        const tbody = table.querySelector('tbody');
        if (!tbody) return;

        const rows = Array.from(tbody.querySelectorAll('tr'));
        const isAsc = th.classList.contains('asc');
        const type = th.dataset.sort; // 'text' | 'number' | 'date'

        // 清除所有排序狀態
        headers.forEach(h => h.classList.remove('asc', 'desc'));

        // 排序
        rows.sort((a, b) => {
          const aVal = a.cells[colIndex]?.textContent?.trim() || '';
          const bVal = b.cells[colIndex]?.textContent?.trim() || '';

          if (type === 'number') {
            return isAsc
              ? parseFloat(bVal) - parseFloat(aVal)
              : parseFloat(aVal) - parseFloat(bVal);
          }
          if (type === 'date') {
            return isAsc
              ? new Date(bVal) - new Date(aVal)
              : new Date(aVal) - new Date(bVal);
          }
          return isAsc
            ? bVal.localeCompare(aVal, 'zh-TW')
            : aVal.localeCompare(bVal, 'zh-TW');
        });

        th.classList.add(isAsc ? 'desc' : 'asc');
        rows.forEach(row => tbody.appendChild(row));
      });
    });
  }

  // ── 9. Tab 切換 ─────────────────────────────────────────────
  function initTabs(container) {
    if (!container) return;
    const items = container.querySelectorAll('.tab-item');
    const panes = container.querySelectorAll('.tab-pane');

    items.forEach(item => {
      item.addEventListener('click', () => {
        const target = item.dataset.tab;
        // 切換 active
        items.forEach(i => i.classList.remove('active'));
        panes.forEach(p => p.classList.remove('active'));
        item.classList.add('active');
        const targetPane = container.querySelector(`#${target}`);
        if (targetPane) targetPane.classList.add('active');
      });
    });
  }

  // ── 10. 數字動畫 ────────────────────────────────────────────
  /**
   * 數字計數動畫
   * @param {HTMLElement} element - 目標元素
   * @param {number} target - 目標數字
   * @param {number} duration - 動畫時長（毫秒）
   */
  function animateCount(element, target, duration = 1500) {
    if (!element) return;
    const start = 0;
    const startTime = performance.now();

    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // easeOutCubic 緩動函式
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = Math.floor(start + (target - start) * eased);
      element.textContent = formatNumber(current);

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        element.textContent = formatNumber(target);
      }
    }
    requestAnimationFrame(update);
  }

  // ── 11. Drag & Drop 上傳 ────────────────────────────────────
  function initDropzone(element, onFilesDropped) {
    if (!element) return;

    ['dragenter', 'dragover'].forEach(eventName => {
      element.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        element.classList.add('active');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      element.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        element.classList.remove('active');
      });
    });

    element.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length && onFilesDropped) {
        onFilesDropped(files);
      }
    });
  }

  // ── 12. 格式化工具 ──────────────────────────────────────────
  /**
   * 格式化位元組大小
   */
  function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  /**
   * 格式化日期時間
   */
  function formatDate(dateStr) {
    if (!dateStr) return '-';
    const d = new Date(dateStr);
    return d.toLocaleString('zh-TW', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
      hour12: false,
    });
  }

  /**
   * 格式化數字（加千分位）
   */
  function formatNumber(num) {
    if (num === null || num === undefined) return '0';
    return num.toLocaleString('zh-TW');
  }

  // ── 13. DOMContentLoaded 初始化 ─────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    // 時鐘
    startClock();

    // Django messages 自動消失
    initDjangoMessages();

    // Sidebar toggle 綁定
    const hamburger = document.getElementById('hamburgerBtn');
    if (hamburger) {
      hamburger.addEventListener('click', toggleSidebar);
    }

    // Sidebar overlay 點擊關閉
    const overlay = document.querySelector('.sidebar-overlay');
    if (overlay) {
      overlay.addEventListener('click', closeSidebar);
    }

    // 自動初始化所有 [data-tabs] 容器的 tab 切換
    document.querySelectorAll('[data-tabs]').forEach(initTabs);

    // 自動初始化所有 [data-sortable] 表格的排序
    document.querySelectorAll('table[data-sortable]').forEach(initSortableTable);

    // 自動初始化所有 [data-animate-count] 元素的數字動畫
    document.querySelectorAll('[data-animate-count]').forEach(el => {
      const target = parseInt(el.dataset.animateCount, 10);
      if (!isNaN(target)) {
        animateCount(el, target);
      }
    });
  });

  // ── 匯出全域函式 ────────────────────────────────────────────
  window.VNAAP = {
    api,
    showToast,
    toggleSidebar,
    closeSidebar,
    confirmAction,
    initSortableTable,
    initTabs,
    animateCount,
    initDropzone,
    formatBytes,
    formatDate,
    formatNumber,
    getCookie,
  };

  // 常用函式直接暴露到全域（方便模板中使用）
  window.showToast = showToast;
  window.toggleSidebar = toggleSidebar;
  window.confirmAction = confirmAction;

})();
