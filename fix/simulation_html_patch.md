# templates/analyzer/simulation.html 修改說明

以下皆為「找到舊內容 → 替換為新內容」的區塊，其餘部分不動。
目的：讓最終判定明確標示「只看規則式偵測」、CNN 分數明確標示「僅供
參考」、每筆封包新增可檢視的標頭資訊（不再黑箱）。

────────────────────────────────────────────────────────────
## 1. 封包數量滑桿下方的門檻提示文字

找到：

```html
<div class="text-muted" style="font-size:.66rem;margin-top:4px;" id="ruleThresholdHint">
  規則式偵測門檻參考：Port Scan >20 · ICMP Flood >50 · SYN Flood >100 · UDP Flood >200
</div>
```

不用修改這段 HTML 本身（文字已經正確），但下面 JS 更新它的地方要改，見第 4 點。

────────────────────────────────────────────────────────────
## 2. 結果卡片：新增「最終判定依據」說明橫幅

找到（在 `<div id="resultContent">` 內、`cnnReliabilityNote` 那個 div 之前）：

```html
      <!-- [新增] CNN 分數可信度提示（訓練/服務資料表示法不一致警示） -->
      <div id="cnnReliabilityNote" style="display:none;margin-bottom:12px;padding:10px 12px;
           background:rgba(234,179,8,.08);border:1px solid rgba(234,179,8,.25);border-radius:8px;">
        <div style="font-size:.72rem;color:var(--yellow);line-height:1.6;">
          ⚠️ <span id="cnnReliabilityText"></span>
        </div>
      </div>
```

替換為（新增一段「最終判定依據」，放在可信度提示之前，優先呈現）：

```html
      <!-- [新增] 最終判定依據說明：明確告知「只看規則式偵測」 -->
      <div style="margin-bottom:10px;padding:10px 12px;background:rgba(34,197,94,.08);
           border:1px solid rgba(34,197,94,.25);border-radius:8px;">
        <div style="font-size:.74rem;color:var(--green);line-height:1.6;">
          ✅ <span id="finalVerdictNote"></span>
        </div>
      </div>

      <!-- [新增] CNN 分數可信度提示（訓練/服務資料表示法不一致警示） -->
      <div id="cnnReliabilityNote" style="display:none;margin-bottom:12px;padding:10px 12px;
           background:rgba(234,179,8,.08);border:1px solid rgba(234,179,8,.25);border-radius:8px;">
        <div style="font-size:.72rem;color:var(--yellow);line-height:1.6;">
          ⚠️ <span id="cnnReliabilityText"></span>
        </div>
      </div>
```

────────────────────────────────────────────────────────────
## 3. 結果表格：新增「封包標頭」欄位

找到：

```html
      <div class="table-wrap" style="max-height:250px;overflow-y:auto;">
        <table>
          <thead><tr><th>#</th><th>大小</th><th>內容摘要</th><th>異常分數</th><th>判定</th><th>觸發依據</th></tr></thead>
          <tbody id="resultTbody"></tbody>
        </table>
      </div>
```

替換為：

```html
      <div class="table-wrap" style="max-height:250px;overflow-y:auto;">
        <table>
          <thead><tr>
            <th>#</th><th>大小</th><th>封包標頭</th><th>內容摘要</th>
            <th>規則判定<br><span style="font-weight:400;font-size:.62rem;color:var(--text-muted);">（最終結論）</span></th>
            <th>CNN 分數<br><span style="font-weight:400;font-size:.62rem;color:var(--text-muted);">（僅供參考）</span></th>
            <th>觸發依據</th>
          </tr></thead>
          <tbody id="resultTbody"></tbody>
        </table>
      </div>
```

────────────────────────────────────────────────────────────
## 4. JS：runSimulation() 內，設定各項文字與統計數字的地方

找到：

```js
    // [新增] 基準線比對面板：顯示這次動態閾值是用哪批資料、怎麼算出來的
    const baselineEl = document.getElementById('baselineInfo');
```

在這一行「之前」插入：

```js
    // [新增] 最終判定依據說明（只看規則式偵測，CNN 僅供參考）
    document.getElementById('finalVerdictNote').textContent =
      res.final_verdict_note || '最終「攻擊／正常」判定僅由規則式流量型樣偵測決定，CNN／VAE 分數僅供參考。';
```

────────────────────────────────────────────────────────────
## 5. JS：規則式偵測面板文字與門檻提示（移除保底判定分支）

找到：

```js
    const bd = res.behavior_detection || {};
    if (bd.thresholds) {
      const t = bd.thresholds;
      let shortcutNote = '';
      if (bd.forced_by_demo_shortcut) {
        shortcutNote = '<br><span style="color:var(--yellow)">⚠ 此結果為示範保底判定，規則引擎於本批次未實際達標觸發</span>';
      }
      ruleContent.insertAdjacentHTML('beforeend', `
        <div style="margin-top:6px;font-size:.7rem;color:var(--text-muted);">
          本次套用門檻（示範用，非正式環境門檻）：
          SYN=${t.syn} · Port Scan=${t.ports} · ICMP=${t.icmp} · UDP=${t.udp}
          ${shortcutNote}
        </div>`);
      // 同步更新左側滑桿下方的靜態提示文字
      document.getElementById('ruleThresholdHint').innerHTML =
        `本次實際門檻（封包數×50%）：SYN >${t.syn} · Port Scan >${t.ports} · ICMP >${t.icmp} · UDP >${t.udp}`;
    }
```

替換為（門檻現在就是正式系統門檻，不再有保底判定，文字據實更新）：

```js
    const bd = res.behavior_detection || {};
    if (bd.thresholds) {
      const t = bd.thresholds;
      ruleContent.insertAdjacentHTML('beforeend', `
        <div style="margin-top:6px;font-size:.7rem;color:var(--text-muted);">
          本次套用門檻（與正式系統相同）：
          SYN=${t.syn} · Port Scan=${t.ports} · ICMP=${t.icmp} · UDP=${t.udp}
        </div>`);
      // 同步更新左側滑桿下方的靜態提示文字
      document.getElementById('ruleThresholdHint').innerHTML =
        `本次實際門檻（與正式系統相同）：SYN >${t.syn} · Port Scan >${t.ports} · ICMP >${t.icmp} · UDP >${t.udp}`;
    }
```

────────────────────────────────────────────────────────────
## 6. JS：表格逐行渲染（顯示封包標頭、拆開「規則判定」與「CNN 分數」兩欄）

找到：

```js
    res.results.forEach((r, i) => {
      const tr = document.createElement('tr');
      tr.className = 'pkt-row';
      tr.style.animationDelay = `${i * 0.04}s`;
      // [新增] 觸發依據欄位
      const trigger = r.rule_triggered_here
        ? `★ ${(r.rule_detail && r.rule_detail[0]) ? r.rule_detail[0].attack_type || '' : ''}`
        : r.behavior_anomaly ? '同批次判定' : (r.cnn_anomaly ? 'CNN 重建誤差' : '—');
      tr.innerHTML = `
        <td class="mono">${r.index + 1}</td>
        <td>${VNAAP.formatBytes(r.size)}</td>
        <td class="mono text-muted" style="font-size:.66rem;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${r.summary || ''}">${r.summary || '-'}</td>
        <td class="mono ${r.is_anomaly ? 'text-red' : 'text-green'}">${r.score.toFixed(6)}</td>
        <td>${
          r.cnn_anomaly && r.behavior_anomaly
          ? '<span class="verdict-badge verdict-anomaly">⚠ CNN + 行為</span>'
          : r.behavior_anomaly
          ? '<span class="verdict-badge verdict-anomaly">⚠ 行為異常</span>'
          : r.cnn_anomaly
          ? '<span class="verdict-badge verdict-anomaly">⚠ CNN 異常</span>'
          : '<span class="verdict-badge verdict-normal">✓ 正常</span>'
        }</td>
        <td style="font-size:.66rem;color:var(--text-muted);">${trigger}</td>`;
      tbody.appendChild(tr);
    });
```

替換為（`is_anomaly` 現在只等於 `behavior_anomaly`，CNN 獨立顯示為參考分數，不再混在判定徽章裡；新增封包標頭欄）：

```js
    res.results.forEach((r, i) => {
      const tr = document.createElement('tr');
      tr.className = 'pkt-row';
      tr.style.animationDelay = `${i * 0.04}s`;

      // 封包標頭（不再黑箱：直接呈現解析後的來源/目的/Port/flags）
      const h = r.header || {};
      const headerParts = [];
      if (h.protocol) headerParts.push(h.protocol);
      if (h.src_ip) headerParts.push(`${h.src_ip}${h.src_port != null ? ':' + h.src_port : ''}`);
      if (h.dst_ip) headerParts.push(`→ ${h.dst_ip}${h.dst_port != null ? ':' + h.dst_port : ''}`);
      if (h.flags) headerParts.push(`[${h.flags}]`);
      if (h.icmp_desc) headerParts.push(h.icmp_desc);
      if (h.arp_op) headerParts.push(`ARP ${h.arp_op}`);
      const headerText = headerParts.join(' ') || '-';

      // 觸發依據欄位
      const trigger = r.rule_triggered_here
        ? `★ ${(r.rule_detail && r.rule_detail[0]) ? r.rule_detail[0].attack_type || '' : ''}`
        : r.behavior_anomaly ? '同批次規則判定' : (r.cnn_anomaly ? 'CNN 分數偏高（未計入判定）' : '—');

      tr.innerHTML = `
        <td class="mono">${r.index + 1}</td>
        <td>${VNAAP.formatBytes(r.size)}</td>
        <td class="mono text-muted" style="font-size:.64rem;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${headerText}">${headerText}</td>
        <td class="mono text-muted" style="font-size:.64rem;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${r.summary || ''}">${r.summary || '-'}</td>
        <td>${
          r.is_anomaly
          ? '<span class="verdict-badge verdict-anomaly">⚠ 異常</span>'
          : '<span class="verdict-badge verdict-normal">✓ 正常</span>'
        }</td>
        <td class="mono ${r.cnn_anomaly ? 'text-yellow' : 'text-muted'}" style="font-size:.72rem;" title="僅供參考，不影響上一欄的最終判定">${r.score.toFixed(6)}</td>
        <td style="font-size:.66rem;color:var(--text-muted);">${trigger}</td>`;
      tbody.appendChild(tr);
    });
```

（若 CSS 中沒有 `text-yellow` class，可直接用 `style="color:var(--yellow)"` 取代該 class，效果相同。）
