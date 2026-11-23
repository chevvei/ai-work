(function () {
  'use strict';
  const chapters = (window.HANDBOOK_CHAPTERS || []).concat(window.HANDBOOK_LAB ? [window.HANDBOOK_LAB] : [], window.HANDBOOK_CASES || [], window.HANDBOOK_PYTHON ? [window.HANDBOOK_PYTHON] : [], window.HANDBOOK_LOCAL || []);
  const $ = (id) => document.getElementById(id);
  let state = { read: {}, checks: {} };
  try { const saved = JSON.parse(localStorage.getItem('inference-book-v1')); if (saved && typeof saved === 'object') state = { read: saved.read || {}, checks: saved.checks || {} }; } catch (_) { /* 禁用存储时仍可阅读。 */ }
  const save = () => { try { localStorage.setItem('inference-book-v1', JSON.stringify(state)); } catch (_) {} };
  const toast = (text) => { $('toast').textContent = text; $('toast').style.display = 'block'; setTimeout(() => { $('toast').style.display = 'none'; }, 2500); };
  chapters.forEach((chapter, i) => {
    if (i === 0 || i === 3 || i === 7 || chapter.id === 'lab' || chapter.id === 'fusion-case') {
      const group = document.createElement('div'); group.className = 'nav-group'; group.textContent = chapter.id === 'fusion-case' ? '场景 · 从问题倒推原理' : chapter.id === 'lab' ? '动手 · 第一条部署链路' : i === 0 ? '起步 · 先恢复必要基础' : i === 3 ? '交付 · 正确地运行模型' : '进阶 · 定位、回答与取舍'; $('toc').append(group);
    }
    const link = document.createElement('a'); link.href = '#' + chapter.id; link.dataset.chapter = chapter.id;
    const name = document.createElement('span'); name.className = 'nav-title'; name.textContent = chapter.title;
    const dot = document.createElement('span'); dot.className = 'nav-dot'; dot.dataset.done = chapter.id;
    link.append(name, dot); $('toc').append(link);
    const section = document.createElement('section'); section.id = chapter.id; section.className = 'chapter';
    const header = document.createElement('div'); header.className = 'chapter-head';
    const titles = document.createElement('div'); const heading = document.createElement('h2'); heading.textContent = chapter.title;
    const sub = document.createElement('p'); sub.textContent = chapter.subtitle; titles.append(heading, sub);
    const badge = document.createElement('span'); badge.className = 'badge'; badge.textContent = chapter.priority;
    header.append(titles, badge);
    const body = document.createElement('div'); body.className = 'chapter-body'; body.innerHTML = chapter.body;
    // 章节内部保留层级：书章为 h2，主题为 h3。
    body.querySelectorAll('h2').forEach((old) => { const h = document.createElement('h3'); h.innerHTML = old.innerHTML; old.replaceWith(h); });
    const row = document.createElement('div'); row.className = 'read-row';
    const label = document.createElement('label'); const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.checked = Boolean(state.read[chapter.id]); checkbox.addEventListener('change', () => { state.read[chapter.id] = checkbox.checked; save(); updateProgress(); }); label.append(checkbox, document.createTextNode(' 我已理解，并能回答本章自查'));
    row.append(label);
    if (chapters[i + 1]) { const next = document.createElement('a'); next.href = '#' + chapters[i + 1].id; next.textContent = '下一章 →'; row.append(next); }
    section.append(header, body, row); $('chapters').append(section);
    section.querySelectorAll('.checks li').forEach((li, j) => { const key = chapter.id + '-' + j; const box = document.createElement('input'); box.type = 'checkbox'; box.setAttribute('aria-label', '自查：' + li.textContent); box.checked = Boolean(state.checks[key]); box.addEventListener('change', () => { state.checks[key] = box.checked; save(); }); const content = document.createElement('span'); while (li.firstChild) content.append(li.firstChild); li.append(box, content); });
  });
  function updateProgress() { const count = chapters.filter((c) => state.read[c.id]).length; $('progress-label').textContent = '已自查 ' + count + ' / ' + chapters.length + ' 章'; $('progress-bar').style.width = (count / Math.max(1, chapters.length) * 100) + '%'; document.querySelectorAll('[data-done]').forEach((el) => { el.textContent = state.read[el.dataset.done] ? '✓' : ''; }); }
  updateProgress();
  $('reset-progress').addEventListener('click', () => { if (!confirm('清空此浏览器中的章节与自查进度？')) return; state = { read: {}, checks: {} }; save(); location.reload(); });
  $('search').addEventListener('input', () => { const q = $('search').value.trim().toLocaleLowerCase(); let count = 0; chapters.forEach((c) => { const matched = !q || $(c.id).textContent.toLocaleLowerCase().includes(q); document.querySelector('[data-chapter="' + c.id + '"]').hidden = !matched; if (matched) count++; }); document.querySelectorAll('.nav-group').forEach((el) => { el.hidden = Boolean(q); }); $('empty').style.display = count ? 'none' : 'block'; });
  const observer = new IntersectionObserver((entries) => { const entry = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0]; if (entry) { document.querySelectorAll('[data-chapter]').forEach((a) => { if (a.dataset.chapter === entry.target.id) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); }); } }, { rootMargin: '-75px 0px -60% 0px', threshold: 0 });
  document.querySelectorAll('.chapter').forEach((el) => observer.observe(el));
  let expanded = false;
  $('expand-all').addEventListener('click', () => { expanded = !expanded; document.querySelectorAll('details').forEach((d) => { d.open = expanded; }); $('expand-all').textContent = expanded ? '收起全部图解' : '展开全部图解'; renderDiagrams(); });
  $('font-size').addEventListener('click', () => { const large = document.documentElement.style.getPropertyValue('--s') !== '18px'; document.documentElement.style.setProperty('--s', large ? '18px' : '16px'); $('font-size').textContent = large ? '字号 A−' : '字号 A+'; });
  $('menu').addEventListener('click', () => { const open = document.body.classList.toggle('menu-open'); $('menu').setAttribute('aria-expanded', String(open)); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { document.body.classList.remove('menu-open'); $('menu').setAttribute('aria-expanded', 'false'); } });
  document.querySelectorAll('a[href^="#"]').forEach((a) => a.addEventListener('click', () => { document.body.classList.remove('menu-open'); $('menu').setAttribute('aria-expanded', 'false'); }));
  document.querySelectorAll('pre:not(.mermaid)').forEach((pre) => { const wrap = document.createElement('div'); wrap.className = 'code-wrap'; pre.before(wrap); wrap.append(pre); const button = document.createElement('button'); button.className = 'copy'; button.textContent = '复制'; button.setAttribute('aria-label', '复制本段代码或命令'); button.addEventListener('click', async () => { const text = pre.textContent; try { if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text); else { const input = document.createElement('textarea'); input.value = text; document.body.append(input); input.select(); const ok = document.execCommand('copy'); input.remove(); if (!ok) throw new Error('copy'); } toast('已复制；执行前请核对环境与路径'); } catch (_) { toast('浏览器未允许复制，请手动选择代码'); } }); wrap.append(button); });
  if (window.mermaid) mermaid.initialize({ startOnLoad: false, theme: 'neutral', securityLevel: 'strict', fontFamily: 'sans-serif', flowchart: { useMaxWidth: true, htmlLabels: false }, suppressErrorRendering: true });
  let rendering = false;
  async function renderDiagrams() { if (!window.mermaid || rendering) return; const nodes = Array.from(document.querySelectorAll('pre.mermaid:not([data-processed])')).filter((node) => node.getClientRects().length); if (!nodes.length) return; rendering = true; try { await mermaid.run({ nodes }); } catch (_) { toast('部分图示未渲染，保留文字流程供阅读'); } finally { rendering = false; } }
  const dialog = document.createElement('dialog'); dialog.className = 'diagram-dialog'; dialog.setAttribute('aria-label', '放大查看图示'); const close = document.createElement('button'); close.className = 'btn'; close.textContent = '关闭图示'; const view = document.createElement('div'); view.className = 'diagram-view'; dialog.append(close, view); document.body.append(dialog); close.addEventListener('click', () => dialog.close());
  document.querySelectorAll('figure.diagram').forEach((figure) => { const actions = document.createElement('div'); actions.className = 'figure-actions'; const hint = document.createElement('span'); hint.textContent = '图示可横向滚动'; const zoom = document.createElement('button'); zoom.className = 'btn'; zoom.textContent = '放大查看'; zoom.addEventListener('click', async () => { await renderDiagrams(); const svg = figure.querySelector('svg'); if (!svg) { toast('请先展开图示'); return; } view.replaceChildren(svg.cloneNode(true)); dialog.showModal(); }); actions.append(hint, zoom); figure.prepend(actions); });
  renderDiagrams();
  document.querySelectorAll('details').forEach((d) => d.addEventListener('toggle', () => { if (d.open) renderDiagrams(); }));
  $('print').addEventListener('click', async () => { const closed = Array.from(document.querySelectorAll('details:not([open])')); closed.forEach((d) => { d.open = true; }); await renderDiagrams(); window.print(); closed.forEach((d) => { d.open = false; }); });
  const descriptions = [
    '输入契约：图片从哪里来？RGB 还是 BGR？尺寸、步长和所有权是什么？先固定同一批样本，才能比较不同后端。',
    '预处理：resize、颜色转换、归一化、布局排列。这里必须与训练时约定一致；指针合法不代表张量正确。',
    '数据传输：CPU 与 GPU 之间的缓冲、拷贝和同步。异步操作期间，输入内存不能提前释放或覆盖。',
    '模型执行：运行时按 shape、dtype 和设备执行计算图。TensorRT engine 是针对环境构建的执行产物，不是原始训练模型。',
    '后处理：分类取分数与类别；检测可能有框解码和 NMS。每个模型的输出契约不同，不能照搬索引。',
    '验证与测量：先确认输入、输出与任务指标，再分别测模型耗时及端到端耗时。性能结论必须注明硬件、形状、精度与统计口径。'
  ];
  let stage = 0, timer = null;
  function showStage(index) { stage = index; document.querySelectorAll('[data-stage]').forEach((button) => button.classList.toggle('active', Number(button.dataset.stage) === stage)); $('pipeline-description').textContent = descriptions[stage]; }
  document.querySelectorAll('[data-stage]').forEach((button) => button.addEventListener('click', () => showStage(Number(button.dataset.stage))));
  $('play-pipeline').addEventListener('click', () => { if (timer) { clearInterval(timer); timer = null; $('play-pipeline').textContent = '播放流程'; } else { showStage(0); timer = setInterval(() => { showStage((stage + 1) % descriptions.length); if (stage === descriptions.length - 1) { clearInterval(timer); timer = null; $('play-pipeline').textContent = '重新播放'; } }, matchMedia('(prefers-reduced-motion: reduce)').matches ? 2400 : 1400); $('play-pipeline').textContent = '暂停播放'; } });
  showStage(0);
  let nhwc = false;
  function renderLayout() { const values = nhwc ? [1,10,100,2,20,200,3,30,300,4,40,400] : [1,2,3,4,10,20,30,40,100,200,300,400]; $('layout-values').replaceChildren(...values.map((v) => { const span = document.createElement('span'); span.textContent = v; return span; })); $('layout-toggle').textContent = nhwc ? '当前 NHWC · 切换 NCHW' : '当前 NCHW · 切换 NHWC'; $('layout-explain').textContent = nhwc ? '连续 NHWC：一个像素的 R、G、B 相邻，再到下一个像素。注意 shape 与 stride 仍需分别核对。' : '连续 NCHW：先存完整 R 平面，再存 G 和 B 平面。这是连续布局算例，不代表所有 NCHW shape 都物理连续。'; }
  $('layout-toggle').addEventListener('click', () => { nhwc = !nhwc; renderLayout(); }); renderLayout();
  function quantize() { const x = Number($('quant-value').value); const scale = 1 / 127; const q = Math.max(-127, Math.min(127, Math.round(x / scale))); const restored = q * scale; $('quant-input').textContent = x.toFixed(2); $('quant-result').textContent = 'scale = 1/127；量化整数 ' + q + ' → 还原值 ' + restored.toFixed(4) + '；绝对误差 ' + Math.abs(restored - x).toFixed(4) + (Math.abs(x) > 1 ? '。超出校准范围发生截断，误差明显增大。' : '。刻度有限导致舍入误差；这不是模型准确率。'); } $('quant-value').addEventListener('input', quantize); quantize();
  let chart;
  if (window.echarts) { const s = getComputedStyle(document.documentElement); const colors = ['--accent','--accent2','--muted','--rule'].map((v) => s.getPropertyValue(v).trim()); chart = echarts.init($('budget-chart'), null, { renderer: 'svg' });
    function budget() { const speed = Number($('speed').value), total = 11 + 4 / speed; $('speed-label').textContent = speed + '×'; chart.setOption({ animation: false, color: colors, tooltip: { trigger: 'axis', appendToBody: true, valueFormatter: (v) => Number(v).toFixed(2) + ' ms' }, legend: { top: 0, textStyle: { color: colors[2] } }, grid: { left: 65, right: 15, top: 45, bottom: 25 }, xAxis: { type: 'value', max: 16, name: 'ms', axisLabel: { color: colors[2] }, splitLine: { lineStyle: { color: colors[3] } } }, yAxis: { type: 'category', data: ['原始','优化后'], axisLabel: { color: colors[2] } }, series: [{name:'预处理',data:[6,6]},{name:'传输',data:[2,2]},{name:'模型',data:[4,4/speed]},{name:'后处理',data:[3,3]}].map((v) => ({...v, type:'bar', stack:'total', barWidth:30})) }); $('budget-summary').textContent = '模型加速 ' + speed + ' 倍，串行总耗时从 15 ms 变为 ' + total.toFixed(2) + ' ms，端到端仅加速 ' + (15/total).toFixed(2) + ' 倍。先定位瓶颈，再决定优化对象。'; }
    $('speed').addEventListener('input', budget); budget(); window.addEventListener('resize', () => chart.resize()); }
  function followHash() { const id = decodeURIComponent(location.hash.slice(1)); const target = document.getElementById(id); if (target) { const parent = target.closest('details'); if (parent) parent.open = true; target.scrollIntoView(); } }
  window.addEventListener('hashchange', followHash); if (location.hash) requestAnimationFrame(followHash);
})();
