(function () {
  'use strict';
  const scriptURL = document.currentScript && document.currentScript.src;
  const workerURL = new URL('python-worker.js', scriptURL || new URL('assets/python-lab.js', document.baseURI)).href;
  const root = document.getElementById('python-playground');
  const examples = window.PYTHON_EXAMPLES;
  if (!root || !Array.isArray(examples) || !examples.length) return;
  const style = document.createElement('style');
  style.textContent = `
    #python-playground{min-width:0;padding:22px;background:var(--bg2);border:1px solid var(--rule);border-radius:12px}
    #python-playground>label{display:block;font-size:13px;color:var(--muted);margin:12px 0 8px}
    #python-playground select{max-width:100%;padding:9px 12px;background:var(--bg);color:var(--ink);border:1px solid var(--rule);border-radius:8px;font:inherit}
    #python-code{display:block;box-sizing:border-box;width:100%;min-height:320px;resize:vertical;padding:18px;border:1px solid var(--rule);border-radius:10px;background:var(--code);color:var(--codeink);font:13px/1.7 var(--mono);tab-size:4}
    #python-code:focus-visible,#python-playground select:focus-visible,#python-output:focus-visible{outline:3px solid var(--accent2);outline-offset:3px}
    #python-playground .python-controls{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}
    #python-playground .btn{min-height:40px}
    #python-run{background:var(--accent);border-color:var(--accent);color:var(--bg2)}
    #python-playground button:disabled{opacity:.5;cursor:not-allowed}
    #python-status{padding:10px 14px;background:var(--soft);border-left:3px solid var(--accent);font-size:13px;overflow-wrap:anywhere}
    #python-expected,#python-hint{font-size:13px;color:var(--muted);overflow-wrap:anywhere}
    #python-output{max-height:420px;white-space:pre-wrap;overflow-wrap:anywhere}
    #python-chart{overflow:hidden;border:1px solid var(--rule);border-radius:10px;background:var(--bg2)}
    @media(max-width:760px){#python-playground{padding:14px}#python-code{font-size:12px;padding:12px}#python-playground .python-controls .btn{flex:1 1 120px}}
  `;
  document.head.append(style);
  const CODE_LIMIT = 64 * 1024;
  const OUTPUT_LIMIT = 256 * 1024;
  const STORAGE_KEY = 'inference-python-draft-v1';
  const encoder = new TextEncoder();
  const fileMode = location.protocol === 'file:';
  let active = null;
  let nextId = 0;
  let chart = null;
  let selected = examples[0];

  function node(tag, text, className) {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (className) el.className = className;
    return el;
  }
  function button(text, handler) {
    const el = node('button', text, 'btn');
    el.type = 'button';
    el.addEventListener('click', handler);
    return el;
  }
  const label = node('label', '实验模板 ');
  const select = node('select');
  select.id = 'python-template';
  examples.forEach((example) => {
    const option = node('option', example.title);
    option.value = example.id;
    select.append(option);
  });
  label.append(select);
  const question = node('p');
  question.id = 'python-question';
  const expected = node('p');
  expected.id = 'python-expected';
  const hint = node('p');
  hint.id = 'python-hint';
  hint.hidden = true;
  const editorLabel = node('label', 'Python 代码（UTF-8 上限 64 KB；Tab 插入四个空格，Shift+Tab 减少缩进，Esc 后 Tab 可移出编辑器）');
  editorLabel.htmlFor = 'python-code';
  const editor = node('textarea');
  editor.id = 'python-code';
  editor.rows = 22;
  editor.spellcheck = false;
  editor.wrap = 'off';
  editor.setAttribute('autocapitalize', 'off');
  editor.setAttribute('autocomplete', 'off');
  editor.setAttribute('autocorrect', 'off');
  let releaseTab = false;
  editor.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') { releaseTab = true; return; }
    if (event.key !== 'Tab') { releaseTab = false; return; }
    if (releaseTab) { releaseTab = false; return; }
    event.preventDefault();
    const start = editor.selectionStart;
    const end = editor.selectionEnd;
    if (event.shiftKey) {
      const lineStart = editor.value.lastIndexOf('\n', start - 1) + 1;
      const spaces = Math.min(4, (/^ */.exec(editor.value.slice(lineStart)) || [''])[0].length);
      if (spaces) {
        editor.setRangeText('', lineStart, lineStart + spaces, 'preserve');
        editor.setSelectionRange(Math.max(lineStart, start - spaces), Math.max(lineStart, end - spaces));
      }
    } else {
      editor.setRangeText('    ', start, end, 'end');
    }
    editor.dispatchEvent(new Event('input', { bubbles: true }));
  });
  const status = node('p', '选择模板后点击运行。首次需要加载本地运行时。');
  status.id = 'python-status';
  status.setAttribute('role', 'status');
  status.setAttribute('aria-live', 'polite');
  const output = node('pre', '尚未运行。');
  output.id = 'python-output';
  output.setAttribute('aria-label', 'Python 标准输出与错误');
  output.tabIndex = 0;
  const chartBox = node('div', undefined, 'python-chart');
  chartBox.id = 'python-chart';
  chartBox.style.height = '360px';
  chartBox.style.width = '100%';
  chartBox.setAttribute('role', 'img');
  chartBox.setAttribute('aria-label', 'Python 运行结果图形；尚未运行');
  const run = button('运行 Python', execute);
  run.id = 'python-run';
  const stop = button('停止', () => finish('已停止；本次运行环境已销毁。'));
  stop.id = 'python-stop';
  const reset = button('重置模板', () => loadExample(selected));
  const challenge = button('加载故障挑战', () => {
    editor.value = selected.challengeCode;
    hint.hidden = true;
    clearResult();
    status.textContent = '已加载含故障代码。修复后运行，并与基准结果核对。';
  });
  const showHint = button('显示提示', () => { hint.hidden = !hint.hidden; showHint.textContent = hint.hidden ? '显示提示' : '收起提示'; });
  const saveDraft = button('保存草稿', () => {
    if (!codeFits()) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ templateId: selected.id, code: editor.value }));
      status.textContent = '草稿已保存到当前浏览器；下次请点击“恢复草稿”加载。';
    } catch (_) { status.textContent = '保存失败：浏览器存储不可用或空间不足。请下载代码保留修改。'; }
  });
  const restoreDraft = button('恢复草稿', () => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) { status.textContent = '当前浏览器没有已保存的草稿。'; return; }
      if (raw.length > CODE_LIMIT * 8) throw new Error('草稿格式无效或超出大小限制。');
      const draft = JSON.parse(raw);
      const example = draft && examples.find((item) => item.id === draft.templateId);
      if (!example || typeof draft.code !== 'string' || encoder.encode(draft.code).length > CODE_LIMIT) {
        throw new Error('草稿格式无效或代码超过 64 KB。');
      }
      loadExample(example);
      editor.value = draft.code;
      status.textContent = '已恢复草稿；点击运行才会执行。';
    } catch (_) { status.textContent = '恢复失败：浏览器存储不可用，或草稿格式无效、超过大小限制。'; }
  });
  const downloadCode = button('下载代码', () => {
    if (codeFits()) download(editor.value, selected.id + '.py');
  });
  const downloadOutput = button('下载输出', () => download(output.textContent, selected.id + '-output.txt'));
  const controls = node('div', undefined, 'python-controls');
  controls.append(run, stop, reset, challenge, showHint, saveDraft, restoreDraft, downloadCode, downloadOutput);
  const note = node('p', '每次运行独立执行，结束即销毁。启动最多 60 秒，ready 后执行最多 5 秒。输出界面最多保留 256 KB。基准仅供逐项核对，不自动判断答案。');
  root.append(label, question, expected, hint, editorLabel, editor, controls, status, note, node('h3', '文本输出'), output, node('h3', '结果图形'), chartBox);
  if (fileMode) {
    const fileNote = node('p', 'file:// 无法启动实验 Worker。请在本机终端执行以下命令，然后打开 http://localhost:8765/inference-handbook.html：');
    const command = node('pre', 'python3 -m http.server 8765 --directory /home/sti/Documents/Trae-CN/default/llm-basics/inference-handbook');
    root.prepend(fileNote, command);
  }
  select.addEventListener('change', () => loadExample(examples.find((item) => item.id === select.value)));
  window.addEventListener('resize', () => { if (chart) chart.resize(); });
  if (window.ResizeObserver) new ResizeObserver(() => { if (chart) chart.resize(); }).observe(chartBox);
  window.addEventListener('pagehide', () => { if (active) finish('页面已离开，运行已停止。'); });

  function busy(value) {
    run.disabled = value || fileMode || typeof Worker === 'undefined';
    stop.disabled = !value;
    select.disabled = value;
    reset.disabled = value;
    challenge.disabled = value;
    restoreDraft.disabled = value;
    editor.readOnly = value;
    root.setAttribute('aria-busy', String(value));
  }
  function clearResult() {
    output.textContent = '尚未运行。';
    if (chart) chart.clear();
    chartBox.setAttribute('aria-label', 'Python 运行结果图形；尚未运行');
  }
  function loadExample(example) {
    selected = example;
    select.value = example.id;
    editor.value = example.code;
    question.textContent = example.question;
    expected.textContent = '基准结果（人工自查）：' + example.expected;
    hint.textContent = '排查提示：' + example.hint;
    hint.hidden = true;
    showHint.textContent = '显示提示';
    clearResult();
    status.textContent = fileMode ? '当前为 file:// 页面，请按上方命令通过 HTTP 打开后运行。' : '已加载模板；点击运行才会执行。';
  }
  function codeFits() {
    if (encoder.encode(editor.value).length <= CODE_LIMIT) return true;
    status.textContent = '代码超过 UTF-8 64 KB 限制，请缩短后重试。';
    return false;
  }
  function limitText(text) {
    // 先按 UTF-16 长度取小窗口，避免把超大字符串完整编码。
    const bytes = encoder.encode(text.slice(0, OUTPUT_LIMIT));
    if (text.length <= OUTPUT_LIMIT && bytes.length <= OUTPUT_LIMIT) return text;
    const marker = '\n[输出超过 256 KB，已截断]';
    const room = OUTPUT_LIMIT - encoder.encode(marker).length;
    let end = Math.min(room, bytes.length);
    while (end > 0 && (bytes[end] & 0xc0) === 0x80) end--;
    return new TextDecoder().decode(bytes.subarray(0, end)) + marker;
  }
  function finish(message) {
    if (active) {
      const job = active;
      active = null;
      clearTimeout(job.startTimer);
      clearTimeout(job.executionTimer);
      job.worker.terminate();
    }
    busy(false);
    status.textContent = message;
  }
  function execute() {
    if (active || fileMode || !codeFits()) return;
    clearResult();
    output.textContent = '运行中…';
    busy(true);
    status.textContent = '正在启动独立 Python Worker（最多 60 秒）…';
    let worker;
    try { worker = new Worker(workerURL); }
    catch (error) {
      output.textContent = limitText(String(error));
      finish('无法创建 Worker，请检查浏览器支持、HTTP 地址及资源路径。');
      return;
    }
    const job = { id: ++nextId, worker: worker, ready: false, startTimer: null, executionTimer: null };
    active = job;
    job.startTimer = setTimeout(() => {
      if (active !== job) return;
      output.textContent = '启动超时：Python 运行时未能在 60 秒内就绪。';
      finish('启动超时，环境已销毁。可检查网络后重试。');
    }, 60000);
    worker.onmessage = (event) => {
      if (active !== job) return;
      const data = event.data;
      if (!data || data.id !== job.id) return;
      if (data.type === 'ready') {
        if (job.ready) return;
        job.ready = true;
        clearTimeout(job.startTimer);
        status.textContent = 'Python 已就绪，正在执行（最多 5 秒）…';
        job.executionTimer = setTimeout(() => {
          if (active !== job) return;
          output.textContent = '执行超时：代码运行超过 5 秒，已终止。';
          finish('执行超时，环境已销毁。请检查循环或缩小数据量。');
        }, 5000);
      } else if (data.type === 'result') {
        const stdout = typeof data.stdout === 'string' ? data.stdout : '';
        const stderr = typeof data.stderr === 'string' ? data.stderr : '';
        // 各段先限长再拼接，避免超大输出使界面内存持续增长。
        output.textContent = limitText('stdout:\n' + limitText(stdout) + '\n\nstderr:\n' + limitText(stderr));
        try {
          if (data.result !== null && data.result !== undefined) {
            const result = validateResult(data.result);
            draw(result);
            output.textContent = limitText(output.textContent + '\n\nresult:\n' + JSON.stringify(result));
            finish('运行结束；请对照基准核对输出。');
          } else {
            finish('运行结束；未返回 result 图形数据，请核对文本输出。');
          }
        } catch (error) {
          output.textContent = limitText(output.textContent + '\n\n图形错误：' + String(error.message || error));
          finish('运行结束，但图形数据无效或绘图库不可用。');
        }
      } else if (data.type === 'error') {
        output.textContent = limitText(typeof data.error === 'string' ? data.error : 'Worker 返回了未知错误。');
        finish('Python 运行失败，环境已销毁。');
      }
    };
    worker.onerror = (event) => {
      if (active !== job) return;
      event.preventDefault();
      output.textContent = limitText(event.message || 'Worker 加载失败。');
      finish('Worker 出错，环境已销毁。');
    };
    worker.onmessageerror = () => {
      if (active !== job) return;
      output.textContent = '无法读取 Worker 返回的数据。';
      finish('消息解析失败，环境已销毁。');
    };
    try { worker.postMessage({ id: job.id, type: 'run', code: editor.value }); }
    catch (error) {
      output.textContent = limitText(String(error));
      finish('发送代码失败，环境已销毁。');
    }
  }
  function validateResult(value) {
    const fail = () => { throw new Error('result 必须为 line/scatter，标题最多 120 字符，1–8 条序列、名称最多 80 字符，共最多 2000 个二维有限数值点。'); };
    const object = (item) => item && typeof item === 'object' && !Array.isArray(item);
    const keysAre = (item, keys) => Object.keys(item).length === keys.length && keys.every((key) => Object.prototype.hasOwnProperty.call(item, key));
    if (!object(value) || !keysAre(value, ['kind', 'title', 'series'])) fail();
    if (!['line', 'scatter'].includes(value.kind) || typeof value.title !== 'string' || value.title.length > 120) fail();
    if (!Array.isArray(value.series) || value.series.length < 1 || value.series.length > 8) fail();
    let total = 0;
    const series = value.series.map((item) => {
      if (!object(item) || !keysAre(item, ['name', 'points']) || typeof item.name !== 'string' || item.name.length > 80 || !Array.isArray(item.points)) fail();
      total += item.points.length;
      if (total > 2000) fail();
      const points = item.points.map((point) => {
        if (!Array.isArray(point) || point.length !== 2 || !point.every((n) => typeof n === 'number' && Number.isFinite(n))) fail();
        return [point[0], point[1]];
      });
      return { name: item.name, points: points };
    });
    return { kind: value.kind, title: value.title, series: series };
  }
  function draw(result) {
    if (!window.echarts) throw new Error('ECharts 未加载，文本结果仍可查看。');
    if (!chart) chart = window.echarts.init(chartBox, null, { renderer: 'svg' });
    const css = getComputedStyle(document.documentElement);
    const colors = ['--accent', '--accent2', '--muted', '--rule'].map((name) => css.getPropertyValue(name).trim());
    chart.setOption({
      animation: false,
      color: colors,
      title: { text: result.title, textStyle: { fontSize: 14, color: colors[2], overflow: 'truncate', width: Math.max(100, chartBox.clientWidth - 30) } },
      tooltip: { show: false, appendToBody: true },
      legend: { show: false },
      grid: { left: 65, right: 30, top: 55, bottom: 45 },
      xAxis: { type: 'value', scale: true, axisLabel: { color: colors[2] }, axisLine: { lineStyle: { color: colors[3] } }, splitLine: { lineStyle: { color: colors[3] } } },
      yAxis: { type: 'value', scale: true, axisLabel: { color: colors[2] }, axisLine: { lineStyle: { color: colors[3] } }, splitLine: { lineStyle: { color: colors[3] } } },
      series: result.series.map((item) => ({
        type: result.kind, name: item.name, data: item.points,
        symbolSize: result.kind === 'scatter' ? 9 : 5,
        showSymbol: true, label: { show: false }, emphasis: { disabled: true }
      }))
    }, true);
    chart.resize();
    chartBox.setAttribute('aria-label', result.title + '；' + result.series.map((item) => item.name + '：' + item.points.length + ' 点').join('；'));
  }
  function download(text, filename) {
    const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
    const link = node('a');
    link.href = url;
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  loadExample(selected);
  busy(false);
  if (!fileMode && typeof Worker === 'undefined') status.textContent = '当前浏览器不支持 Worker，无法运行 Python；仍可编辑和下载代码。';
})();