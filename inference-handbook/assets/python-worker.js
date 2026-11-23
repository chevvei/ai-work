'use strict';
// 每次运行使用独立 Worker；停止或超时由界面 terminate，避免共享状态。
self.onmessage = async function (event) {
  const { id, type, code } = event.data || {};
  if (type !== 'run') return;
  if (typeof code !== 'string' || new TextEncoder().encode(code).length > 65536) {
    self.postMessage({ id, type: 'error', error: '代码不能为空或超过 64 KiB。' }); return;
  }
  try {
    importScripts('../_shared/pyodide/pyodide.js');
    const py = await self.loadPyodide({ indexURL: new URL('../_shared/pyodide/', self.location.href).href });
    self.postMessage({ id, type: 'ready' });
    py.globals.set('_lab_source', code);
    // 输出写入有界内存；Python result 必须是可 JSON 序列化的基础结构。
    const encoded = await py.runPythonAsync(`
import sys, io, json, traceback
class _BoundedOutput(io.TextIOBase):
    def __init__(self):
        self.parts = []
        self.size = 0
        self.truncated = False
    def write(self, text):
        text = str(text)
        if self.size < 131072:
            part = text[:131072-self.size]
            self.parts.append(part)
            self.size += len(part)
        if self.size >= 131072:
            self.truncated = True
        return len(text)
    def flush(self):
        pass
    def value(self):
        return ''.join(self.parts) + ('\\n[输出达到上限，已截断]' if self.truncated else '')
_out, _err = _BoundedOutput(), _BoundedOutput()
_old_out, _old_err = sys.stdout, sys.stderr
_scope = {'__name__': '__main__'}
_payload = None
try:
    sys.stdout, sys.stderr = _out, _err
    exec(compile(_lab_source, '<网页实验>', 'exec'), _scope)
    _plot = _scope.get('result')
    if _plot is not None:
        _plot_text = json.dumps(_plot, allow_nan=False)
        if len(_plot_text) > 131072:
            raise ValueError('图表结果超过 128 KiB，请减少点数')
    _payload = {'stdout': _out.value(), 'stderr': _err.value(), 'result': _plot}
except BaseException:
    traceback.print_exc(file=_err)
    _payload = {'stdout': _out.value(), 'stderr': _err.value(), 'result': None}
finally:
    sys.stdout, sys.stderr = _old_out, _old_err
json.dumps(_payload, allow_nan=False)
`);
    const payload = JSON.parse(encoded);
    self.postMessage({ id, type: 'result', ...payload });
  } catch (error) {
    self.postMessage({ id, type: 'error', error: String(error && error.message || error) });
  }
};
