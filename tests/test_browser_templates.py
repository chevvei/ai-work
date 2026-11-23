"""浏览器练习在标准 Python 中的参考结果，浏览器 Worker 另做端到端验证。"""
import contextlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]

@unittest.skipUnless(shutil.which('node'), '需要 Node 读取 JavaScript 模板数据')
class BrowserTemplatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = ROOT / 'inference-handbook/assets/python-templates.js'
        command = "global.window={};require(process.argv[1]);process.stdout.write(JSON.stringify(window.PYTHON_EXAMPLES));"
        cls.templates = json.loads(subprocess.check_output(['node', '-e', command, str(script)], text=True))

    def test_five_templates_execute_and_match_reference(self):
        scopes = {}
        for template in self.templates:
            with self.subTest(template=template['id']):
                scope = {}
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(template['code'], '<教学模板>', 'exec'), scope)
                result = scope['result']
                self.assertIn(result['kind'], ['line', 'scatter'])
                self.assertLessEqual(sum(len(s['points']) for s in result['series']), 2000)
                for series in result['series']:
                    for point in series['points']:
                        self.assertEqual(len(point), 2)
                        self.assertTrue(all(math.isfinite(v) for v in point))
                scopes[template['id']] = scope
        self.assertEqual(len(scopes), 5)
        self.assertEqual(scopes['projection']['result']['series'][0]['points'][0], [680.0, 360.0])
        rollout = scopes['rollout']['result']['series']
        self.assertAlmostEqual(abs(rollout[0]['points'][-1][1] - rollout[1]['points'][-1][1]), .4)

if __name__ == '__main__':
    unittest.main()
