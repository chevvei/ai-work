(function () {
  'use strict';
  const projection = `# 针孔投影：相机坐标 X 向右、Y 向下、Z 向前，长度单位米
points = [(1, 0, 20), (1, 1, 40), (-1, -1, 20), (0, 0, -1)]
fx, fy, cx, cy = 800, 800, 640, 360
pixels = []
for x, y, z in points:
    if z <= 0:
        continue
    u = fx * x / z + cx
    v = fy * y / z + cy
    pixels.append([u, v])
print("有效点数:", len(pixels))
print("像素坐标:", pixels)
result = {"kind": "scatter", "title": "针孔投影（像素）", "series": [{"name": "投影点", "points": pixels}]}
`;
  const voxel = `# floor 对负数向下取整；每格用质心代表，边长 1 米
import math
points = [(-0.1, 0.1, 0.1), (0.1, 0.1, 0.1), (0.9, 0.9, 0.9), (1.2, 0.2, 0.2)]
size = 1.0
cells = {}
for p in points:
    key = tuple(math.floor(v / size) for v in p)
    cells.setdefault(key, []).append(p)
centers = []
for key, group in sorted(cells.items()):
    center = [sum(p[i] for p in group) / len(group) for i in range(3)]
    centers.append(center)
    print("体素", key, "点数", len(group), "质心", [round(v, 3) for v in center])
print("输入点数:", len(points), "输出点数:", len(centers))
result = {"kind": "scatter", "title": "体素质心 XY（米）", "series": [{"name": "原始点", "points": [[p[0], p[1]] for p in points]}, {"name": "质心", "points": [[p[0], p[1]] for p in centers]}]}
`;
  const action = `# 动作解码：训练统计 p1/p99 对应归一化范围 [-1,1]
normalized = [-1.0, 0.5, 1.3]
p1, p99 = -0.1, 0.1
clipped = [max(-1.0, min(1.0, value)) for value in normalized]
decoded = [p1 + (value + 1) / 2 * (p99 - p1) for value in clipped]
print("归一化输入:", normalized)
print("裁剪后:", clipped)
print("解码动作（米）:", [round(value, 6) for value in decoded])
result = {"kind": "scatter", "title": "动作解码：归一化输入 / 位移米", "series": [{"name": "解码动作", "points": [[x, y] for x, y in zip(normalized, decoded)]}]}
`;
  const rollout = `# 世界模型滚动预测：恒加速度真值 vs 忽略加速度的预测器
x, velocity, acceleration = 0.0, 1.0, 0.2
dt, steps = 0.1, 20
predicted = 0.0
truth_points, predicted_points = [[0.0, x]], [[0.0, predicted]]
for step in range(1, steps + 1):
    x += velocity * dt + 0.5 * acceleration * dt ** 2
    velocity += acceleration * dt
    predicted += 1.0 * dt
    time = round(step * dt, 6)
    truth_points.append([time, x])
    predicted_points.append([time, predicted])
    print("时间", time, "真值", round(x, 6), "预测", round(predicted, 6), "误差", round(abs(x - predicted), 6))
print("终点真值:", round(x, 6), "米")
print("终点预测:", round(predicted, 6), "米")
print("终点绝对误差:", round(abs(x - predicted), 6), "米")
result = {"kind": "line", "title": "世界模型 rollout：时间秒 / 位置米", "series": [{"name": "恒加速度真值", "points": truth_points}, {"name": "忽略加速度预测", "points": predicted_points}]}
`;
  const quantization = `# 对称 INT8：明确采用 Python round（半数取偶），范围 [-127,127]
values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
scale = 1 / 127
integers = [max(-127, min(127, round(x / scale))) for x in values]
restored = [q * scale for q in integers]
errors = [abs(x - y) for x, y in zip(values, restored)]
print("量化整数:", integers)
print("还原值:", [round(x, 6) for x in restored])
print("最大绝对误差:", round(max(errors), 6))
print("范围内最大误差:", round(max(e for x, e in zip(values, errors) if abs(x) <= 1), 6))
result = {"kind": "line", "title": "量化：输入 / 还原值", "series": [{"name": "理想值", "points": [[x, x] for x in values]}, {"name": "INT8 还原", "points": [[x, y] for x, y in zip(values, restored)]}]}
`;
  window.HANDBOOK_PYTHON = {
    id: 'python-lab', title: '19 · 直接运行 Python：代码实验台', priority: 'P0',
    subtitle: '修改真实 Python，观察数值输出与图形，用明确基准自查。',
    body: '<p>选择投影、体素、动作、rollout 或量化实验，修改代码后点击运行。脚本仅使用 Python 标准库，每次运行都是全新环境，首次需要加载本地运行时。不会自动执行代码。</p><p>图形由脚本的 result 数据生成，仅接受 line / scatter、最多 8 条序列和共 2000 个二维有限数值点。下方基准用于人工对照，不是 AI 评分。挑战会装入含故障的代码，提示说明排查方向。</p><p>草稿仅在点击保存时写入当前浏览器；恢复也需点击按钮。请勿在实验中填写密钥或私人数据。</p><div id="python-playground"></div>'
  };
  window.PYTHON_EXAMPLES = [
    {id: 'projection', title: '投影 · 深度分母', question: '为什么同样的 X 在更远的 Z 处更靠近主点？修复挑战中的深度分母。', code: projection, expected: '有效点数 3；像素 [[680.0, 360.0], [660.0, 380.0], [600.0, 320.0]]。Z≤0 的点被排除。', hint: 'u = fx × X / Z + cx，v = fy × Y / Z + cy；挑战把 Z 错用成 X，须修复两个分母。', challengeCode: projection.replace('fx * x / z', 'fx * x / x').replace('fy * y / z', 'fy * y / x')},
    {id: 'voxel', title: '体素 · 负坐标边界', question: '边长 1 米时，-0.1 与 0.1 是否属于同一个体素？', code: voxel, expected: '4 点变为 3 个体素；键 (-1,0,0)、(0,0,0)、(1,0,0)，质心分别 [-0.1,0.1,0.1]、[0.5,0.5,0.5]、[1.2,0.2,0.2]。', hint: 'int(-0.1) 是 0，而 floor(-0.1) 是 -1。体素键需向下取整。', challengeCode: voxel.replace('math.floor(v / size)', 'int(v / size)')},
    {id: 'action', title: '动作 · 裁剪与解码', question: 'p1/p99 为 [-0.1,0.1] 米时，归一化动作 [-1,0.5,1.3] 应怎样裁剪、还原？', code: action, expected: '裁剪后 [-1.0,0.5,1.0]；解码动作 [-0.1,0.05,0.1] 米。1.3 不能直接产生 0.13 米动作。', hint: '先把归一化值裁剪至 [-1,1]，再用 p1 + (value + 1) / 2 × (p99 - p1) 还原。', challengeCode: action.replace('max(-1.0, min(1.0, value))', 'value')},
    {id: 'rollout', title: 'Rollout · 滚动预测误差', question: '初始位置 0、速度 1 m/s、加速度 0.2 m/s²，以 0.1 秒步长滚动 20 步，忽略加速度会偏差多少？', code: rollout, expected: '2 秒时真值位置 2.4 米，忽略加速度的预测为 2.0 米，终点绝对误差 0.4 米。两条曲线各 21 点。', hint: '真值位置用更新前速度计算 x += v×dt + 0.5×a×dt²，随后更新 v += a×dt。挑战漏掉速度更新，会使真值曲线也算错。', challengeCode: rollout.replace('velocity += acceleration * dt', 'velocity += 0.0')},
    {id: 'quantization', title: '量化 · 饱和与误差', question: '超出校准范围 [-1,1] 后误差为什么显著增大？', code: quantization, expected: '整数 [-127,-127,-64,0,64,127,127]；还原 [-1,-1,-0.503937,0,0.503937,1,1]；最大误差 1.0，范围内最大误差 0.003937。', hint: 'INT8 必须饱和到 [-127,127]；去掉截断会产生范围外整数，误差看似更小但不再是约定的量化。', challengeCode: quantization.replace('max(-127, min(127, round(x / scale)))', 'round(x / scale)')}
  ];
})();
