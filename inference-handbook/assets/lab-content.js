window.HANDBOOK_LAB = {
  id: 'lab',
  title: '实战 · 跑通第一条部署链路',
  subtitle: '在 3090 Ti 24GB 上建立可复现的冒烟证据：固定输入、数值比较与计时边界',
  priority: 'P0',
  body: `
<p class="callout"><strong>本关交付：</strong>ResNet18 DEFAULT 公开权重 → 固定随机 FP32 输入 → PyTorch 基线 → ONNX → ORT CPU 数值对照 → 明确口径的 benchmark。随后尝试 TensorRT engine 构建与测量。本脚本只做 synthetic 链路冒烟，<strong>不计算真实任务 accuracy</strong>。24GB 显存不是兼容性证明，也不保证固定加速倍数。</p>
<p><a href="assets/lab.py" download="lab.py">下载可运行的 lab.py</a>（通过教程页面的相对链接保存）。脚本与本章节文件在 assets 同目录；链接以教程入口位于 inference-handbook 根目录为前提。也可使用下面的终端 cp 保存副本，不需要 file:// 链接。输出默认保存在 /tmp/inference-handbook-lab，可被系统清理；每次运行会覆盖同名产物，保留对比实验时请换绝对路径目录。页面中的命令块可由教程外部 UI 添加复制按钮。</p>
<h2>0 · 准备独立解释器与脚本</h2>
<p>以下命令使用 Bash。先核实 Python 3.11 是否存在；不存在时停止这组命令，根据操作系统文档准备解释器，或确认当前 Python 版本受所选 torch/torchvision 支持。不要假设 python3 就是 3.11。运行命令不会替你安装系统解释器。</p>
<pre class="codebox"><code>command -v python3.11
python3.11 --version
# 上两行成功后，再执行本组：
python3.11 -m venv "$HOME/.venvs/inference-handbook"
source "$HOME/.venvs/inference-handbook/bin/activate"
python --version
python -m pip --version
LAB="/home/sti/Documents/Trae-CN/default/llm-basics/inference-handbook/assets/lab.py"
cp "$LAB" ./lab.py
LAB="$PWD/lab.py"
WORK="/tmp/inference-handbook-lab"
python "$LAB" --help
python "$LAB" inspect</code></pre>
<p>每次打开新终端都要重新激活 venv、设置 LAB 和 WORK。也可以直接把 LAB 设为仓库中的脚本绝对路径。请保留下载副本与原脚本的对应关系。</p>
<p><strong>依赖需要联网下载：</strong>打开 <a href="https://pytorch.org/get-started/locally/" target="_blank" rel="noopener noreferrer">PyTorch 官方安装选择器</a>，选 Linux、Pip、Python，再依据驱动兼容性选择该页面当前支持的 CUDA 计算平台。将选择器生成的命令中的 pip/pip3 换成当前解释器的 python -m pip 后原样执行其余参数；不要把驱动页面显示的 CUDA 数字拼进 wheel URL。torch 与 torchvision 必须是官方支持的版本组合。若先走 CPU，选择 CPU 安装命令。</p>
<pre class="codebox"><code># 先执行官方选择器生成的 torch/torchvision 安装命令。
# 用 python -m pip 替换其 pip/pip3，保留官方包参数与索引地址。
# 下列命令也会联网下载依赖；确认后手动执行：
python -m pip install numpy onnx onnxruntime
python -m pip check
python "$LAB" inspect</code></pre>
<p><strong>成功信号：</strong>解释器与 pip 均指向独立 venv，inspect 显示期望版本。<strong>失败排查：</strong>venv 模块缺失按系统文档处理；torchvision 算子加载错误先检查版本组合；不要用 sudo pip 修补当前环境。<strong>面试映射：</strong>解释“可复现环境”为什么需要明确解释器、包版本和驱动，而非只写 pip install。</p>
<h2>1 · inspect：先看机器，再决定后端</h2>
<pre class="codebox"><code>python "$LAB" inspect</code></pre>
<p>此命令只使用标准库读取 Python/已安装包版本，并用 nvidia-smi 查询 GPU 名称、驱动版本与总显存，不导入 torch、不输出环境变量、不创建产物。3090 Ti 通常应报告约 24GB 显存；以本机实际结果为准。nvidia-smi 不存在时仍能继续 CPU 路线；驱动可见不等于当前 torch 的 CUDA 可用。</p>
<p><strong>成功信号：</strong>输出 JSON，能辨认当前解释器、缺失包、GPU 与驱动。<strong>失败排查：</strong>GPU 查询失败先检查驱动及设备可见性，包显示未安装先确认是否激活正确 venv。<strong>面试映射：</strong>区分驱动、CUDA Toolkit、框架自带运行库与 TensorRT 的作用。</p>
<h2>2 · baseline：固定输入，保存可比较的参考</h2>
<pre class="codebox"><code># 第一次会下载 torchvision ResNet18 DEFAULT 公开权重。
# 先确认网络、存储与权重使用条款；只有 --allow-download 才允许缺缓存时下载。
python "$LAB" baseline --work-dir "$WORK" --device auto --batch 1 --allow-download
# 权重缓存存在后的离线重复运行：
python "$LAB" baseline --work-dir "$WORK" --device auto --batch 1</code></pre>
<p>保存 input.npy、baseline.npy、baseline.json。随机种子固定为 2026，在 CPU 上生成 1×3×224×224 的标准正态 FP32 张量，再交给目标设备；模型明确 eval 与 inference_mode。auto 在 torch CUDA 不可用时回退 CPU；也可显式 --device cpu。显式 cuda 不可用会报错。首次下载是模型权重下载，不是验证集下载；本练习不下载图片。公开权重来源为 torchvision 的 ResNet18_Weights.DEFAULT，JSON 记录其枚举名称及库版本。</p>
<p><strong>成功信号：</strong>输出形状为 [1,1000]，JSON 记录输入哈希、shape、设备与版本。随机输入的分类结果没有业务含义。<strong>失败排查：</strong>缺缓存会明确提示加 --allow-download；网络失败检查访问与缓存，不要换随机初始化权重冒充同一实验。CUDA 不可用先用 CPU 跑通，再核对官方安装组合。<strong>面试映射：</strong>解释 eval 与 inference_mode、固定种子与跨硬件位级一致性的区别。</p>
<h2>3 · export：新增模型检查，不改原应用</h2>
<pre class="codebox"><code>python "$LAB" export --work-dir "$WORK" --device cpu --batch 1
python "$LAB" export --help</code></pre>
<p>脚本重新加载同一 DEFAULT 权重，在 CPU 导出 resnet18.onnx，并调用 onnx.checker.check_model；成功写出 export.json。采用 torch.onnx.export(dynamo=False, opset_version=17)，明确使用传统导出路径以适配常见新版 torch。输入名 input、输出名 logits，仅 batch 动态，通道和空间尺寸固定 3×224×224。导出成功与 checker 通过说明图结构可检查，仍需下一步执行比较；它们不保证所有后端支持，更不代表任务质量通过。</p>
<p><strong>成功信号：</strong>ONNX 文件存在，export.json 的 onnx_checker 为 passed。<strong>失败排查：</strong>检查 onnx 包、torch/torchvision 版本与报错算子；遇版本差异对照对应版本文档，别无解释地更改 opset。输入名和动态轴以该图为准。<strong>面试映射：</strong>解释动态 batch、固定 H/W 与 TensorRT profile 的关系。</p>
<h2>4 · compare：同输入、同参考，先验证数值</h2>
<pre class="codebox"><code>python "$LAB" compare --work-dir "$WORK" --atol 0.0001 --rtol 0.001</code></pre>
<p>读取保存的输入、参考输出和输入哈希；明确使用 ORT 的 CPUExecutionProvider，保存 ort-output.npy 与 compare.json。报告 numpy allclose、max_abs、mean_abs、top1_agreement；allclose 按 |candidate-reference| ≤ atol + rtol×|reference| 比较。NaN/Inf 或 shape 不一致直接失败。top1 agreement 是两个后端预测类别相同的比例，<strong>不是与真实标签比较的 accuracy</strong>。默认容差只是本 FP32 冒烟的起点，不能照搬到所有模型与精度路径；不通过时进程返回非零。</p>
<p><strong>成功信号：</strong>allclose 为 true，误差值有限，并保存实际报告。<strong>失败排查：</strong>先确认 baseline 与 ONNX 使用同一模型、输入、shape、版本；输入哈希不匹配时重跑 baseline，再导出比较。不要先放宽容差。只跑 batch=1 不能证明全部动态范围有效，可用另一绝对 WORK 目录以 batch=8 重复 baseline→export→compare。<strong>面试映射：</strong>说明数值误差、类别一致性和任务指标是三个不同验收问题。</p>
<details><summary><span class="eye" aria-hidden="true"></span>真实图片和标签是下一项独立验收</summary>
<p>本简化脚本不提供 --image-dir，避免把任意 ImageFolder 目录标签误认为 ImageNet 的 1000 类映射。若后续加入真实验证集，应使用 DEFAULT.weights 对应的预处理（在 torchvision 中调用该权重枚举的 transforms()），固定样本清单，核对 ImageFolder.class_to_idx 与模型输出类别的显式映射、数据许可及训练/验证划分。目录字母序不保证与模型标签一致。然后分别测参考与部署后端的真实任务指标和指标变化；本练习没有提供这些指标或数值。输入随机张量也不模拟真实图像分布。</p>
</details>
<h2>5 · benchmark：预热、同步，记录统计口径</h2>
<pre class="codebox"><code>python "$LAB" benchmark --work-dir "$WORK" --backend torch --device auto --batch 1 --warmup 20 --runs 100
# 保存 GPU/torch 产物后再执行，避免同名报告被覆盖：
cp "$WORK/benchmark.json" "$WORK/benchmark-torch.json"
cp "$WORK/benchmark.csv" "$WORK/benchmark-torch.csv"
# ORT 这条命令明确测 CPU；不是 ORT GPU 性能：
python "$LAB" benchmark --work-dir "$WORK" --backend ort --device cpu --batch 1 --warmup 20 --runs 100</code></pre>
<p>输出 benchmark.csv 的逐次毫秒样本及 benchmark.json 的 median_ms、p95_ms、samples、warmup、batch、设备与计时范围。p95 使用 (n−1)×0.95 位置的线性插值。torch CUDA 路径先预热，每次计时前后 cuda.synchronize，再用 perf_counter 计墙钟时间，避免只测到异步提交；输入已驻留设备，计入 forward、主机调用、输出分配及同步开销，不含模型加载、H2D/D2H、图像预后处理。CPU ORT 的 session.run 计入主机 API 与输出分配。ORT benchmark 仍复用 torch 生成固定输入，因而也需要 torch/torchvision 和缓存权重。</p>
<p><strong>成功信号：</strong>CSV 有 100 个样本，JSON 写明当前设备和计时边界。<strong>失败排查：</strong>大抖动先查看 GPU 是否有其他任务、温度/功耗状态与 CPU 负载；记录运行条件再比较，别只挑最好的一次。CPU ORT 与 GPU torch 的数字不能直接归因于后端优劣，端到端还需纳入采集、预处理、拷贝、排队、后处理。<strong>面试映射：</strong>解释异步计时陷阱、预热、median/p95 与请求样本数；batch 时延不能直接当成单图时延。</p>
<h2>6 · TensorRT：先核对本机版本，再构建 engine</h2>
<p>TensorRT 不由上面的 pip 命令安装。请按 <a href="https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html" target="_blank" rel="noopener noreferrer">NVIDIA TensorRT 安装文档</a>及所选版本支持矩阵准备独立环境，核对操作系统、驱动、CUDA 与 GPU。下载软件包、容器或依赖均需你明确执行，本教程不自动执行，也不假定 trtexec 位于 PATH。切换环境后重新设置 WORK；如果 trtexec 未在 PATH，用实际安装位置调用，不能凭空写一个路径。</p>
<pre class="codebox"><code>command -v trtexec
trtexec --help
# 确认本机 --help 支持下面全部参数再执行：
WORK="/tmp/inference-handbook-lab"
trtexec --onnx="$WORK/resnet18.onnx" --saveEngine="$WORK/resnet18-fp16.engine" --fp16 --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:8x3x224x224 --shapes=input:1x3x224x224</code></pre>
<p>profile 的 min/opt/max 定义允许的 batch 范围及优化点；--shapes 选择本次实际运行的 batch=1。--fp16 是允许 FP16 优化，不保证每层都用 FP16。版本之间参数可能调整：若本版本没有 --fp16 或其他选项，请停止使用这组命令，按该版本文档选择精度工作流，不能把未知参数错误当成模型错误。engine 通常与 TensorRT/GPU/构建配置有兼容约束，不能假定随处可复制运行。</p>
<p><strong>成功信号：</strong>engine 文件生成，日志有当前运行 shape 和性能统计。<strong>失败排查：</strong>先核对版本、参数支持、输入名 input、profile 与 ONNX；再读解析器不支持算子或显存错误。保留构建日志与软件版本，实测 batch=8 前应再按本版本参数明确选择该 shape。<strong>面试映射：</strong>解释为什么 profile、精度选择和运行 shape 都是性能报告的一部分。</p>
<p><strong>边界：</strong>trtexec 不等于输出精度验证，本脚本 compare 仅比较 PyTorch 与 ORT CPU，未比较 TensorRT FP16 输出。应另写 TensorRT 输出采集，在相同输入上做数值及真实任务验证。trtexec 日志的 engine/GPU 计算性能也不等于业务端到端时延；按照本版本日志区分 GPU compute、传输和 host latency，不能把某一个数字改名为业务响应时间。</p>
<details><summary><span class="eye" aria-hidden="true"></span>下一关：接入 C++，当前没有完整实装</summary>
<p>下一关需要实现 C++ 输入预处理、ONNX/TensorRT 运行时绑定、缓冲所有权、stream/event 生命周期、输出解码与异常处理，再逐层比对同一固定输入并测端到端时延。这两个文件没有提供完整 C++ 集成，没有完成 TensorRT 输出验证或真实任务评测，不能把这次 Python/ONNX 冒烟描述为生产闭环。面试可诚实展示已有 JSON/CSV，清楚标注已跑后端、未完成步骤与测量口径。</p>
</details>
<ul class="checks"><li>我能下载或复制脚本，并从 inspect 识别真实运行环境。</li><li>我保存了固定输入、参考输出与 ONNX 数值报告，知道它们不证明任务 accuracy。</li><li>我能说明每个 benchmark 的设备、batch、预热、样本数和计时范围。</li><li>我知道 TensorRT 输出验证、真实任务评估和 C++ 集成仍需单独完成。</li></ul>
`
};
