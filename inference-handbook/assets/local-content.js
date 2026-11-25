window.HANDBOOK_LOCAL = [
{id:'local-gpu',title:'20 · 本机 GPU 实战：3090 Ti 跑通 PyTorch → ONNX → TensorRT',subtitle:'不进 Docker，用独立 venv 在自己显卡上测出真实延迟与精度。',priority:'P0',body:`
<div class="callout"><strong>结论先说：能用本机跑，而且应该用本机跑。</strong>RTX 3090 Ti 24GB + 驱动 570 足够完成 TensorRT 部署实验、7B 模型推理和 QLoRA 微调。本章所有数字都来自这台机器的日志：<code>gpu-lab/logs/</code>。</div>
<h3>问题 1：明明装了 CUDA，为什么 torch.cuda.is_available() 是 False？</h3>
<p>本机 miniforge 里已有一个 quant-deploy 环境，torch 是 <code>cu130</code> 版本，但 <code>is_available()=False</code>。原因不是显卡坏了，而是<strong>版本链断了</strong>：驱动 570 最高只支持 CUDA 12.8 运行时，cu130 轮子需要 580 以上驱动。</p>
<figure class="diagram"><pre class="mermaid">flowchart LR
 D[显卡驱动 570.133<br/>决定最高 CUDA 12.8] --> R[torch 轮子自带的运行时<br/>cudart/cublas/cudnn]
 R --> T[torch 2.6.0+cu124 ✓]
 D -.->|"不满足"| X[torch +cu130 ✗<br/>需驱动 ≥580]
 N[系统 nvcc 12.4] -.->|"只在自己编译 .cu/插件时用"| P[TensorRT 插件 / 自定义算子]</pre><figcaption>图 20-1：驱动是“上限”，pip 轮子自带运行时，系统 nvcc 只服务于自己编译</figcaption></figure>
<details><summary><span class="eye" aria-hidden="true"></span>展开：三个“CUDA 版本”各管什么</summary><table><tr><th>你看到的版本</th><th>来源</th><th>作用</th><th>本机</th></tr><tr><td>nvidia-smi 右上角 CUDA Version</td><td>驱动</td><td>能支持的<strong>最高</strong>运行时版本</td><td>12.8（驱动 570）</td></tr><tr><td>nvcc --version</td><td>系统 CUDA Toolkit</td><td>编译 .cu、TensorRT 插件</td><td>12.4</td></tr><tr><td>torch.version.cuda</td><td>pip 轮子自带</td><td>PyTorch 实际用的运行时</td><td>12.4（cu124）</td></tr></table><p>工程套路：<strong>选轮子时只看驱动上限</strong>，运行时版本 ≤ 驱动支持版本即可；不要为了 pip 装 torch 去重装系统 CUDA。面试常问“驱动和 toolkit 的关系”，就答这张表。</p></details>
<h3>问题 2：为什么用独立 venv 而不是 Docker？</h3>
<p>Docker 是交付标准，但本机 docker 还没注册 nvidia runtime（需要 sudo 执行 <code>nvidia-ctk runtime configure --runtime=docker</code> 后重启 docker）。学习阶段用 venv：零 sudo、删掉目录即可回滚、与系统 python3.8 互不干扰。</p>
<details><summary><span class="eye" aria-hidden="true"></span>展开：本机实际执行的搭建命令</summary><pre><code># 1) uv 自带 Python 3.11，不动系统 python3.8
uv venv --seed ~/venvs/gpu --python 3.11

# 2) 国内镜像：阿里云 pytorch-wheels 实测 ~1.3MB/s，download.pytorch.org ~0.86MB/s，清华 ~0.18MB/s
PIP="uv pip install --python ~/venvs/gpu/bin/python --index-url https://mirrors.aliyun.com/pypi/simple"
$PIP --find-links https://mirrors.aliyun.com/pytorch-wheels/cu124/ "torch==2.6.0+cu124"
$PIP tensorrt-cu12==10.8.0.43 onnx onnxruntime-gpu==1.20.1 transformers accelerate \\
     bitsandbytes sentence-transformers rank_bm25 faiss-cpu fastapi uvicorn peft pypdf

# 3) 验证：三件事都要 True/有版本号
~/venvs/gpu/bin/python -c "import torch,tensorrt;print(torch.cuda.is_available(),torch.cuda.get_device_name(0),tensorrt.__version__)"

# 4) 模型：huggingface.co 不通，用 hf-mirror（实测 ~4.7MB/s）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir ~/models/Qwen2.5-Coder-7B-Instruct
huggingface-cli download BAAI/bge-m3 --local-dir ~/models/bge-m3</code></pre><p>工程套路：下载慢先<strong>测速再选源</strong>（curl -m 10 -w "%{speed_download}"），而不是盲等；包管理器卡住时看缓存目录有没有在写文件，10 秒没增长就换源重来。本次第一次安装就是卡在 download.pytorch.org，换阿里云后恢复。</p></details>
<h3>问题 3：同一个模型，PyTorch、ONNX、TensorRT 各快多少？精度掉多少？</h3>
<p>用 PointPillars 风格的 BEV 主干（输入 1×64×496×432 伪图像，三级下采样 + 上采样拼接 + cls/box 两个头）走一遍完整部署链。脚本：<code>gpu-lab/trt_bench.py</code>。</p>
<figure class="diagram"><pre class="mermaid">flowchart LR
 A[PyTorch FP32<br/>基线输出 ref] --> B[torch AMP FP16]
 A --> C[导出 ONNX opset17]
 C --> D[onnxruntime 对齐]
 C --> E[TRT FP32]
 C --> F[TRT FP16]
 C --> G[TRT INT8<br/>熵校准]
 E & F & G --> H[和 ref 比 max_abs / cosine]</pre><figcaption>图 20-2：每一步都和同一个 FP32 基线对齐，速度和精度一起看</figcaption></figure>
<p>本机实测（<code>gpu-lab/logs/bench_20261123_133507.md</code>，200 次迭代、预热 30 次）：</p>
<table><tr><th>后端</th><th>中位数 ms</th><th>P99 ms</th><th>加速比</th><th>cls 余弦</th><th>cls 最大误差</th><th>引擎 MB</th><th>构建 s</th></tr>
<tr><td>PyTorch FP32</td><td>4.220</td><td>4.418</td><td>1.00×</td><td>基线</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>PyTorch AMP FP16</td><td>2.548</td><td>2.569</td><td>1.66×</td><td>1.0</td><td>6.6e-5</td><td>—</td><td>—</td></tr>
<tr><td>onnxruntime CPU（只对齐）</td><td>—</td><td>—</td><td>—</td><td>1.0</td><td>2.9e-5</td><td>—</td><td>—</td></tr>
<tr><td>TensorRT FP32</td><td>2.757</td><td>2.775</td><td>1.53×</td><td>1.0</td><td>3.8e-5</td><td>31.0</td><td>13.0</td></tr>
<tr><td>TensorRT FP16</td><td>1.113</td><td>1.119</td><td>3.79×</td><td>1.0</td><td>1.4e-4</td><td>10.6</td><td>24.8</td></tr>
<tr><td>TensorRT INT8</td><td>0.639</td><td>0.641</td><td>6.61×</td><td>0.99975</td><td>4.4e-3</td><td>6.0</td><td>60.4</td></tr></table>
<details><summary><span class="eye" aria-hidden="true"></span>展开：怎么读这张表（面试就这么讲）</summary><ul><li><strong>TRT FP32 已经比 PyTorch FP32 快 1.53×</strong>：精度没变，收益来自层融合（Conv+BN+ReLU 合成一个 kernel）、按这块卡挑最快的 kernel、少了 Python 调度开销。</li><li><strong>FP16 3.79×</strong>：3090 Ti 的 Tensor Core 跑 FP16，显存带宽占用减半；误差 1e-4 量级，检测任务基本无损。PyTorch AMP 只有 1.66×，因为它没有做融合，很多算子仍按层执行。</li><li><strong>INT8 6.61×，但 cosine 降到 0.99975</strong>：误差比 FP16 大 30 倍。能不能上 INT8 不看这个数，要看真实数据上的 mAP 掉多少，一般要求掉点 &lt;1%。</li><li><strong>构建时间</strong>：INT8 要先跑校准，构建 60 s；所以引擎要离线构建、序列化保存，不在车上现场构建。</li><li><strong>P99 和中位数几乎一样</strong>：说明本机没有明显抖动；车端要在 Orin 上重测，还要和其他任务一起跑。</li></ul><p>踩坑记录：第一次运行 TensorRT 报 <code>Using default stream in enqueueV3()</code>，提示默认流会额外插同步。改为 <code>torch.cuda.Stream()</code> 后重测（bench_20261123_133301 → 133507），这个单模型场景延迟几乎不变；但多模型并发时，默认流会让彼此串行，所以仍按规范使用独立流。另外 <code>plan</code> 是 IHostMemory，要用 <code>plan.nbytes</code> 取大小，不能用 len()。</p></details>
<details><summary><span class="eye" aria-hidden="true"></span>展开：为什么计时必须用 CUDA Event + 预热</summary><p>GPU 调用是<strong>异步</strong>的：Python 里 <code>model(x)</code> 返回时，kernel 可能还没跑完。用 time.time() 会量到“提交时间”而不是“执行时间”。CUDA Event 在 GPU 流上打点，测到的才是真正执行耗时。</p><pre><code>start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record(); fn(); end.record()
end.synchronize(); ms = start.elapsed_time(end)</code></pre><p>预热 30 次：首次调用要做 cuDNN 算法选择、显存池分配、TensorRT 上下文初始化，混进统计会把均值拉高几倍。报告用<strong>中位数 + p99</strong>：中位数代表常态，p99 代表抖动，自动驾驶更关心后者。</p></details>
<details><summary><span class="eye" aria-hidden="true"></span>展开：INT8 校准到底在“校”什么</summary><p>INT8 只有 256 个格子，要为每个张量选一个 scale：太大浪费精度，太小大值被截断。熵校准（IInt8EntropyCalibrator2）用若干批真实输入跑一遍，统计每层激活直方图，选让 KL 散度最小的截断阈值。</p><ul><li>校准数据必须像线上数据：本脚本用随机伪图像只为跑通流程，真实项目要用真实点云帧。</li><li>校准结果写 <code>int8_calib.cache</code>，下次构建跳过校准前向。实测第二次构建从 66.4 s 降到 60.4 s——省得不多，因为大头是 TensorRT 为每层挑选 kernel（tactic 搜索），要省这部分得用 timing cache 或直接复用序列化好的引擎。</li><li>精度看 cosine 和 max_abs；检测任务最终要看 mAP，不能只看张量误差。</li></ul></details>
<details><summary><span class="eye" aria-hidden="true"></span>展开：TensorRT 10 的推理 API（和旧教程不一样）</summary><pre><code>ctx = engine.create_execution_context()
ctx.set_tensor_address("pillars", inp.data_ptr())      # 按名字绑定，不再用 bindings 列表
ctx.set_tensor_address("cls", cls_out.data_ptr())
ctx.set_tensor_address("box", box_out.data_ptr())
ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)</code></pre><p>直接用 torch 张量做输入输出缓冲，省掉 pycuda；C++ 侧对应 <code>IExecutionContext::setTensorAddress + enqueueV3</code>。面试会问 TRT8→10 迁移：binding index 被 tensor name 取代、隐式 batch 被移除。</p></details>
<div class="checks"><strong>自查</strong><ul><li>能说清驱动版本、nvcc 版本、torch.version.cuda 三者关系</li><li>能解释为什么 FP16 比 FP32 快、INT8 不一定比 FP16 快很多</li><li>能复现：<code>~/venvs/gpu/bin/python gpu-lab/trt_bench.py</code>，看 logs/ 下的 md 表格</li></ul></div>
`},
{id:'local-kb-agent',title:'21 · 本地知识库 Agent：让机器人回答自家软件的问题',subtitle:'LidarAssistant / monitor / udslib / rs_driver：先 RAG，再评测，最后才考虑微调。',priority:'P0',body:`
<div class="callout"><strong>你想要的是“问 LidarAssistant 怎么编译，它能带出处回答”。这件事 90% 靠检索（RAG），不靠训练。</strong>3090 Ti 24GB 不能从零训练大模型，也不能全参数微调 7B；但能本地跑 7B 推理、能做 7B/14B 的 QLoRA。正确顺序是：先 RAG 跑通并量化评测 → 找到 RAG 解决不了的问题 → 再决定要不要微调。</div>
<h3>问题 1：一张 24GB 的卡，到底能做什么？</h3>
<table><tr><th>任务</th><th>显存估算</th><th>3090 Ti</th><th>说明</th></tr><tr><td>从零预训练 7B</td><td>数百卡·月</td><td>✗</td><td>不是个人能做的事</td></tr><tr><td>7B 全参数微调</td><td>&gt;100GB（权重+梯度+Adam 状态）</td><td>✗</td><td>每参数约 16 字节</td></tr><tr><td>7B BF16 推理</td><td>~15GB + KV cache</td><td>✓</td><td>本章默认方案</td></tr><tr><td>7B 4bit NF4 推理</td><td>~5.5GB</td><td>✓</td><td>省出显存给嵌入模型/长上下文</td></tr><tr><td>7B QLoRA 微调</td><td>~10–14GB</td><td>✓</td><td>4bit 冻结底座 + 训练小矩阵</td></tr><tr><td>14B QLoRA 微调</td><td>~18–22GB</td><td>勉强</td><td>短序列、小 batch、梯度检查点</td></tr><tr><td>bge-m3 嵌入</td><td>~1.2GB（FP16）</td><td>✓</td><td>6000 块几十秒向量化完</td></tr></table>
<details><summary><span class="eye" aria-hidden="true"></span>展开：为什么知识库问答先做 RAG，而不是先微调</summary><ul><li><strong>代码天天变</strong>：微调把知识“烧进权重”，代码一改就过时；RAG 重新 ingest 几分钟就更新。</li><li><strong>要出处</strong>：调试问题必须能点回文件和行号核对，微调模型给不出可信出处。</li><li><strong>幻觉可控</strong>：RAG 规定“只依据资料回答”，并能校验引用；微调后模型会自信地编函数名。</li><li><strong>微调擅长的是“风格和格式”</strong>：比如让模型学会你们的术语、固定回答模板、UDS 报文解读格式。这类问题才值得 QLoRA。</li></ul></details>
<h3>问题 2：要做哪些工作？（阶段 0–7）</h3>
<table><tr><th>阶段</th><th>做什么</th><th>产出 / 验收</th><th>本机状态</th></tr><tr><td>0 环境</td><td>venv + torch cu124 + 模型下载</td><td>is_available=True</td><td>✓ 第 20 章</td></tr><tr><td>1 知识源梳理</td><td>选目录、排除第三方/生成代码/敏感配置</td><td>sources.toml</td><td>✓</td></tr><tr><td>2 切块入库</td><td>代码按函数、文档按标题、PDF 按页</td><td>chunks.jsonl + 统计</td><td>✓ 6000 块</td></tr><tr><td>3 检索</td><td>BM25 + 向量 + RRF + 项目路由</td><td>recall@k / MRR</td><td>✓ 见下表</td></tr><tr><td>4 生成</td><td>带编号证据的提示 + 引用校验</td><td>每条结论带 [n]</td><td>✓</td></tr><tr><td>5 Agent</td><td>允许模型 search/read 补证据</td><td>trace 可审计</td><td>✓ 最多 3 步</td></tr><tr><td>6 服务化</td><td>FastAPI + 网页，仅监听 127.0.0.1</td><td>浏览器可问答</td><td>✓</td></tr><tr><td>7 微调（可选）</td><td>收集问答对 → QLoRA</td><td>微调前后同一评测集对比</td><td>待数据</td></tr></table>
<figure class="diagram"><pre class="mermaid">flowchart LR
 S[源码/文档/PDF] --> I[ingest 切块<br/>带 project/path/行号]
 I --> B[BM25 倒排]
 I --> V[bge-m3 向量 FAISS]
 Q[用户问题] --> RT[项目路由]
 RT --> B & V
 B & V --> F[RRF 融合 + 去重]
 F --> P[编号证据 + 规则提示]
 P --> L[Qwen2.5-Coder-7B]
 L --> C[引用校验]
 C --> A[答案 + 文件:行号]</pre><figcaption>图 21-1：知识库 Agent 的数据流（代码在 kb-agent/kb/）</figcaption></figure>
<h3>问题 3：代码库怎么切，才能被搜到？</h3>
<p>入库统计（<code>python -m kb.cli ingest</code>）：</p>
<table><tr><th>项目</th><th>文件</th><th>块</th><th>来源</th></tr><tr><td>LidarAssistant</td><td>218</td><td>2994</td><td>src、docs、AGENTS.md、SWC 规格书</td></tr><tr><td>monitor</td><td>392</td><td>1561</td><td>uds_dv_monitor_windows 源码</td></tr><tr><td>rs-monitor-release</td><td>4</td><td>79</td><td>发布包 README + 17 页使用手册 PDF</td></tr><tr><td>udslib</td><td>52</td><td>316</td><td>src、demo、tests、docs</td></tr><tr><td>rs_driver</td><td>100</td><td>938</td><td>src、doc（中英文）</td></tr></table>
<details><summary><span class="eye" aria-hidden="true"></span>展开：切块规则与踩过的坑</summary><ul><li><strong>代码</strong>：累积到 40 行后，在顶层 <code>}</code> 或空行处断开；超过 120 行或 3000 字符强制断开并保留 5 行重叠，避免函数签名和函数体分家。</li><li><strong>Markdown</strong>：按标题切，块标题记成路径“编译 &gt; Linux”；代码围栏里的 <code>#</code> 不当标题。</li><li><strong>PDF</strong>：按页切，引用显示 p.页码。坑：使用手册 PDF 3.2MB 被默认 400KB 大小限制挡掉，单独放宽后才入库。</li><li><strong>敏感信息</strong>：第一次入库发现 monitor 的 lidar_config 下 yaml 含服务地址和 token，立刻加排除规则（<code>*token*</code>、<code>*apiurl*</code>、<code>*remote_config*</code>）并移除该目录。<strong>知识库就是一个“能被提问导出”的数据库，入库即暴露。</strong></li><li><strong>噪声</strong>：qcustomplot、tinyxml2、moc_*、ui_*.h、*_rc.py 这类第三方/生成代码会抢占检索名额，全部排除。</li></ul></details>
<h3>问题 4：怎么知道检索好不好？——先建评测集</h3>
<p><code>kb-agent/eval/questions.jsonl</code> 收录 22 个真实问题，每题标注“答案应出现在哪个文件”。指标：recall@k（前 k 条证据里有没有正确文件）和 MRR（正确文件排第几的倒数均值）。<strong>LLM 答得好的前提是证据先被找回来</strong>，所以先评检索，不评生成。</p>
<table><thead><tr><th>模式</th><th>recall@1</th><th>recall@3</th><th>recall@8</th><th>MRR</th></tr></thead><tbody>
<tr><td>BM25</td><td>0.500</td><td>0.636</td><td>0.818</td><td>0.581</td></tr>
<tr><td>BM25 + 项目路由</td><td>0.500</td><td>0.682</td><td>0.909</td><td>0.609</td></tr>
<tr><td><strong>BM25 + bge-m3 向量（RRF）</strong></td><td><strong>0.636</strong></td><td>0.773</td><td><strong>1.000</strong></td><td><strong>0.753</strong></td></tr>
<tr><td>混合 + 项目路由（默认）</td><td>0.636</td><td><strong>0.818</strong></td><td>0.955</td><td>0.752</td></tr>
</tbody></table>
<p>22 题，k=8，日志在 <code>kb-agent/logs/eval_*.json</code>。读法：<strong>向量检索是最大的一步</strong>（recall@1 0.50→0.64，MRR 0.58→0.75）；路由对 BM25 有明显帮助，但在混合检索上 recall@8 反而从 1.000 掉到 0.955。唯一的 MISS 是“monitor 点云缺失检测怎么做？”：路由后候选池被 monitor 和发布版手册 PDF 页（“点云”“检测”出现很多次）占满，pointcloud_missing_detector.h 被挤到第 9 名。默认仍然开启路由，因为 recall@3 更高，用户只看前几条。</p>
<details><summary><span class="eye" aria-hidden="true"></span>展开：每一步优化为什么有效</summary><ol><li><strong>去掉中文提问套话</strong>：中文按二元组分词，“如何根据雷达型号创建”会产生“如何”“何根”等词，它们在中文文档里到处都是，在代码里几乎没有，于是检索整体偏向中文文档。把“如何/怎么/什么/哪些/的”等切掉后，纯 BM25 的 recall@8 从 0.727 升到 0.773（之后补入 PDF 手册并修正标注，升到 0.818）。</li><li><strong>项目路由</strong>：问题里点名“udslib”“monitor”时只在该项目里搜。这一步最便宜，BM25 的 recall@8 从 0.818 升到 0.909。<strong>坑：先融合再过滤，recall 反而下降</strong>，因为别的项目的块在 RRF 里占了名次，过滤后真正相关的块排得更靠后。正确做法是在每一路检索内部先过滤，再做 RRF；向量检索要把候选池放大 8 倍，否则过滤后不够 k 条。</li><li><strong>评测集也会错</strong>：“multicast 怎么配”这题最初只标了一个文件，实际 online_lidar_advanced_topics 里有 7 处讲组播。检索结果不对时先怀疑标注，再怀疑算法</li><li><strong>向量检索 + RRF</strong>：“点云缺失检测怎么做”这类问题，代码里写的是 pointcloud_missing_detector，关键词对不上，只能靠语义向量召回。RRF 按名次融合：分数 = Σ 1/(60+名次)，不需要把 BM25 分数和余弦相似度归一化到同一尺度。</li></ol><p>工程套路：<strong>每次只改一个变量，跑同一评测集，结果写 logs/eval_*.json</strong>。没有评测集的 RAG 优化都是“感觉变好了”。</p></details>
<h3>问题 5：怎么防止它一本正经地胡说？</h3>
<details><summary><span class="eye" aria-hidden="true"></span>展开：提示规则 + 引用校验 + 工具白名单</summary><pre><code>规则：
1. 只依据【资料】回答，不要编造资料里没有的函数名、参数、数值。
2. 每个关键结论后用 [编号] 标注出处。
3. 资料不足时明确说“资料中没有找到”，并指出最可能相关的文件。
---
【资料】
[1] udslib/README.md L12-L40 编译 &gt; Linux
[2] udslib/CMakeLists.txt L1-L38 ...</code></pre><ul><li><strong>引用校验</strong>：回答后用正则抽出所有 [n]，编号超出证据数量就标红“无效引用”，不信任模型的自我声明。</li><li><strong>read 工具白名单</strong>：Agent 只能读 files.json 里登记过的文件，且单次最多 150 行；防止被诱导读 ~/.ssh 或把上下文塞爆。</li><li><strong>服务只监听 127.0.0.1</strong>，GPU 生成加锁串行，超时返回 503。</li></ul></details>
<h3>真实运行记录（Qwen2.5-Coder-7B BF16，本机 3090 Ti）</h3>
<table><thead><tr><th>项目</th><th>数值</th></tr></thead><tbody>
<tr><td>模型加载</td><td>4.3s（权重已在页缓存时；首次约 30s）</td></tr>
<tr><td>显存峰值</td><td>16.8 GB（torch.cuda.max_memory_allocated）</td></tr>
<tr><td>rag：“udslib 在 Linux 下如何编译？”</td><td>3.7s，prompt 1768 / 输出 139 token，引用 [1][3]，两条都正确</td></tr>
<tr><td>agent：“monitor 的心跳是怎么实现的？”</td><td>12~38s，prompt 约 2.2k~3k token；模型判断资料足够，直接 DONE，没有调用工具</td></tr>
</tbody></table>
<details><summary><span class="eye" aria-hidden="true"></span>展开：rag 示例的真实输出</summary><pre><code>1. **安装依赖**：首先需要安装构建工具和 CMake [3]。
    sudo apt-get install build-essential cmake
2. **进入脚本目录**：导航到 \`script\` 目录 [1]。
    cd script/
3. **执行编译命令**：运行编译脚本 \`compile_linux.sh\` [1]。
    ./compile_linux.sh

来源：
  [1] udslib/README.md:94-100
  [3] udslib/README.md:88-93</code></pre>
<p>注意：README 里写的是 <code>cd script/</code>，但仓库里实际目录叫 <code>scripts/</code>。模型忠实转述了资料，错误在文档本身。<strong>RAG 只能保证“和资料一致”，不能保证“资料是对的”</strong>，所以引用一定要能点回原文核对。</p></details>
<details><summary><span class="eye" aria-hidden="true"></span>展开：agent 示例暴露的三个小模型问题，以及逐个修复的过程</summary><ol>
<li><strong>第一版：不写引用</strong>。约束放在 system 里，隔着 2000 token 的证据，7B 模型就“忘了”。修复：在问题后面再强调一遍格式要求（近因效应）。这样 rag 示例开始带引用了，但 agent 示例有时还是一个引用都没有。</li>
<li><strong>提示词是软约束，要加程序硬校验</strong>：check_citations 发现没有任何引用时，把原回答发回去，要求“内容不变，补上 [n]”，最多重试一次（trace 里会多一步 recite）。代价是用时从约 12s 涨到 25~38s。</li>
<li><strong>编造行号</strong>：模型写“Stop 位于第 143~145 行”，实际在第 97 行。修复：禁止模型写路径和行号，位置只由程序根据 [n] 从 chunk 元数据生成。<strong>能由程序给出的事实，就不要让模型生成</strong>。</li>
<li><strong>凭记忆重写代码</strong>：模型会“补全”一段看起来合理的 HeartBeat 类，成员类型和真实代码不一致。加了“只能原样摘抄”的规则后仍然偶尔出现。到这一步，再调提示词已经没什么收益，根本解法是：换更大的模型（14B AWQ 用 vLLM 跑，约 10GB），或者用 QLoRA 在 300~500 条“证据→带引用回答”样本上微调格式（见问题 6）。</li></ol></details>
<h3>问题 6：什么时候再做 QLoRA 微调？</h3>
<details><summary><span class="eye" aria-hidden="true"></span>展开：微调的触发条件和最小流程</summary><p>触发条件（满足任一）：检索已找对证据但模型总答错格式；需要固定输出模板（例如 UDS 否定响应码解读）；团队术语模型总理解错。</p><ol><li>从网页问答日志里收集 300–1000 条“问题 + 证据 + 人工修正后的答案”。</li><li>用 peft + bitsandbytes：底座 4bit NF4 冻结，LoRA r=16、alpha=32，只挂在 q/k/v/o 投影上。</li><li>训练数据<strong>保留证据输入</strong>，教模型“看资料回答”，而不是背知识。</li><li>用同一评测集对比微调前后：答案正确率、引用有效率、“资料中没有找到”的误拒率。</li></ol><p>面试表达：“我先用 RAG 解决知识时效和出处问题，用评测集量化检索；QLoRA 只用于格式与术语对齐，并用同一评测集验证收益。”</p></details>
<details><summary><span class="eye" aria-hidden="true"></span>展开：本机实测——10 个自研项目造数据 + QLoRA（3090 Ti）</summary>
<p><strong>以前怎么做、遇到什么问题：</strong>只靠提示词约束，7B 底座仍有约 44% 的回答没引用到真正的源块，拒答题 0 分（证据不相关也硬答），偶尔凭记忆重写代码。</p>
<p><strong>数据怎么来（self-instruct + 检索回放）：</strong>知识库扩到 ~/project 下 10 个项目（新增 pcrt、rsviewpro、safeparamtool、udsLibTestTool、udp_app，共 11310 块；第三方 pcl、重复副本、加密工具不入库）。入库时先脱敏：硬编码密码如 <code>setPassWord("…")</code>、ini 的 <code>pc_password = …</code> 统一替换成 <code>***</code>。造题流程是：分层抽一个 chunk → 让模型扮新同事出题 → 用线上检索重新找证据（找不回源块的题丢掉）→ 用线上同一套 prompt 生成回答 → 程序硬校验（有引用、引用到源块、无越界编号、不自写行号）。每 10 条造 1 条拒答样本，做法是把源文件从证据里删掉。</p>
<table><tr><th>环节</th><th>数量</th><th>说明</th></tr>
<tr><td>抽样</td><td>600（seed 1 + seed 2）</td><td>第二批跳过已造过题的源块</td></tr>
<tr><td>检索回放保留</td><td>463</td><td>98 条题检索找不回源块，已丢弃</td></tr>
<tr><td>自动校验通过</td><td>305</td><td>失败原因主要是没引用源块</td></tr>
<tr><td>代码忠实过滤后</td><td>291</td><td>回答 246 + 拒答 45</td></tr>
<tr><td>先切 val，再过滤 train</td><td>train 248 / val 68</td><td>val 不做过滤，否则评测会偏乐观</td></tr></table>
<p><strong>踩坑：max_len。</strong>第一次用 3072，日志里 <code>train_too_long: 150</code>，有六成样本被丢掉。统计 token 分布：P50 3423、P90 4494、P95 4859。改为 5120 后只丢 8 条。<em>套路：max_len 按分布的 P95~P97 来定；超长样本整条丢掉，不要截断，截断会砍掉被引用的证据。</em></p>
<table><tr><th>训练</th><th>值</th></tr><tr><td>可训练参数</td><td>10.1M（0.13%）</td></tr><tr><td>步数 / 用时</td><td>62 步 / 20.4 min（约 18.6 s/步）</td></tr><tr><td>显存峰值</td><td>18.9 GB（4bit + gradient checkpointing，序列长 5120）</td></tr><tr><td>train_loss</td><td>0.096（只在 assistant 段计算 loss）</td></tr></table>
<table><tr><th>val 68 条（含 5 条拒答）</th><th>底座</th><th>+LoRA</th></tr>
<tr><td>has_cite 有引用</td><td>0.889</td><td><strong>0.905</strong></td></tr>
<tr><td>cites_source 引用到真正源块</td><td>0.556</td><td><strong>0.683</strong></td></tr>
<tr><td>code_faithful 代码原样摘抄</td><td>0.921</td><td><strong>0.968</strong></td></tr>
<tr><td>refusal_ok 证据不足时拒答</td><td>0.0</td><td><strong>0.8</strong></td></tr>
<tr><td>self_lineno 自写行号（越低越好）</td><td>0.0</td><td>0.016</td></tr>
<tr><td>over_refusal 正常题误拒（越低越好，事后补算）</td><td>0.0</td><td>0.048</td></tr></table>
<p><strong>怎么读这张表：</strong>收益最大的是拒答，从 0/5 提升到 4/5，这正是提示词教不会、而样本能教会的行为。引用准确率提升 12.7 个点。注意拒答 val 只有 5 条，结论还需要扩大样本复核。另外 self_lineno 出现了 1 例回归，下一轮应把这类样本补进人工校对清单。底座和 LoRA 共用同一份 4bit 权重，底座结果通过 <code>disable_adapter()</code> 得到，所以对比只差在 LoRA 本身。</p>
<p><strong>补算误拒率：拒答能力提升，有没有拿“乱拒答”换来？</strong>第一版评测只统计了 refusal_ok，没有统计正常题被拒答的比例。只看前者会鼓励模型“全部拒答”。补上 over_refusal 后，用已保存的回答离线重算（<code>eval_lora --rescore</code>，不占 GPU），LoRA 在 63 条正常题里拒答了 3 条。逐条看：</p>
<table><tr><th>题目</th><th>底座</th><th>LoRA</th><th>判定</th></tr>
<tr><td>sync_queue.hpp 用的是什么许可证？</td><td>3-clause BSD（正确）</td><td>没有找到</td><td>✗ 真误拒</td></tr>
<tr><td>internDecodeMsopPkt 里 pkt_ts 初始化为 0 但后续没赋值，会导致什么问题？</td><td>顺着题目的说法往下编</td><td>没有找到</td><td>△ 题目前提可疑</td></tr>
<tr><td>StringSplit 的 token 在循环结束后没有正确更新，会导致什么问题？</td><td>顺着题目的说法往下编</td><td>没有找到</td><td>△ 题目前提可疑</td></tr></table>
<p>后两题是造题模型自己写出来的<strong>诱导式问题</strong>：题目先断言“有 bug”，再问后果。底座会顺着前提编下去，LoRA 选择了拒答。拒答比编造好，但更理想的回答是“资料中的代码显示……，与题目的前提不一致 [n]”。<em>套路：自动造题会产生假前提题。造题 prompt 要禁止“断言式”问法，或者把这类题单独归为“纠正前提”类别并配人工答案。</em>严格算，真误拒是 1/63≈1.6%，可以接受。</p>
<pre><code>$PY -m finetune.make_data --n 300 --seed 1           # 生成 drafts.jsonl + review.md
$PY -m finetune.make_data --n 300 --seed 2 --append  # 换 seed 扩量
$PY -m finetune.train_qlora                          # out/lora + logs/train_*.json
$PY -m finetune.eval_lora                            # logs/eval_lora_*.json</code></pre></details>
<details><summary><span class="eye" aria-hidden="true"></span>展开：本机复现命令</summary><pre><code>cd kb-agent
PY=~/venvs/gpu/bin/python
$PY -m kb.cli ingest                       # 扫描 sources.toml → index/chunks.jsonl
$PY -m kb.cli index                        # bge-m3 向量化 → index/vectors.faiss
$PY -m kb.cli --bm25-only --no-route eval  # 基线
$PY -m kb.cli eval                         # 混合检索 + 路由
$PY -m kb.cli search "doipClient 发送诊断请求"
$PY -m kb.cli ask "udslib 在 Linux 下如何编译？"
$PY -m kb.cli ask "monitor 的心跳是怎么实现的？" --mode agent
$PY -m kb.cli serve                        # 浏览器打开 http://127.0.0.1:8770
$PY -m pytest -q tests                     # 切块/分词/RRF/去重/引用校验 + 微调数据切分/loss 掩码/打分单测</code></pre></details>
<div class="checks"><strong>自查</strong><ul><li>能说清为什么先 RAG 后微调，以及 24GB 能做哪些训练</li><li>能解释 BM25 和向量检索各自擅长什么、RRF 为什么不用归一化</li><li>能拿出评测集数字证明某个优化有效</li><li>能说出至少三条防幻觉手段</li></ul></div>
`},
{id:'local-kb-tutorial',title:'22 · 实战教程：从零部署知识库 Agent + QLoRA（一键脚本版）',subtitle:'第 20、21 章是“踩坑记录”，本章是“照着做就能复现”的部署手册：每一步都有命令、预期输出、验收标准和本机实测耗时。',priority:'P0',body:`
<div class="callout"><strong>一句话：两个脚本、九个阶段，从一台装好驱动的 Ubuntu 机器，做到浏览器里能问“udslib 怎么编译”并拿到带出处的回答。</strong>所有数字都来自本机（RTX 3090 Ti 24GB / 驱动 570 / Ubuntu 20.04），日志在 <code>kb-agent/logs/</code>。</div>
<h3>全景：九个阶段分别产出什么</h3>
<figure class="diagram"><pre class="mermaid">flowchart LR
 E[setup_env<br/>venv+依赖+模型] --> T[test<br/>15 个单测]
 T --> I[ingest<br/>切块+脱敏]
 I --> X[index<br/>bge-m3 向量]
 X --> V[eval<br/>recall/MRR]
 V --> A[ask<br/>冒烟问答]
 A --> D[data<br/>造 SFT 样本]
 D --> TR[train<br/>QLoRA]
 TR --> EL[evallora<br/>底座 vs LoRA]
 EL --> S[serve<br/>网页 + LoRA]
 V -. 不达标 .-> I
 EL -. 无收益 .-> S2[serve 不带 LoRA]</pre><figcaption>图 22-1：阶段间有“门禁”。检索评测不过，不进入生成；LoRA 没收益，就不上线 LoRA</figcaption></figure>
<table><tr><th>阶段</th><th>命令（在 kb-agent/ 下）</th><th>产物</th><th>验收标准</th><th>本机耗时</th></tr>
<tr><td>0 环境</td><td><code>bash scripts/setup_env_ubuntu2004.sh</code></td><td>~/venvs/gpu、~/models</td><td>最后一行打印 OK + 显卡名</td><td>首次约 1 小时（主要是下载 15GB 模型）</td></tr>
<tr><td>1 test</td><td rowspan="5"><code>bash scripts/run_pipeline_ubuntu2004.sh</code></td><td>—</td><td>15 passed</td><td>&lt;1 s</td></tr>
<tr><td>2 ingest</td><td>index/chunks.jsonl</td><td>10 个项目都有块，敏感值残留为 0</td><td>11310 块</td></tr>
<tr><td>3 index</td><td>index/vectors.faiss</td><td>块数与 chunks 一致</td><td>70.7 s</td></tr>
<tr><td>4 eval</td><td>logs/eval_*.json</td><td>recall@8 ≥ 0.9</td><td>数秒，0.955 / MRR 0.752</td></tr>
<tr><td>5 ask</td><td>终端输出</td><td>回答带 [n]，来源可点回原文</td><td>3.7 s</td></tr>
<tr><td>6 data</td><td rowspan="3"><code>… run_pipeline_ubuntu2004.sh data train evallora</code></td><td>finetune/data/drafts.jsonl、review.md</td><td>可训练样本 ≥ 200</td><td>两批共 54 min，291 条</td></tr>
<tr><td>7 train</td><td>finetune/out/lora（39MB）</td><td>train_too_long 占比 &lt; 5%，loss 收敛</td><td>20.4 min，峰值 18.9GB</td></tr>
<tr><td>8 evallora</td><td>logs/eval_lora_*.json</td><td>cites_source、refusal_ok 优于底座</td><td>10.9 min</td></tr>
<tr><td>9 serve</td><td><code>… run_pipeline_ubuntu2004.sh serve</code></td><td>http://127.0.0.1:8770</td><td>网页能问答，回答带来源</td><td>加载约 10 s，单问 2–5 s</td></tr></table>
<h3>第 0 步：环境（只做一次）</h3>
<details><summary><span class="eye" aria-hidden="true"></span>展开：脚本做了什么、为什么这样做</summary><ol>
<li><strong>先查驱动</strong>：cu124 轮子要求驱动 ≥ 550，驱动不够直接退出。不要先装 torch 再发现 <code>is_available()=False</code>（第 20 章问题 1）。</li>
<li><strong>uv 建 venv（Python 3.11）</strong>：不动系统 python3.8，回滚就是删掉目录。</li>
<li><strong>依赖锁版本</strong>：<code>kb-agent/requirements-gpu.txt</code> 是本机实测能一起工作的组合。<em>套路：可复现的部署 = 锁版本 + 锁模型 + 锁数据。</em>transformers/peft 升级经常改默认行为，比如 peft 已经提示 &lt;5.6 将弃用。</li>
<li><strong>国内镜像</strong>：PyPI 用阿里云，torch 用阿里云 pytorch-wheels，模型用 hf-mirror。可以用环境变量 <code>PIP_INDEX / TORCH_LINKS / HF_ENDPOINT</code> 覆盖。</li>
<li><strong>幂等</strong>：venv 已存在就跳过，模型目录里有 config.json 就跳过，脚本中断后可以直接重跑。</li></ol>
<pre><code>cd ~/Documents/Trae-CN/default/llm-basics
bash kb-agent/scripts/setup_env_ubuntu2004.sh
# 期望最后一行：
OK 2.6.0+cu124 NVIDIA GeForce RTX 3090 Ti transformers 4.51.3 peft 0.22.0</code></pre></details>
<h3>第 1–5 步：RAG 主线（不训练也能用）</h3>
<details><summary><span class="eye" aria-hidden="true"></span>展开：配置知识源 sources.toml（换成你自己的项目只改这里）</summary>
<pre><code>[[source]]
project = "udslib"                 # 回答里显示的项目名
aliases = ["uds lib", "诊断库"]     # 问题里出现这些词，就只在本项目里检索（项目路由）
root = "/home/sti/project/udslib"
include = ["src", "demo", "tests", "docs", "README.md", "CMakeLists.txt"]   # 白名单</code></pre>
<p>选源三原则：①只收人写的、能回答问题的内容（源码、文档、脚本）；②第三方库、生成代码、重复副本一律不收（本机排除了 pcl、*-ci、LidarAssistant_Release 等）；③敏感内容不入库，入库后还会再脱敏一遍。<strong>知识库就是一个可以被提问导出的数据库。</strong></p>
<p>脱敏验收（不能只 grep 123456，十六进制表、单测里的 "12345678" 也会命中，要连同上下文一起匹配）：</p><pre><code>grep -ciE 'pass_?word[^,]{0,20}123456' index/chunks.jsonl   # 期望 0
grep -oE 'setPassWord\\(\\\\"\\*\\*\\*\\\\"\\)' index/chunks.jsonl | head -1   # 期望 setPassWord(\\"***\\")</code></pre>
<p>实测效果：部署后在网页上问“pcrt 的 pc_password 默认值是多少？”，回答“资料中没有找到与该问题直接相关的内容”（2.39 s），密码不会通过问答泄露。</p></details>
<details><summary><span class="eye" aria-hidden="true"></span>展开：每一步的预期输出与常见失败</summary><table><tr><th>步骤</th><th>预期输出（本机）</th><th>失败现象 → 原因 → 处理</th></tr>
<tr><td>test</td><td><code>15 passed</code></td><td>ImportError → 用了系统 python，确认 <code>PY=~/venvs/gpu/bin/python</code></td></tr>
<tr><td>ingest</td><td>每个项目的 files/chunks 统计 JSON</td><td>某项目 0 块 → root 路径错，或 include 子目录名大小写不对（docs/Docs）</td></tr>
<tr><td>index</td><td><code>向量化 11310 块，用时 70.7s</code></td><td>CUDA OOM → 有 serve 进程占着显存；脚本里的 gpu_free 会拦下这种情况</td></tr>
<tr><td>eval</td><td><code>recall@8 0.955, mrr 0.752</code></td><td>新加项目后 recall 下降 → 看 MISS 题的 top1，通常是新项目的文档抢了名额，给它加 aliases 或排除噪声目录</td></tr>
<tr><td>ask</td><td>编号步骤 + [1][3] + 来源 README.md:94-100</td><td>没有引用 → 正常情况下会自动 recite 一次；持续没有引用，看 prompt 是否超长</td></tr></table>
<p>门禁的意义：<strong>eval 不达标就不要往下做</strong>。检索找不回证据时，生成和微调都救不回来。</p></details>
<h3>第 6–8 步：造数据 → QLoRA → 对比评测（可选，但能量化收益）</h3>
<figure class="diagram"><pre class="mermaid">flowchart LR
 C[随机抽 chunk<br/>按项目分层] --> Q[7B 扮新同事出题]
 Q --> R{线上检索<br/>能找回源块?}
 R -- 否 --> X1[丢弃]
 R -- 是 --> G[线上同一 prompt 生成回答]
 G --> K{程序硬校验<br/>引用/源块/行号/代码忠实}
 K -- 不过 --> X2[丢弃或人工 fix]
 K -- 过 --> S[(SFT 样本)]
 S --> SP[先切 val 15%]
 SP --> TR[train 248]
 SP --> VA[val 68 不过滤]</pre><figcaption>图 22-2：拒绝采样造数据。过滤器和评测共用同一把尺子，val 必须在过滤之前切出来</figcaption></figure>
<details><summary><span class="eye" aria-hidden="true"></span>展开：命令与实测漏斗</summary>
<pre><code>N=300 SEEDS="1 2" bash scripts/run_pipeline_ubuntu2004.sh data train evallora
# data：     {"auto_pass": 164, "kept": 254, ...}  第二批 --append 后 kept 463
# train：    [data] {"train": 248, "train_too_long": 8, "val": 68}
#            [done] {"train_loss": 0.0962, "seconds": 1222.6, "peak_mem_gb": 18.94}
# evallora： base / lora 两组指标</code></pre>
<table><tr><th>漏斗</th><th>条数</th></tr><tr><td>抽样（2×300）</td><td>600</td></tr><tr><td>检索能找回源块</td><td>463</td></tr><tr><td>自动校验通过</td><td>305</td></tr><tr><td>代码忠实（可训练）</td><td>291 = 回答 246 + 拒答 45</td></tr></table>
<p><strong>人工校对（推荐）</strong>：打开 <code>finetune/data/review.md</code>，把每条的“状态”改为 ok、fix 或 drop。选 fix 时直接改“### 回答”段落。train_qlora 会自动读取这些修改，人工 fix 的样本不受自动过滤影响。</p>
<p><strong>参数从哪里来</strong>：max_len=5120 来自 token 分布的 P97（P50 3423 / P95 4859）。第一次按经验设成 3072，结果丢掉了六成样本。r=16、lr 2e-4、2 epoch 是 7B QLoRA 常用的起点；数据少于 300 条时不要超过 3 epoch，否则会背题。</p></details>
<table><tr><th>val 68 条</th><th>底座</th><th>+LoRA</th><th>上线判断</th></tr>
<tr><td>引用到真正源块</td><td>0.556</td><td>0.683</td><td>✓ 提升</td></tr>
<tr><td>证据不足时拒答（5 条）</td><td>0.0</td><td>0.8</td><td>✓ 提升，但样本少</td></tr>
<tr><td>代码原样摘抄</td><td>0.921</td><td>0.968</td><td>✓</td></tr>
<tr><td>自写行号（越低越好）</td><td>0.0</td><td>0.016</td><td>⚠ 1 例回归，进入下一轮校对</td></tr>
<tr><td>正常题误拒（越低越好）</td><td>0.0</td><td>0.048</td><td>⚠ 3 例，其中 2 例是假前提题，真误拒 1 例</td></tr></table>
<details><summary><span class="eye" aria-hidden="true"></span>展开：评测补强——误拒率、扩充拒答集、离线重算</summary>
<p><strong>以前的问题：</strong>①只看 refusal_ok，看不出模型是不是“逢题就拒”；②拒答 val 只有 5 条，0.8 的意思是 4/5，置信区间很宽；③每改一次打分规则就重跑 11 分钟生成，很浪费。</p>
<table><tr><th>补强</th><th>做法</th><th>命令</th></tr>
<tr><td>误拒率 over_refusal</td><td>正常题回答里出现“没有找到”就记一次误拒，和 refusal_ok 放在一起看</td><td>默认输出</td></tr>
<tr><td>扩充拒答集</td><td>把 val 的 63 道正常题各派生一道拒答题：问题不变，把源文件的<strong>所有块</strong>都从证据里删掉（只删源块不够，同文件的相邻块往往也含答案）。拒答样本从 5 条增加到 59 条（54 条派生；另有 9 题删掉源文件后证据为空，跳过），干扰证据仍然来自真实检索，难度高于随便塞几块无关内容</td><td><code>eval_lora --extra-refusal</code>（约 20 分钟）</td></tr>
<tr><td>只看样本构成</td><td>不加载模型，打印 answer/refusal 条数</td><td><code>eval_lora --extra-refusal --dry-run</code> → <code>{"answer": 63, "refusal": 59}</code></td></tr>
<tr><td>离线重算</td><td>读取已保存的 base/lora 回答，按新规则重新打分并写回日志</td><td><code>eval_lora --rescore logs/eval_lora_X.json</code>（约 1 秒）</td></tr></table>
<p><strong>扩充集实测结果</strong>（<code>logs/eval_lora_20261124_121006.json</code>，122 条，用时 930 s）：</p>
<table><tr><th>指标</th><th>底座</th><th>+LoRA</th><th>验收</th></tr>
<tr><td>refusal_ok（59 条拒答题）</td><td>0.119</td><td>0.746</td><td>✓ 明显优于底座</td></tr>
<tr><td>over_refusal（63 条正常题）</td><td>0.0</td><td>0.048</td><td>✓ ≤ 5%</td></tr>
<tr><td>cites_source</td><td>0.556</td><td>0.683</td><td>✓</td></tr>
<tr><td>code_faithful</td><td>0.921</td><td>0.968</td><td>✓</td></tr></table>
<p><strong>标签噪声：</strong>LoRA 有 14 道派生拒答题没有拒答。逐条检查后发现，其中 13 道的剩余证据里仍然有答案，例如源块在 <code>diagconfig.h</code>，证据里还留着 <code>diagconfig.cpp</code>；源块在说明文档里，证据里还留着对应的代码文件。对这 13 道题来说，模型给出带引用的回答反而是对的。所以 0.746 是<strong>偏低的保守值</strong>，扣除噪声后约为 44/46≈0.96。底座的 0.119 同样受噪声影响，但两边的差距足够大，结论不受影响。<em>套路：自动派生的负样本一定要抽检“失败”的样本。失败样本里最容易混进标签错误，模型在这些题上“答错”，往往是对的。</em></p>
<p><em>套路：评测日志要保存原始回答，不能只存分数。打分规则一定会改，只要回答还在，重算只需要 1 秒。</em>流水线里用 <code>EVAL_ARGS=--extra-refusal bash scripts/run_pipeline_ubuntu2004.sh evallora</code> 透传参数。</p></details>
<details><summary><span class="eye" aria-hidden="true"></span>展开：下一轮迭代清单（按收益排序）</summary><ol>
<li><s><strong>跑扩充拒答集</strong></s> ✓ 已完成（见上方结果）。剩余工作：派生拒答题时，把同名文件（foo.h 和 foo.cpp）以及引用了同一符号的文档也一起删掉，减少标签噪声，然后重跑。</li>
<li><strong>修造题 prompt</strong>：禁止“X 没有被正确更新，会导致什么问题”这种断言式问法。已经造出来的假前提题挑出来人工写“纠正前提”的答案，然后放回训练集。</li>
<li><strong>补人工校对</strong>：review.md 只抽了 30 条：20 条自动校验通过被预填为 ok，10 条 todo 还没人看，人工 fix 为 0。优先处理这 10 条，再补校对 self_lineno 命中的样本和误拒样本，人工 fix 的样本会绕过自动过滤进入训练。</li>
<li><strong>拒答比例</strong>：拒答训练样本目前约占 15%（45/291）。如果扩充集上误拒率升高，先把造拒答的频率从 1/10 降到 1/15，再重训。</li>
<li><strong>形态 C 实测</strong>：本机还没装 vLLM。merge_and_unload 之后，用 vLLM 起服务，并用同一份 val 对比合并前后的指标。合并到 BF16 底座会和 4bit 训练产生精度差，必须重新评测才能上线。</li></ol></details>
<h3>第 9 步：部署上线</h3>
<details open><summary><span class="eye" aria-hidden="true"></span>展开：三种部署形态怎么选</summary><table><tr><th>形态</th><th>命令</th><th>显存</th><th>适用</th><th>本机</th></tr>
<tr><td>A. transformers + LoRA 适配器（默认）</td><td><code>python -m kb.cli --lora finetune/out/lora serve</code></td><td>约 17.6GB（BF16 底座 + 39MB 适配器）</td><td>个人或小团队、单卡、串行请求</td><td>✓ 已验证：ask 4.19 s</td></tr>
<tr><td>B. 4bit 底座 + LoRA</td><td>再加 <code>--load-4bit</code></td><td>约 6GB</td><td>显卡还要跑别的任务（如 TensorRT 实验）</td><td>与训练时同精度</td></tr>
<tr><td>C. 合并权重后交给 vLLM</td><td><code>merge_and_unload()</code> 后 save_pretrained，vLLM 起 OpenAI 接口，再 <code>--api-base http://127.0.0.1:8000/v1</code></td><td>取决于 vLLM 的 gpu_memory_utilization</td><td>多人并发、需要连续批处理</td><td>未在本机实测</td></tr></table>
<p>为什么默认用 A：底座和适配器分开发布，升级只换 39MB 的目录；回滚只需去掉 <code>--lora</code> 参数。业务层只依赖 <code>chat(messages)</code> 一个接口（kb/llm.py），A、B、C 之间切换不需要改问答代码。</p>
<pre><code>bash scripts/run_pipeline_ubuntu2004.sh serve          # 检测到 finetune/out/lora 就自动带上
PORT=8771 bash scripts/run_pipeline_ubuntu2004.sh serve  # 换端口
curl -s -X POST 127.0.0.1:8770/api/ask -H 'Content-Type: application/json' \\
     -d '{"question":"udslib 在 Linux 下如何编译？"}'</code></pre></details>
<details><summary><span class="eye" aria-hidden="true"></span>展开：上线检查清单（安全与运维）</summary><ul>
<li><strong>只监听 127.0.0.1</strong>：源码不出本机。要给同事用，在前面加带鉴权的反向代理（nginx + LDAP/Basic Auth），不要直接改成 0.0.0.0。</li>
<li><strong>GPU 串行</strong>：server 里有一把锁，并发请求排队，超时返回 503。需要并发时换形态 C。</li>
<li><strong>索引更新</strong>：代码变了只需要 <code>ingest index eval</code>（约 1.5 分钟），不用重训。适配器学的是“怎么看资料答题”，不是知识本身。</li>
<li><strong>适配器版本化</strong>：把 <code>out/lora</code> 按日期重命名为 <code>lora_20261123</code>，并和 <code>logs/train_*.json</code>、<code>eval_lora_*.json</code> 一起归档。上线前必须有对应的评测记录。</li>
<li><strong>何时重训</strong>：回答格式出现系统性退化，或新增项目类型（比如加入 PDF 规格书）时。日常代码更新不需要重训。</li>
<li><strong>数据与权重不进 git</strong>：<code>finetune/data/</code>、<code>finetune/out/</code>、<code>index/</code> 已加入 .gitignore，因为里面是源码片段。</li></ul></details>
<details><summary><span class="eye" aria-hidden="true"></span>展开：故障速查表</summary><table><tr><th>现象</th><th>先看哪里</th><th>处理</th></tr>
<tr><td>is_available() False</td><td>nvidia-smi 驱动版本</td><td>换与驱动匹配的轮子（第 20 章）</td></tr>
<tr><td>训练日志 train_too_long 很大</td><td>[data] 那一行</td><td>统计 token 分布，把 --max-len 设到 P95–P97</td></tr>
<tr><td>训练 OOM</td><td>nvidia-smi 有没有残留进程</td><td>停掉 serve；--max-len 降到 4096；保持 bs=1 + grad_accum</td></tr>
<tr><td>loss 很快接近 0 但 val 变差</td><td>epoch 和样本数</td><td>过拟合：减少 epoch、增加数据或换 seed</td></tr>
<tr><td>LoRA 版本拒答过多</td><td>evallora 里正常题的 has_cite</td><td>拒答样本占比太高，从 1/10 降到 1/15</td></tr>
<tr><td>后台日志长时间不更新</td><td>是否被管道缓冲</td><td>python -u 或 print(flush=True)，用 nvidia-smi 看 GPU 是否在跑</td></tr></table></details>
<div class="checks"><strong>自查</strong><ul><li>能在一台新机器上只用两个脚本复现到 serve</li><li>能说出每个阶段的验收标准，以及哪一步是门禁</li><li>能解释为什么 val 要在过滤前切、max_len 怎么定</li><li>能说清 A/B/C 三种部署形态的取舍，以及 LoRA 的回滚方式</li></ul></div>
`},
{id:'local-kb-service',title:'23 · 知识库 Agent 上线：技术架构、模型放哪、为什么不训练也能学会',subtitle:'第 21、22 章把它做成了命令行工具，本章讲它怎么变成局域网里大家都能用的产品：架构分层、模型与数据的存放位置、“学习闭环”为什么不需要重新训练，以及流式输出、停止生成、追问建议这些交互背后的工程做法。',priority:'P0',body:`
<div class="callout"><strong>一句话：知识放在“索引”里，不放在“权重”里。</strong>模型只负责“读资料、组织语言、标出处”，公司代码和文档切成块存进向量索引。新知识进来时，卡片审核通过后会自动增量入库并热切换索引，几分钟内生效，不需要重新训练。QLoRA 只用来教模型“怎么回答”（格式、引用、何时拒答），不负责“记住知识”。</div>

<h3>23.1 技术架构：五层，各管一件事</h3>
<figure class="diagram"><pre class="mermaid">flowchart TB
 subgraph 前端[前端]
  U[chat.html：提问 / 停止 / 追问 / 点赞 / 纠错]
 end
 subgraph 服务[FastAPI server.py :8770]
  S1["/api/ask_stream（SSE）"]
  S2["/api/stop"]
  S3["/api/admin：审核·卡片·入库"]
  L[("GPU 锁：生成串行")]
 end
 subgraph 编排[agent.py]
  A1[检索] --> A2[拼证据 + 提示词] --> A3[生成] --> A4[引用校验 / 补引用] --> A5[追问建议]
 end
 subgraph 检索[retrieve.py]
  R1[BM25 关键词] & R2[bge-m3 向量 FAISS] --> R3[RRF 融合] --> R4[bge-reranker 重排]
 end
 subgraph 存储[磁盘]
  D1[("index/ 12988 块")]
  D2[("data/kb.db 问答·反馈·投票")]
  D3[("知识卡片 Markdown")]
 end
 U --> S1 --> L --> A1
 U --> S2
 A1 --> R1
 A3 --> M["Qwen2.5-Coder-7B + LoRA"]
 R4 --> D1
 S3 --> D2 & D3</pre><figcaption>图 23-1：从浏览器到 GPU 的完整链路（代码在 kb-agent/kb/）</figcaption></figure>
<table><tr><th>层</th><th>文件</th><th>职责</th><th>关键设计</th></tr>
<tr><td>前端</td><td><code>chat.html</code> / <code>admin.html</code></td><td>问答、历史、榜单；后台审核与开关</td><td>单文件、无构建；fetch + ReadableStream 解析 SSE；深浅主题用 CSS 变量</td></tr>
<tr><td>服务</td><td><code>server.py</code></td><td>HTTP 接口、鉴权、队列</td><td>一把 GPU 锁让生成串行，避免并发把显存打爆；排队状态作为事件推给前端</td></tr>
<tr><td>编排</td><td><code>agent.py</code></td><td>检索 → 生成 → 校验</td><td>提示词是“软约束”，引用校验是“硬约束”：没引用就打回重写一次</td></tr>
<tr><td>检索</td><td><code>retrieve.py</code></td><td>找证据</td><td>关键词 + 向量双路召回，RRF 融合，交叉编码器重排；问题里点名项目时只搜该项目</td></tr>
<tr><td>模型</td><td><code>llm.py</code></td><td>生成文本</td><td>本地 transformers 或 OpenAI 兼容接口（vLLM）二选一，上层代码不改</td></tr>
<tr><td>存储</td><td><code>store.py</code> / <code>cards.py</code></td><td>问答库、反馈、卡片</td><td>SQLite WAL；旧库启动时用 PRAGMA table_info 检查并补列，不需要迁移脚本</td></tr></table>

<h3>23.2 模型放哪：三个模型 + 一个适配器</h3>
<table><tr><th>组件</th><th>位置</th><th>磁盘</th><th>作用</th><th>换掉它会怎样</th></tr>
<tr><td>生成模型 Qwen2.5-Coder-7B-Instruct</td><td><code>~/models/Qwen2.5-Coder-7B-Instruct</code></td><td>15 GB（bf16）</td><td>读证据、写答案</td><td>换更大的模型，答案更通顺；检索不变，出处照样准</td></tr>
<tr><td>向量模型 bge-m3</td><td><code>~/models/bge-m3</code></td><td>2.2 GB</td><td>把问题和代码块变成向量</td><td>换了必须重建 index/（向量空间不同）</td></tr>
<tr><td>重排模型 bge-reranker-v2-m3</td><td><code>~/models/bge-reranker-v2-m3</code></td><td>2.2 GB</td><td>对前 30 个候选逐对打分</td><td>不存在时自动跳过，召回率会下降</td></tr>
<tr><td>QLoRA 适配器</td><td><code>kb-agent/finetune/out/lora</code></td><td>39 MB</td><td>教会“按格式答、带 [n]、不会就说”</td><td>去掉 --lora 就回到原模型，等于一键回滚</td></tr>
<tr><td>向量索引</td><td><code>kb-agent/index/</code></td><td>68 MB</td><td>12988 个代码/文档块（含代码概要块）+ FAISS</td><td>知识本体；可随时增量重建</td></tr></table>
<p>模型根目录可以用环境变量 <code>KB_MODELS</code> 改。服务运行时整卡占用约 <strong>19 GB</strong>（3090 Ti 24 GB），主要是 7B 的 bf16 权重加 KV cache。显存紧张时加 <code>--load-4bit</code>，生成模型可降到约 5.5 GB。<em>套路：模型（大、几乎不变）和索引（小、天天变）分开存放，换模型和更新知识互不影响。</em></p>

<h3>23.3 为什么“不用训练”也能学会新知识</h3>
<details open><summary><span class="eye" aria-hidden="true"></span>先讲之前的想法：把知识“训进”模型</summary>
<p>直觉是：拿公司代码做微调，模型就“懂”了。实际做下来（第 21、22 章）有三个问题：</p>
<ul><li><strong>慢</strong>：一轮 QLoRA 要几十分钟到几小时，代码每天在变，训练永远追不上。</li>
<li><strong>记不准</strong>：7B 模型对函数名、宏的值、行号这类“精确事实”记忆很差，训完反而更自信地编造。</li>
<li><strong>无法溯源和删除</strong>：知识混在权重里，说不清答案来自哪个文件；某段代码作废了，也没法从权重里“删掉”。</li></ul></details>
<p><strong>现在的方案</strong>：模型像一个“开卷考试的考生”。每次提问先检索出相关的 8 段资料塞进提示词，模型只照着资料答，并标注 [编号]。知识更新就等于“换一本参考书”：</p>
<table><tr><th>想让它学会</th><th>怎么做</th><th>多久生效</th><th>要不要 GPU 训练</th></tr>
<tr><td>新代码、新文档</td><td>后台点“重新入库”（增量切块 + 向量化，热切换索引，不停服务）</td><td>分钟级</td><td>否</td></tr>
<tr><td>答错或答不出的问题</td><td>点踩 / 答“没有找到”自动进审核队列 → 管理员填正确答案和出处 → 生成知识卡片 → 自动增量入库</td><td>审核通过后分钟级</td><td>否</td></tr>
<tr><td>同一个问题的不同说法</td><td>卡片的“其他问法”字段，入库时一起向量化</td><td>随卡片入库</td><td>否</td></tr>
<tr><td>业务说法 ↔ 代码名称（“校验和”↔ CRC32）</td><td>提示词规则 + 代码概要块（离线为每个文件生成“可能的问法”）</td><td>提示词即时；概要随入库</td><td>否</td></tr>
<tr><td>回答格式、引用习惯、拒答分寸</td><td>QLoRA（唯一需要训练的部分）</td><td>小时级，低频</td><td>是</td></tr></table>
<p>防止“越学越差”的门禁：<code>kb.cli eval --gate</code> 用带标准出处的题（当前 46 道，含卡片自动生成的回归题）和 <code>logs/eval_baseline.json</code> 比。只要有一道老题从命中变成 MISS，或者 recall@8 下降超过 0.02，就返回退出码 2，新卡片不能上线。当前 recall@8 = 1.0，recall@3 = 0.935（只报告，不作为门禁）。为什么门禁用 recall@8 而不是 recall@3：送进模型的就是前 8 段，排第 2 还是第 5 对答案影响不大；recall@3 容易因为重排的小波动误报。<em>套路：可以学得快，但上线前必须过考试。</em></p>
<details><summary><span class="eye" aria-hidden="true"></span>展开：RAG 与微调的分工（一张表记住）</summary>
<table><tr><th></th><th>RAG（检索增强）</th><th>微调（LoRA）</th></tr>
<tr><td>解决什么</td><td>知道什么（事实、代码、出处）</td><td>怎么说（格式、风格、拒答）</td></tr>
<tr><td>更新成本</td><td>重新入库，分钟级</td><td>重新训练，小时级</td></tr>
<tr><td>可溯源</td><td>每句带 [n] → 文件:行号</td><td>不可溯源</td></tr>
<tr><td>删除知识</td><td>删文件重建索引</td><td>几乎做不到</td></tr>
<tr><td>失败表现</td><td>检索不到 → 明说“资料中没有找到”</td><td>一本正经地编造</td></tr></table></details>

<h3>23.4 交互功能背后的工程做法</h3>
<table><tr><th>功能</th><th>之前的问题</th><th>现在的做法</th><th>实测</th></tr>
<tr><td>流式输出</td><td>整段生成完才返回，用户要盯着空白等十几秒</td><td><code>generate</code> 放在子线程，用 <code>TextIteratorStreamer</code> 边生成边推；服务端用 SSE，事件类型有 run / status / token / followups / done / error（早期还有 reset，用来“补引用重写”；改成程序直接补 [n] 之后就删了，见 24.8）</td><td>首字 1.4 s，进度提示 0.35 s 出现</td></tr>
<tr><td>停止生成</td><td>前端断开连接，GPU 还在继续生成没人看的答案</td><td><code>StoppingCriteria</code> 每解码一个 token 检查一次 <code>threading.Event</code>；<code>/api/stop</code> 按 run id 停止，只能停自己的；关页面也会触发停止</td><td>点击后 0.3 s 停下，已生成部分照常入库</td></tr>
<tr><td>追问建议</td><td>用户不知道下一步还能问什么</td><td>优先取被引用文件“代码概要”里离线预测的问法（不额外占 GPU），不足 2 条才让模型补；用二元组相似度去掉重复</td><td>一般 0 次额外生成</td></tr>
<tr><td>补引用</td><td>小模型经常忘了标 [n]</td><td>最初的做法：把原答案打回让模型重写一次，前端收到 reset 事件后清空重新显示。<strong>后来改成程序补</strong>（见 24.8）</td><td>—</td></tr>
<tr><td>深浅主题代码块</td><td>浅色页面里代码块仍是深色</td><td>代码块颜色全部改用 CSS 变量，并在 <code>:root[data-theme=light]</code> 下覆盖 GitHub Light 高亮色</td><td>—</td></tr></table>
<details><summary><span class="eye" aria-hidden="true"></span>展开：踩过的坑</summary><ul>
<li><strong>no_grad 是线程局部的</strong>：主线程开了 <code>torch.no_grad()</code>，子线程里的 generate 并不受影响，显存会悄悄上涨。必须在子线程里再开一次。</li>
<li><strong>线程里的异常不会传到 HTTP 响应</strong>：工作线程必须 try/except，把异常转成 error 事件放进队列，否则前端只会一直转圈。</li>
<li><strong>提示词里别举“像真的一样”的例子</strong>：写了示例函数名 <code>crc32()</code>，7B 模型把它当成事实抄进了答案。</li>
<li><strong>近因效应</strong>：证据有上千 token 时，模型会忘了 system 里的格式要求，所以在问题后面再重复一遍关键约束（ANSWER_TAIL）。</li></ul></details>

<h3>23.5 下一步可以做什么</h3>
<ul><li><strong>换成 vLLM 部署</strong>：<code>--api-base</code> 已经支持 OpenAI 兼容接口。vLLM 的连续批处理和 PagedAttention 可以去掉“GPU 串行锁”，多人同时提问也不用排队（参见本书推理章节）。</li>
<li><strong>追问更贴题</strong>：目前按“被引用文件”取候选，有时会偏离原问题；可以改成按与原问题的相似度给候选排序。</li>
<li><strong>量化对比</strong>：AWQ / GPTQ 4bit 与 bf16 在同一套 45 题上对比 recall 和答案质量，用数据决定要不要换。</li></ul>
<div class="checks"><strong>自查</strong><ul><li>能画出五层架构，并说出每层由哪个文件负责</li><li>能说出三个模型各放在哪里、多大，以及换掉向量模型为什么必须重建索引</li><li>能解释“知识在索引、风格在 LoRA”，以及新知识从点踩到生效的完整路径</li><li>能讲清停止生成为什么要在 StoppingCriteria 里检查，而不是只断开连接</li></ul></div>
`},
{id:'local-vllm-bench',title:'24 · 换上 vLLM：部署、并发与端到端评测',subtitle:'第 23 章的服务用 transformers 直接生成，一次只能答一个人。本章把生成引擎换成 vLLM（带 LoRA），去掉串行锁改成并发槽，并用同一套 45 题把“答得对不对、多快、能扛几个人”一次测清楚。',priority:'P0',body:`
<div class="callout"><strong>一句话：引擎换了，业务代码只改了一个参数。</strong>kb-agent 通过 OpenAI 兼容接口调用 vLLM，单人输出速度提升约 28%，8 人并发时总吞吐是原来的 <strong>5.7 倍</strong>，引用命中率没有下降（0.822 → 0.867）。代价是并发越高首字越慢，因为 KV cache 只够约 2.8 路满长请求。</div>

<h3>24.1 先讲之前的做法和问题</h3>
<p>第 23 章用 <code>TransformersLLM</code> 在进程内加载模型，配一把 GPU 锁让生成串行。问题有两个：</p>
<ul><li><strong>不能并发</strong>：<code>model.generate</code> 一次只处理一个请求。第二个人进来只能排队，3 个人同时问，最后一个要等 30 秒以上。</li>
<li><strong>显存浪费</strong>：HF 的 KV cache 按最大长度连续分配。vLLM 的 PagedAttention 按页（block）分配，多个请求共享显存池，同样的显存能同时服务更多请求。</li></ul>
<p>vLLM 的<strong>连续批处理</strong>（continuous batching）可以理解成“公交车不等满员”：每解码一步都可以有新请求上车、已答完的请求下车，GPU 一直在做批量矩阵乘，而不是给一个人单独开一趟车。</p>

<h3>24.2 安装：单独的 venv</h3>
<pre><code class="language-bash">python3 -m venv ~/venvs/vllm
~/venvs/vllm/bin/python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ vllm==0.10.1.1
# 坑：pip 会顺带装上最新的 transformers 5.x，启动时报
#   AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
~/venvs/vllm/bin/python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ "transformers>=4.55,<4.57"</code></pre>
<p><em>套路：推理引擎单独一个 venv，和训练环境（peft / bitsandbytes）分开。vLLM 会锁定 torch 和 CUDA 版本，混在一起迟早互相升级出问题。用 <code>python -m pip</code> 确保包装进了你要用的那个解释器。</em></p>

<h3>24.3 启动命令，逐个参数解释</h3>
<pre><code class="language-bash">~/venvs/vllm/bin/vllm serve ~/models/Qwen2.5-Coder-7B-Instruct \\
  --served-model-name qwen2.5-coder-7b --host 127.0.0.1 --port 8001 \\
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.72 \\
  --enable-lora --max-lora-rank 16 \\
  --lora-modules kb=kb-agent/finetune/out/lora \\
  --enable-prefix-caching

# kb-agent 改用 vLLM：模型名写 LoRA 的名字 kb，就会带上适配器
/home/sti/venvs/gpu/bin/python -m kb.cli --api-base http://127.0.0.1:8001/v1 --api-model kb \\
  serve --host 0.0.0.0 --port 8770</code></pre>
<table><tr><th>参数</th><th>取值</th><th>为什么</th></tr>
<tr><td><code>--port</code></td><td>8001</td><td>8000 已被别的服务占用，启动时会报地址被占用。先用 <code>lsof -i:8000</code> 查一下</td></tr>
<tr><td><code>--max-model-len</code></td><td>8192</td><td>RAG 提示词（8 段证据）加上回答一般在 3–5k token；原生 32k 太长，会挤占 KV cache</td></tr>
<tr><td><code>--gpu-memory-utilization</code></td><td>0.72</td><td>vLLM 会把这个比例的显存<strong>全部预占</strong>。同一张卡上 kb-agent 还要加载 bge-m3 和 reranker（约 4.5 GB），设成默认 0.9 的话 kb-agent 会 OOM</td></tr>
<tr><td><code>--enable-lora</code> / <code>--lora-modules kb=...</code></td><td>rank 16</td><td>底座只加载一份；请求里 model 写 <code>kb</code> 走适配器，写 <code>qwen2.5-coder-7b</code> 走原模型，A/B 对比和回滚都不用重启</td></tr>
<tr><td><code>--max-lora-rank</code></td><td>16</td><td>必须 ≥ 训练时的 r，否则加载适配器时报错</td></tr>
<tr><td><code>--enable-prefix-caching</code></td><td>开</td><td>所有请求的 system 提示词相同，前缀的 KV 只需算一次，可以降低首字延迟</td></tr></table>

<h3>24.4 读懂启动日志：显存都去哪了</h3>
<table><tr><th>日志关键字</th><th>实测</th><th>含义</th></tr>
<tr><td>Model loading took</td><td>14.32 GiB / 5.06 s</td><td>7B bf16 权重（7.6B × 2 字节）</td></tr>
<tr><td>Available KV cache memory</td><td>1.21 GiB</td><td>24 GB × 0.72 − 权重 − 激活和 CUDA graph 预留</td></tr>
<tr><td>GPU KV cache size</td><td>22,736 tokens</td><td>Qwen2.5-7B 用 GQA，每 token 的 KV 约 56 KB（28 层 × 4 个 KV 头 × 128 维 × 2(K,V) × 2 字节）</td></tr>
<tr><td>Maximum concurrency for 8,192 tokens</td><td>2.78x</td><td>22736 / 8192：如果每个请求都跑满 8k，只能同时跑 2.78 个</td></tr>
<tr><td>Graph capturing finished</td><td>15 s</td><td>为常见 batch 大小预录 CUDA graph，减少 decode 阶段的 kernel 启动开销</td></tr>
<tr><td>init engine took</td><td>50 s</td><td>冷启动总时间；只有 vLLM 时整卡占用 18.5 GB，加上 kb-agent 约 23 GB</td></tr></table>
<p><em>经验：调 <code>--gpu-memory-utilization</code> 时盯住“KV cache tokens”这一行。它除以平均请求长度（不是 max-model-len），才是真正能同时服务的人数。RAG 请求平均约 4k token，所以实际能并发 5 路左右，超过就要排队等 KV 页。</em></p>

<h3>24.5 架构变化：串行锁 → 并发槽</h3>
<figure class="diagram"><pre class="mermaid">flowchart LR
 U1[用户 A] & U2[用户 B] & U3[用户 C] --> S["server.py 并发槽 BoundedSemaphore(8)"]
 S --> AG[agent.py 检索 + 拼提示词]
 AG -->|"HTTP 流式 model=kb"| V["vLLM :8001 连续批处理"]
 V --> W["Qwen2.5-Coder-7B 底座"]
 V -.->|"按请求挂载"| L["LoRA 适配器 kb"]
 ADM[后台：重新入库] -->|"exclusive 拿满 8 个槽"| S</pre><figcaption>图 24-1：kb-agent 不再自己生成，而是把请求流式转发给 vLLM</figcaption></figure>
<pre><code class="language-python"># server.py：并发上限由“引擎能力”决定，业务代码只管 acquire/release
n_slots = args.concurrency or (8 if args.api_base else 1)
gpu = threading.BoundedSemaphore(n_slots)
excl = threading.Lock()

@contextmanager
def exclusive():
    """换索引 / 向量化要独占：一次拿满所有槽，等在途问答结束，同时挡住新问答"""
    with excl:                      # 防止两个独占者各拿一半，互相死锁
        for _ in range(n_slots):
            gpu.acquire()
    try:
        yield
    finally:
        for _ in range(n_slots):
            gpu.release()</code></pre>
<details><summary><span class="eye" aria-hidden="true"></span>展开：为什么 8 个槽，而不是无限</summary>
<p>vLLM 自己会排队，但 kb-agent 这边每个请求还要做检索（bge-m3 + reranker，也在 GPU 上），并且占一个线程。槽数的作用是<strong>背压</strong>：超过 8 个就在入口排队，并推送“排队中”事件给前端，避免请求全部堆进 vLLM 后首字延迟一起变差。从 bench 看，c8 的 TTFT p90 已经接近 5 s，再往上加槽，用户体感会变差。</p></details>
<p>用量统计也要改：多个请求在不同线程里同时调用同一个 <code>OpenAICompatLLM</code> 对象，<code>self.last_usage</code> 会互相覆盖，所以改成 <code>threading.local()</code>。<em>套路：一个对象从“单线程独占”变成“多线程共享”时，先把所有 self.xxx 可变状态过一遍。</em></p>

<h3>24.6 流式与停止：远端引擎的“停止”就是断开连接</h3>
<p>transformers 后端靠 <code>StoppingCriteria</code> 停止（第 23 章）。vLLM 在另一个进程里，没法往里塞回调，但它有个约定：<strong>客户端断开连接，就 abort 该请求，KV 页立即回收</strong>。所以 <code>OpenAICompatLLM</code> 总是用 <code>stream=True</code>，每读一行检查一次 <code>stop_event</code>，置位就 break，<code>with urlopen</code> 退出时关闭连接。</p>
<pre><code class="language-python">body = {..., "stream": True, "stream_options": {"include_usage": True}}  # 最后一帧带 token 用量
with urllib.request.urlopen(req, timeout=self.timeout) as resp:
    for line in resp:
        if stop_event is not None and stop_event.is_set():
            break                    # 关连接 = vLLM abort
        ...</code></pre>
<details open><summary><span class="eye" aria-hidden="true"></span>大坑：同步 SSE 生成器的 finally 永远不执行</summary>
<p>现象：浏览器关掉页面后，vLLM 日志里的生成吞吐一直不为 0，直到跑满 max_tokens。</p>
<p>排查过程：</p>
<ol><li>先怀疑 urllib：单独写脚本 <code>r.close()</code>，vLLM 立即停止，说明 urllib 没问题。</li>
<li>再看服务端：SSE 用的是同步生成器，停止信号写在 <code>finally: ev.set()</code> 里。</li>
<li>根因：Starlette 把<strong>同步生成器</strong>放进线程池迭代。客户端断开时只取消外层协程，线程池里的生成器既不会被 close，也不会走 finally，于是 <code>ev</code> 永远不会被置位。</li></ol>
<pre><code class="language-python">async def events():                       # 必须是 async 生成器
    loop = asyncio.get_running_loop()
    try:
        while (e := await loop.run_in_executor(None, q.get)) is not None:
            yield f"data: {json.dumps(e, ensure_ascii=False)}\\n\\n"
    finally:                              # 断开时在 await 处收到 CancelledError，这里立即执行
        ev.set()</code></pre>
<p>验证：断开后的下一个 10 秒统计窗口里，<code>Avg generation throughput: 0.0 tokens/s</code>。</p></details>
<p><em>经验：不要用 <code>/metrics</code> 的 <code>num_requests_running</code> 判断请求是否被中止。实测它在 abort 后会一直显示 1 个“幽灵”请求。以日志里的生成吞吐降到 0 为准。</em></p>

<h3>24.7 端到端评测：同一套题，质量和性能一起测</h3>
<p>第 23 章的 <code>eval</code> 只回答“证据找没找到”。新增 <code>bench</code> 子命令，每题记录首字延迟 TTFT、总耗时、输出 token、<strong>引用命中</strong>（答案里 [n] 指向的文件是否包含标准出处）和是否拒答，结果写入 <code>logs/bench_*.json</code>。</p>
<pre><code class="language-bash"># 基线：transformers + LoRA，只能串行
/home/sti/venvs/gpu/bin/python -m kb.cli --lora finetune/out/lora bench --concurrency 1
# vLLM：同一套题，并发 1 / 4 / 8
/home/sti/venvs/gpu/bin/python -m kb.cli --api-base http://127.0.0.1:8001/v1 --api-model kb bench --concurrency 4</code></pre>
<table><tr><th>配置</th><th>引用命中</th><th>有引用</th><th>拒答</th><th>TTFT p50 / p90</th><th>单题耗时 p50 / p90</th><th>45 题总时长</th><th>QPS</th><th>输出 tok/s</th></tr>
<tr><td>transformers + LoRA，c1</td><td>0.822</td><td>0.889</td><td>0.067</td><td>0.87 / 1.10 s</td><td>8.76 / 24.36 s</td><td>549.7 s</td><td>0.082</td><td>25.3</td></tr>
<tr><td>vLLM + LoRA，c1</td><td>0.844</td><td>0.911</td><td>0.067</td><td>0.73 / 0.94 s</td><td>7.85 / 19.46 s</td><td>436.2 s</td><td>0.103</td><td>32.5</td></tr>
<tr><td>vLLM + LoRA，c4</td><td>0.867</td><td>0.956</td><td>0.044</td><td>0.98 / 1.74 s</td><td>10.36 / 22.54 s</td><td>135.4 s</td><td>0.332</td><td>106.4</td></tr>
<tr><td>vLLM + LoRA，c8</td><td><strong>0.867</strong></td><td>0.933</td><td>0.067</td><td>1.86 / 4.97 s</td><td>14.75 / 28.19 s</td><td><strong>96.7 s</strong></td><td><strong>0.466</strong></td><td><strong>144.6</strong></td></tr></table>
<p>怎么读这张表：</p>
<ul><li><strong>单人也快了</strong>：c1 对 c1，输出速度 25.3 → 32.5 tok/s（+28%）。原因是 CUDA graph 减少了 kernel 启动开销，以及 vLLM 优化过的 attention kernel。</li>
<li><strong>吞吐几乎随并发线性增长</strong>：c4 是 c1 的 3.3 倍，c8 是 4.5 倍。decode 阶段受显存带宽限制，一次读权重算 8 个 token 和算 1 个 token 耗时差不多，这就是批处理的收益来源。</li>
<li><strong>代价是单个请求变慢</strong>：c8 的 TTFT p90 是 4.97 s。新请求的 prefill 要插进正在 decode 的批次里，而且 KV cache 只够约 2.8 路满长请求，超出的要等 KV 页。</li>
<li><strong>质量没退化</strong>：引用命中 0.822 → 0.867。贪心解码下 vLLM 和 HF 的数值实现略有不同，几道边界题答案有变化，属于正常波动，不是变聪明了。<em>只要质量没下降就可以换。</em></li></ul>
<details><summary><span class="eye" aria-hidden="true"></span>展开：bench 的几个设计取舍</summary><ul>
<li><strong>质量和性能必须同一次跑</strong>：只看速度会选出“快但乱答”的配置（比如把 max_tokens 调小）。</li>
<li><strong>看分位数，不看平均值</strong>：p90 反映“最倒霉的那 10% 用户”的体验；平均值会被大量短答案拉低。</li>
<li><strong>transformers 后端禁止 c&gt;1</strong>：它本来就不支持并发，强行并发只会测出锁等待时间，直接报错退出更诚实。</li>
<li><strong>TTFT 取第一个 token 事件的时间</strong>：包含检索和重排（约 0.3–0.5 s），是用户真实感受到的首字时间，不只是引擎内部的 prefill 时间。</li></ul></details>

<h3>24.8 “回答总生成两遍”：用程序补引用代替模型重写</h3>
<p><strong>现象</strong>：上线 vLLM 后，用户反馈答案经常先完整出现一遍，被清空后又从头出现一遍，等待时间也翻倍。</p>
<p><strong>排查</strong>：抓一次 SSE，事件里有一个 <code>reset</code>。第一遍答案里一个 [n] 都没有，触发了第 23 章的“补引用重写”。用最近 10 个真实问题复测，<strong>5 个</strong>首遍缺引用，其中长列表、多步骤的答案最容易漏标。bench 的引用率看起来正常，是因为它统计的是重写之后的结果，掩盖了这个问题。</p>
<p><strong>修法</strong>：不再让模型重写，改由程序补。逐行计算这一行和每段资料的二元组（相邻两个字符）重合度，取最高的那段，重合度 ≥ 0.5 就在句末标点前插入 [n]。代码块、表格行、标题不动，因为在这些行尾加字会破坏 Markdown 结构。</p>
<pre><code class="language-python">def attach_citations(answer, chunks, min_overlap=0.5):
    grams = [bigrams(c.text[:2400]) for c in chunks]
    for line in answer.split("\\n"):          # 跳过 \`\`\` 代码块、| 表格、# 标题、已有 [n] 的行
        g = bigrams(line)
        score, n = max((len(g & cg) / len(g), i) for i, cg in enumerate(grams, 1))
        if score >= min_overlap: ...          # 在句末标点前插入 [n]</code></pre>
<table><tr><th></th><th>之前：模型重写</th><th>现在：程序补</th></tr>
<tr><td>同一问题耗时</td><td>约 20 s（生成两遍）</td><td>约 10 s</td></tr>
<tr><td>GPU 开销</td><td>多一次完整生成</td><td>0（几毫秒的集合运算）</td></tr>
<tr><td>用户观感</td><td>答案清空重来</td><td>一次出完，done 时引用编号出现</td></tr>
<tr><td>风险</td><td>重写时可能改动内容</td><td>纯改写、没有字面重合的句子补不上；阈值取 0.5 宁缺毋滥</td></tr></table>
<p><em>套路：能用确定性程序做的后处理，就不要再调一次模型。小模型的回答大多是“摘抄 + 改写”，字面重合度足以定位出处。评测指标要统计“首遍”的表现，否则兜底逻辑会把问题藏起来。</em></p>

<h3>24.9 排障速查</h3>
<table><tr><th>现象</th><th>先查</th><th>处理</th></tr>
<tr><td>vLLM 启动报 <code>all_special_tokens_extended</code></td><td><code>pip show transformers</code></td><td>降到 4.55–4.56</td></tr>
<tr><td>kb-agent 加载 bge-m3 时 OOM</td><td><code>nvidia-smi</code> 看 vLLM 占了多少</td><td>降低 <code>--gpu-memory-utilization</code>，至少给 kb-agent 留 5 GB</td></tr>
<tr><td>请求 model=kb 返回 404</td><td><code>curl :8001/v1/models</code></td><td>没开 <code>--enable-lora</code>，或 lora-modules 路径不对</td></tr>
<tr><td>关页面后 GPU 仍在生成</td><td>vLLM 日志的 generation throughput</td><td>SSE 生成器必须是 async（见 24.6）</td></tr>
<tr><td>并发高时首字很慢</td><td>日志里的 Running / Waiting 请求数和 KV cache usage</td><td>减少并发槽、缩短 max-model-len，或者用量化腾出 KV 空间（下一章）</td></tr></table>
<div class="checks"><strong>自查</strong><ul><li>能解释连续批处理为什么能提高吞吐，以及代价落在哪个指标上</li><li>能从启动日志算出 KV cache 能同时容纳几个请求，并说出 gpu-memory-utilization 为什么设成 0.72</li><li>能说清并发槽和 exclusive() 分别解决什么问题</li><li>能讲出同步 SSE 生成器 finally 不执行的根因和修法</li><li>能读懂 bench 表：哪个配置适合 1 人用，哪个适合 10 人用</li></ul></div>
`},
{id:'local-nightly',title:'25 · 夜间流水线：让显卡下班后把经验沉淀下来',subtitle:'白天显卡给用户答题，晚上 21:30 之后空着。本章用 cron + 线上服务的管理接口，搭一条每晚自动跑的流水线：增量入库、代码概要、端到端自测、问答体检、合成训练草稿，早上看一份报告就知道系统昨晚变好还是变坏。',priority:'P0',body:`
<div class="callout"><strong>一句话：先把“测量”和“沉淀”自动化，再谈“训练”。</strong>RAG 系统变聪明主要靠三层：知识库（知道什么）、检索模型（找不找得到）、LLM（怎么说）。每晚自动做的是零风险的前两类事情，外加给第三类攒数据。训练必须等人审过的数据和评测卡口都就绪后再做。</div>

<h3>25.1 先讲之前的做法和问题</h3>
<ul><li><strong>入库靠手点</strong>：代码改了，要有人去后台点“重新入库”。忘了点，回答就停留在旧版本。</li>
<li><strong>代码概要只做了一部分</strong>：概要（文件讲什么 + 预测问法）对检索和追问建议帮助很大，但要用 GPU 逐文件生成。新加的 5 个项目还有 <strong>1019 个文件</strong>没做，白天跑又会和用户抢卡。</li>
<li><strong>质量靠感觉</strong>：bench 要手动跑，没有按天的趋势。改完东西说不清是变好还是变坏。</li>
<li><strong>坏例子散在数据库里</strong>：点踩、没找到、无引用的问题都在 kb.db 里，但没人每天去翻。</li></ul>

<h3>25.2 整体设计：飞轮</h3>
<figure class="diagram"><pre class="mermaid">flowchart LR
  A["白天问答<br/>日志 / 点赞 / 点踩 / 没找到"] -->|"沉淀"| B["kb.db + 审核队列"]
  B -->|"21:30 cron"| C["nightly.py"]
  C -->|"rebuild"| D["增量入库 + 检索卡口"]
  C -->|"summarize"| E["代码概要<br/>07:30 前自动停"]
  C -->|"selftest"| F["端到端自测"]
  C -->|"make_data"| G["合成草稿<br/>只追加、等人审"]
  D --> H["次日线上服务"]
  E --> D
  F -->|"报告"| R["logs/nightly/日期.md"]
  B -->|"体检"| R
  G -.->|"人审通过后"| T["每周训练<br/>LoRA / 检索模型"]
  T -.->|"过卡口才上线"| H
  H --> A</pre></figure>
<p>三个关键决定：</p>
<ol><li><strong>用 GPU 的步骤都走线上服务的 HTTP 管理接口</strong>，不另起 GPU 进程。vLLM 已经占了 72% 显存，kb-agent 进程里还有 bge-m3 和 reranker，再加载一份 24G 的卡放不下。而且自测测的就是线上真正在跑的那套。唯一的例外是合成草稿：它用子进程跑 <code>make_data</code>，生成直接调用 vLLM，向量模型放在 CPU 上（<code>--embed-device cpu</code>），所以不额外占显存。</li>
<li><strong>夜里只“造”不“用”</strong>：合成数据只追加到草稿文件，不进训练集，也不进卡口题库。没人审过的数据如果直接拿来训练，模型的错误会被它自己一轮轮放大。</li>
<li><strong>到点必须收工</strong>：概要生成到 07:30 自动请求停止。已生成的按文件落盘，下一晚接着做。</li></ol>

<h3>25.3 每一步做什么</h3>
<table><tr><th>步骤</th><th>接口 / 命令</th><th>GPU 用法</th><th>失败的影响</th></tr>
<tr><td>增量入库</td><td><code>POST /api/admin/rebuild</code></td><td>bge-m3 只编码改动的块（按内容 hash 复用）</td><td>没过检索卡口就不替换，线上不受影响</td></tr>
<tr><td>代码概要</td><td><code>POST /api/admin/summarize</code></td><td>vLLM 逐文件生成，每个文件占一个并发槽</td><td>到点停；做完自动再入库一次</td></tr>
<tr><td>端到端自测</td><td><code>POST /api/admin/selftest</code>（新增）</td><td>线上模型并发 4 跑题库</td><td>报告里写失败原因，不影响其他步骤</td></tr>
<tr><td>问答体检</td><td>只读打开 <code>kb.db</code></td><td>无</td><td>—</td></tr>
<tr><td>合成草稿</td><td><code>finetune.make_data --api-base ... --embed-device cpu</code></td><td>出题、作答走 vLLM；检索的问题编码放 CPU</td><td>只追加 drafts.jsonl，不动人工校对清单</td></tr></table>
<details open><summary><span class="eye" aria-hidden="true"></span>selftest：评测不写问答库</summary>
<p>直接调用 <code>/api/ask</code> 也能测，但每题都会写进 kb.db，热门榜和统计会被测试题污染。所以新增了管理接口：复用 <code>bench_rows()</code>，每题占一个并发槽和真实用户公平排队，结果只返回、不入库。</p>
<pre><code class="language-python">@app.post("/api/admin/selftest", dependencies=[Depends(require_admin)])
def selftest(limit: int | None = None, concurrency: int = 4):
    class Slotted:                       # 每题占一个并发槽
        def ask(self, *a, **kw):
            with gpu:
                return agent.ask(*a, **kw)
    rows = load_questions(config=args.config)[:limit]
    summary, items = bench_rows(Slotted(), rows, max(1, min(concurrency, n_slots)), log=lambda s: None)
    return {"summary": summary, "items": items}</code></pre>
<p><em>套路：把命令行里的评测逻辑抽成纯函数（bench_rows / bench_summary），CLI 和服务端各包一层。同一套打分逻辑，离线对比和线上自测的数字才可比。</em></p></details>
<details><summary><span class="eye" aria-hidden="true"></span>新指标：首遍引用率 first_cited</summary>
<p>第 24.8 节把“漏标引用”改成程序补之后，<code>cited_any</code> 几乎总是 1，已经看不出模型本身守不守规矩。新增 <code>first_cited</code>：trace 里没有 <code>attach_cite</code> 步骤，说明模型自己写了 [n]。这个指标是以后 LoRA 微调要拉高的目标。<em>经验：兜底逻辑上线后，要同时统计“兜底前”的指标，否则问题被藏起来，也就没有优化方向了。</em></p></details>

<h3>25.4 调度：cron + flock + 健康检查</h3>
<pre><code class="language-bash"># crontab -e
30 21 * * * /home/sti/Documents/Trae-CN/default/llm-basics/kb-agent/scripts/nightly_ubuntu2004.sh

# nightly_ubuntu2004.sh 的关键三行
exec 9>"$LOG_DIR/.lock"; flock -n 9 || exit 0          # 上一次没跑完就跳过，不叠加
curl -sf -m 5 http://127.0.0.1:8770/api/config || exit 1  # 服务不在线只记日志
/home/sti/venvs/gpu/bin/python scripts/nightly.py &gt;&gt;"$LOG" 2&gt;&amp;1</code></pre>
<ul><li><strong>cron 的环境很干净</strong>：没有你的 PATH、没有激活的 venv。所以脚本里一律写绝对路径的 Python。</li>
<li><strong>flock 比 PID 文件可靠</strong>：进程崩溃后锁会被内核自动释放，不会留下“僵尸锁”。</li>
<li><strong>手动补跑</strong>：<code>scripts/nightly_ubuntu2004.sh --skip summarize --synth 0</code> 只做入库和自测，几分钟就能出报告。</li></ul>

<h3>25.5 合成草稿：和线上服务共用一张卡</h3>
<p>第 21 章的 <code>make_data</code> 原来自己加载 transformers 模型和 bge-m3，必须独占显卡。改了两个参数后，它可以在线上服务运行时一起跑：</p>
<pre><code class="language-bash">python -m finetune.make_data --n 100 --seed 20261124 --append --review 0 \\
    --api-base http://127.0.0.1:8001/v1 --api-model qwen2.5-coder-7b --embed-device cpu</code></pre>
<ul><li><code>--api-base</code>：出题和作答都发给 vLLM。vLLM 本来就在做连续批处理，多一路请求几乎不影响概要生成的速度。</li>
<li><code>--embed-device cpu</code>：索引向量早就算好了，检索时只需要编码一句问题，CPU 上约 0.2 s，完全够用。</li>
<li><code>--review 0</code>：不重写 review.md，因为里面有人已经校对过的状态。</li>
<li><strong>踩坑：并发写同一个文件</strong>。最初 <code>--append</code> 的实现是“启动时读入全部旧行 → 结束时整文件重写”。手动跑的 300 条和 nightly 的 100 条同时运行时，后结束的那个会把先结束的结果覆盖掉。改成追加模式只用 <code>open(..., "a")</code> 写本次新增的行，两个进程就互不覆盖。经验：多个进程共享的数据文件，一律只追加、不重写。</li>
<li><code>--api-model qwen2.5-coder-7b</code>：用底座而不是 LoRA 模型造数据，避免模型自己蒸馏自己、把已有的偏差越训越深。</li>
<li>seed 用日期：每晚抽到不同的资料块，已经造过题的源块会自动跳过。</li></ul>

<h3>25.6 首晚结果</h3>
<div id="nightly-first">
<p>2026-11-24 首晚：21:07 开始，21:55 结束，总共 47 分钟。报告在 <code>logs/nightly/20261124.md</code>。</p>
<table><tr><th>步骤</th><th>结果</th><th>说明</th></tr>
<tr><td>增量入库</td><td>11968 块，recall@8 = 1.0，GATE PASS</td><td>向量全部复用（新编码 0），只用了 0.5 s。增量哈希起作用了。</td></tr>
<tr><td>代码概要</td><td>1019 个文件全部补完，用时 45 分钟</td><td>和合成任务一起压在 vLLM 上，GPU 利用率 95%。</td></tr>
<tr><td>端到端自测（45 题）</td><td>cite_hit 0.867，cited_any 0.933，first_cited 0.689，p50 8.4 s，TTFT 1.05 s</td><td>6 题没有引用到标准答案的文件，已列在报告里。</td></tr>
<tr><td>线上体检（24 h）</td><td>63 问，11 人，没找到 14，无引用 15</td><td>“没找到”多数是概要补齐之前问的。</td></tr>
<tr><td>合成草稿（seed 20261124，300 块）</td><td>自动通过 92，不通过 75，检索找不回丢弃 36，已造过跳过 44</td><td>草稿从 466 条增加到 633 条。</td></tr></table>
<p>三个值得记住的发现：</p>
<ol><li><strong>first_cited 只有 0.69</strong>：约 3 成回答第一遍没带引用，靠程序补引用才合格。这就是 LoRA SFT 要改的行为：用人审过的“每句带 [n]”样本训练，目标是把这个数提到 0.9 以上。不是“教知识”，是“教规矩”。</li>
<li><strong>概要补齐后，“没找到”的老问题被救回来了</strong>：跑完后重问“udslib 刷写数据的校验和是怎么算的”“EM4 遮挡区域几行几列”“功能安全故障监控怎么实现”，三题都给出了带引用的答案（之前都是“资料中没有找到”）。这说明这类问题缺的是“口语 → 文件”的桥（概要块），而不是模型能力。</li>
<li><strong>合成题的自动通过率约 55%</strong>：不通过的主要是问题太琐碎（如“文件的版权声明是什么”），或者回答没引用源块。这正是需要人审的原因。出题 prompt 已加约束：“问功能、流程、参数含义或排查方法，不要问版权、作者、注释、头文件包含”，从第二晚起生效。</li></ol>
<p>首晚结束后显卡还空着，又追加了一轮 400 块的合成（seed 2026112402），用的就是“只追加”的新写法。结果：自动通过 97，不通过 67，检索找不回丢弃 49，已造过跳过 108，用时 16 分钟。草稿从 633 条增加到 797 条。</p>
</div>

<h3>25.7 后续路线：数据攒够之后再训练</h3>
<table><tr><th>阶段</th><th>做什么</th><th>前提</th></tr>
<tr><td>现在（每晚）</td><td>入库、概要、自测、体检、合成草稿</td><td>无</td></tr>
<tr><td>下一步</td><td>合成题经人审后扩充评测集到 200 题以上</td><td>每天花 10 分钟审 review.md</td></tr>
<tr><td>每周</td><td>检索模型微调：用（真实问题，被引用片段）做对比学习</td><td>评测集够大，能看出 recall 的变化</td></tr>
<tr><td>每周</td><td>LoRA 增量 SFT：用点赞和审核过的答案，混入旧数据防遗忘</td><td>first_cited 有基线；新 adapter 过卡口才热加载</td></tr>
<tr><td>以后</td><td>DPO：用同一问题的点赞答案和点踩答案组成偏好对</td><td>点踩数据攒到上百条</td></tr></table>
<p><em>为什么不每晚都训练：7B 模型训一次 LoRA 要 1 小时以上，而且每天新增的、审过的数据只有十几条。训练太频繁，噪声比信号多。把知识交给检索层（每晚更新），把“怎么说”交给权重（每周更新），这是两种不同节奏的学习。</em></p>
<div class="checks"><strong>自查</strong><ul><li>能说清为什么夜间任务要走线上服务的接口，而不是另起进程</li><li>能解释“夜里只造不用”防的是什么</li><li>能说出 first_cited 和 cited_any 的区别，以及为什么兜底之后要看兜底前的指标</li><li>能说出 flock 相比 PID 文件的优点</li></ul></div>
`},
];
