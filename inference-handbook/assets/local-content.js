window.HANDBOOK_LOCAL = [
{id:'local-gpu',title:'20 · 本机 GPU 实战：3090 Ti 跑通 PyTorch → ONNX → TensorRT',subtitle:'不进 Docker，用独立 venv 在自己显卡上测出真实延迟与精度。',priority:'P0',body:`
<div class="callout"><strong>结论先说：能用本机跑，而且应该用本机跑。</strong>RTX 3090 Ti 24GB + 驱动 570 足够完成 TensorRT 部署实验、7B 模型推理和 QLoRA 微调。本章所有数字都来自这台机器的日志：<code>gpu-lab/logs/</code>。</div>
<h3>问题 1：明明装了 CUDA，为什么 torch.cuda.is_available() 是 False？</h3>
<p>本机 miniforge 里已有一个 quant-deploy 环境，torch 是 <code>cu130</code> 版本，但 <code>is_available()=False</code>。原因不是显卡坏了，而是<strong>版本链断了</strong>：驱动 570 最高只支持 CUDA 12.8 运行时，cu130 轮子需要 580 以上驱动。</p>
<figure class="diagram"><pre class="mermaid">flowchart LR
 D[显卡驱动 570.133<br/>决定最高 CUDA 12.8] --> R[torch 轮子自带的运行时<br/>cudart/cublas/cudnn]
 R --> T[torch 2.6.0+cu124 ✓]
 D -.不满足.-> X[torch +cu130 ✗<br/>需驱动 ≥580]
 N[系统 nvcc 12.4] -.只在自己编译 .cu/插件时用.-> P[TensorRT 插件 / 自定义算子]</pre><figcaption>图 20-1：驱动是“上限”，pip 轮子自带运行时，系统 nvcc 只服务于自己编译</figcaption></figure>
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
<tr><td>self_lineno 自写行号（越低越好）</td><td>0.0</td><td>0.016</td></tr></table>
<p><strong>怎么读这张表：</strong>收益最大的是拒答，从 0/5 提升到 4/5，这正是提示词教不会、而样本能教会的行为。引用准确率提升 12.7 个点。注意拒答 val 只有 5 条，结论还需要扩大样本复核。另外 self_lineno 出现了 1 例回归，下一轮应把这类样本补进人工校对清单。底座和 LoRA 共用同一份 4bit 权重，底座结果通过 <code>disable_adapter()</code> 得到，所以对比只差在 LoRA 本身。</p>
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
$PY -m pytest -q tests                     # 切块/分词/RRF/去重/引用校验单测</code></pre></details>
<div class="checks"><strong>自查</strong><ul><li>能说清为什么先 RAG 后微调，以及 24GB 能做哪些训练</li><li>能解释 BM25 和向量检索各自擅长什么、RRF 为什么不用归一化</li><li>能拿出评测集数字证明某个优化有效</li><li>能说出至少三条防幻觉手段</li></ul></div>
`},
{id:'local-kb-tutorial',title:'22 · 实战教程：从零部署知识库 Agent + QLoRA（一键脚本版）',subtitle:'第 20、21 章是“踩坑记录”，本章是“照着做就能复现”的部署手册：每一步都有命令、预期输出、验收标准和本机实测耗时。',priority:'P0',body:`
<div class="callout"><strong>一句话：两个脚本、九个阶段，从一台装好驱动的 Ubuntu 机器，做到浏览器里能问“udslib 怎么编译”并拿到带出处的回答。</strong>所有数字都来自本机（RTX 3090 Ti 24GB / 驱动 570 / Ubuntu 20.04），日志在 <code>kb-agent/logs/</code>。</div>
<h3>全景：九个阶段分别产出什么</h3>
<figure class="diagram"><pre class="mermaid">flowchart LR
 E[setup_env<br/>venv+依赖+模型] --> T[test<br/>8 个单测]
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
<tr><td>1 test</td><td rowspan="5"><code>bash scripts/run_pipeline_ubuntu2004.sh</code></td><td>—</td><td>8 passed</td><td>&lt;1 s</td></tr>
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
<tr><td>test</td><td><code>8 passed</code></td><td>ImportError → 用了系统 python，确认 <code>PY=~/venvs/gpu/bin/python</code></td></tr>
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
<tr><td>自写行号（越低越好）</td><td>0.0</td><td>0.016</td><td>⚠ 1 例回归，进入下一轮校对</td></tr></table>
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
];
