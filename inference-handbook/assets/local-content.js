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
`}
];
