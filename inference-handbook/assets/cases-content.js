window.HANDBOOK_CASES = [
{id:'fusion-case',title:'14 · 点云与图像：同一个障碍，为什么对不上？',subtitle:'从前方 20 m 障碍物出发，倒推坐标、时间、关联与可信距离。',priority:'P0',body:`
<div class="callout"><strong>任务不是“把两个传感器拼起来”，而是证明：图像中的这个目标，与这些三维点，确实描述同一时刻的同一个物体。</strong>先做离线感知实验，不接真实车辆或机器人控制。</div>
<h3>问题 1：相机已经看见障碍，为什么还需要雷达？</h3><p>假设前方出现一个障碍。图像分类或检测能提供类别与像素位置，但普通二维检测框并不直接给出可信的米制距离。激光雷达测得三维几何，却不一定仅凭稀疏点就能稳定区分类别。我们的第一个目标是：把雷达点投到图像，检查检测框附近是否存在一致的距离支持。</p><p>这里先用<strong>预设矩形 ROI</strong>代替检测器，让你把几何问题单独看清。网页没有真实照片或检测模型，不能把框当成识别结果；工程中应再接入真实检测输出。</p>
<figure class="diagram"><pre class="mermaid">flowchart TD
 A[同帧图像与雷达点] --> B[核对时间与坐标约定]
 B --> C[外参变换到相机坐标]
 C --> D[内参与畸变模型投影]
 D --> E[过滤深度与视场]
 E --> F[候选点与图像目标关联]
 F --> G[检查遮挡与距离一致性]
 G --> H[输出结果及有效性状态]</pre><figcaption>图 14-1：先证明对应关系，再谈融合结果</figcaption></figure>
<details><summary><span class="eye" aria-hidden="true"></span>展开：为什么不能直接把 xyz 当作像素？</summary><p>因为坐标系同时规定了原点、轴方向和单位。教学雷达轴为 x 向前、y 向左、z 向上；相机轴为 x 向右、y 向下、z 向前。它们的“第一个数”含义不同。即使设备装在同一位置，也必须先转轴。</p><pre><code>p_camera = R_camera_lidar × p_lidar + t_camera_lidar
R = [[ 0, -1,  0],
     [ 0,  0, -1],
     [ 1,  0,  0]]
t = [0, 0, 0]  # 仅本教学算例假设共原点
p_lidar = [20, -1, 0] m
p_camera = [1, 0, 20] m</code></pre><p>本例的 t=0 不是实际安装标定结果。真实设备存在安装位移和旋转，应使用标定文件。T_camera_lidar 表示“把雷达坐标表示转换成相机坐标表示”；反向使用要取逆，不能仅把平移改个符号。</p><p>面试追问：齐次矩阵为什么有 4×4？因为它能把旋转和平移合并成一次乘法；矩阵连乘顺序对应坐标变换顺序，不可交换。</p></details>
<details><summary><span class="eye" aria-hidden="true"></span>展开：一步步算出像素 (680,360)</summary><p>针孔模型用深度做透视除法：离得越远，同样 1 m 横向位移占据的像素越少。这里假设无畸变图像，fx=fy=800 px，主点 cx=640、cy=360，分辨率 1280×720。</p><pre><code>u = fx × Xc / Zc + cx = 800 × 1 / 20 + 640 = 680
v = fy × Yc / Zc + cy = 800 × 0 / 20 + 360 = 360
有效条件：Zc &gt; 0，0 ≤ u &lt; 1280，0 ≤ v &lt; 720</code></pre><p>Zc 是相机光轴深度，不是沿地面的距离，也不是欧氏距离。这个点的欧氏距离为 √401 m。真实镜头若有畸变，应匹配“原始图像+畸变模型”或“校正图像+校正参数”，不能混用。图像做缩放、裁剪、letterbox 后，像素坐标与内参也必须做相应转换。</p><p>面试追问：点在相机后方为何不能画？负深度也能做除法，但没有前向成像意义；必须在除法及像素关联前过滤。</p></details>
<div id="fusion-explorer"></div>
<details><summary><span class="eye" aria-hidden="true"></span>展开：100 ms 为什么能错 20 px？</summary><p>固定相机，目标沿相机 x 方向以 +5 m/s 运动。当前点中心 X=1 m、Z=20 m。雷达落后图像 100 ms，旧点 X=1−5×0.1=0.5 m，因此旧投影 u=660，相对当前 u=680 偏 −20 px。这里是人为设定的匀速运动，不是测得的目标速度。</p><p>时间戳匹配不等于运动补偿。自车运动通常需要带时间的位姿，把历史点变换到参考时刻；旋转扫描雷达甚至一帧里各点时间也不同，需要去畸变。动态目标还会独立运动，仅补偿自车不能修复它。网页复选框使用已知目标速度进行 oracle 补偿，真实系统必须估计运动并处理不确定性。</p><p>故障实验：先设 yaw=0、delay=100，观察 −20 px；再勾选补偿回到基准；最后增加 yaw，看时间补偿为什么不能修复标定旋转误差。此处 yaw 定义为相机 y 轴旋转扰动，不应直接等同车辆坐标中的 yaw。</p></details>
<details><summary><span class="eye" aria-hidden="true"></span>展开：框内取中位数，为什么还会错？</summary><p>二维框包含整个矩形，并不等于物体的精确轮廓。框里可能混入地面、远处墙面、邻车和遮挡物；点云还可能稀疏或落在不同表面。中位数能抑制少量离群值，但若背景占多数，中位数会稳定地给出错误距离。</p><p>应检查有效点数、深度分布、连通/聚类、实例掩码、遮挡关系与多帧一致性。无法可靠关联时返回“无有效距离”，而非默认 0 m 或填一个看似精确的值。投影落进框只生成候选关系，不证明点属于该物体。这也是投影叠加与学习式 BEVFusion 的根本区别。<sup><a href="#cite-7">[7]</a></sup></p><p>工作验收：保存同帧标识、时间偏差、标定版本、有效点数、深度统计与失败原因。视觉叠加只是必要的调试手段，不替代真值评估。</p></details>
<ul class="checks"><li>我能手算 [20,-1,0] 为什么投影到 [680,360]。</li><li>我能区分内参、外参、畸变和时间补偿。</li><li>我知道“框内有点”为什么不足以证明融合正确。</li></ul>`},
{id:'fusion-deploy',title:'15 · 从点云到 BEV：为什么要格子化？',subtitle:'从变长数据难以批处理的问题，理解体素、柱状表示与部署成本。',priority:'P1',body:`
<div class="callout"><strong>点云不是一张固定大小的图片。</strong>每帧点数变化、分布不均；让模型稳定处理它，需要明确空间表示和容量边界，而不只是换一种文件格式。</div>
<h3>先提出工程问题</h3><p>你收到一帧 N×4 的点，字段是 x、y、z、反射强度。N 随场景、距离和传感器状态变化。如果直接把上帧形状或内存容量复用给本帧，可能截断数据或越界；如果为了覆盖最坏情况全部填充，又会浪费显存与计算。</p><p>体素把三维空间划成小盒子，点落入盒子后做聚合；pillar 常把竖直维度合并成柱，再编码高度等特征；BEV 则是从上往下看的空间表达。它们是相关但不同的概念，不能把“统计格子数”称作完整 PointPillars 模型。</p>
<figure class="diagram"><pre class="mermaid">flowchart TD
 A[变长点集 N乘4] --> B[裁剪空间范围]
 B --> C[按体素或柱索引分组]
 C --> D[聚合并编码特征]
 D --> E[稀疏表示或稠密BEV]
 E --> F[检测或占用预测]
 F --> G[精度与资源评估]</pre><figcaption>图 15-1：表示方式决定数据契约与计算成本</figcaption></figure>
<details><summary><span class="eye" aria-hidden="true"></span>展开：为什么索引用 floor 而不是随手转 int？</summary><p>索引公式 i=floor((x−xmin)/s)，s 为体素边长。先把坐标移到网格原点，再除以边长。使用半开区间 [xmin,xmax)，避免最大边界产生越界索引。负数截断与向下取整不同，因此不能拿 int 代替 floor 来处理一般坐标。</p><pre><code>范围：x ∈ [-10,10)，y ∈ [-5,5)，z ∈ [-2,2)
边长：0.5 m
点 x=-0.1 m：ix=floor((-0.1+10)/0.5)=19
点 x=10 m：先判定越界，不进入网格
点数截断策略、每体素容量及溢出统计必须明确</code></pre><p>点落在哪个格子是一回事，格内怎样编码是另一回事。简单均值、学习式点编码、稀疏卷积具有不同的信息损失与成本。不要因为可视化上格子相似，就以为复现了论文。</p></details>
<div id="voxel-explorer"></div>
<details><summary><span class="eye" aria-hidden="true"></span>展开：细网格为什么可能让部署突然变慢？</summary><p>固定 XY 范围，边长减半时平面格子数量约变 4 倍；三维全网格约变 8 倍。若 BEV 特征为 [B,C,H,W]，FP32 字节数是 B×C×H×W×4。这只是一张特征图，不含权重、其他激活、工作区和临时缓冲。</p><p>稀疏实现不一定分配完整三维网格，但仍受非空体素数、邻接关系、分组/排序和内存访问影响。减少点不一定线性加速；某些瓶颈在预处理和数据搬运，而不是网络 FLOPs。网页是几何与内存算例，不能作为真实模型吞吐结论。</p><p>面试追问：动态 shape 的 TensorRT profile 能解决所有变长问题吗？不能。它规定允许的输入形状范围，不能自动修复外部体素化、插件支持、容量溢出或错误张量含义。必须记录 N、非空格数、输入尺寸和各阶段时间。</p></details>
<details><summary><span class="eye" aria-hidden="true"></span>展开：相机与点云到 BEV 后为什么仍不能直接相加？</summary><p>雷达点天然有米制几何，图像每个像素却对应一条空间射线，缺乏唯一深度。图像特征进入 BEV 往往需要深度估计、几何提升、采样或注意力等机制。两路还要统一坐标、分辨率、通道与时间。简单把两块同 shape 数组相加，并不代表语义和空间一致。</p><p>部署工作包括确认多相机顺序、内外参批次、图像增强变换、特征格原点及尺度；处理特殊算子和动态数据；逐层对齐输出。模型整体掉点时，先排查数据契约，不要直接归因于 FP16。</p></details>
<details><summary><span class="eye" aria-hidden="true"></span>展开：用真实 KITTI 数据怎样接上第一步？</summary><p>KITTI object 格式常用 P2、R0_rect、Tr_velo_to_cam。投影链为 q=P2×R0_rect×Tr_velo_to_cam×[x,y,z,1]，再做齐次除法。P2 是 3×4，可能有非零末列，不能简化成只有 fx/fy 的 K。使用同一帧点云、图像和标定，确认原始图像尺寸。<sup><a href="#cite-6">[6]</a></sup></p><p>后续若加入检测或融合模型，按官方划分建立验证集，先对齐 FP32 基线，再比较部署后任务指标。测试样本既要有常见场景，也要覆盖远目标、遮挡和不同点数分布；不能挑几张效果好的截图冒充完整评估。</p></details>
<ul class="checks"><li>我能解释体素、pillar、BEV 的区别。</li><li>我能估算一张特征图的内存，而不把它当峰值显存。</li><li>我知道输入点数变化需要哪些容量和统计信息。</li></ul>`},
{id:'vla-case',title:'16 · VLA：一句“往右拿”，怎么变成动作？',subtitle:'从输出单位错误、状态过期的问题，理解动作契约与闭环。',priority:'P2',body:`
<div class="callout"><strong>VLA 是把视觉、语言与动作联系起来的模型，不是“看图聊天后随便调用机械臂”。</strong>动作的参考系、单位、时间和训练数据约定，都是模型接口的一部分。</div>
<h3>一个具体失败</h3><p>桌面机器人收到“把右侧物体拿过来”。模型输出 0.5，你直接把它当 0.5 m 的位移，这可能严重超出工作空间。实际上它可能是归一化动作、关节值、末端位姿增量，或离散动作 token。没有元数据就无法正确解释。</p><p>这一章不执行真实抓取。我们用一个手写三维动作数组，学习检查接口与时间契约；真实 OpenVLA 等项目需要读取对应实现、检查点及数据统计。不同 VLA 的动作 token、连续头或生成方式并不相同。<sup><a href="#cite-8">[8]</a></sup></p>
<figure class="diagram"><pre class="mermaid">flowchart TD
 A[图像与任务语言] --> C[策略模型]
 B[本体状态及时间戳] --> C
 C --> D[动作表示与解码]
 D --> E[单位和参考系检查]
 E --> F[时效及安全边界检查]
 F --> G[控制器接口]
 G --> H[新观测与状态反馈]
 H --> C</pre><figcaption>图 16-1：模型输出只是闭环中的一环；教学页面不连接控制器</figcaption></figure>
<details><summary><span class="eye" aria-hidden="true"></span>展开：0.5 如何解码成 0.05 m？</summary><p>本例定义归一化值 q∈[-1,1]，物理范围 low=-0.1 m、high=0.1 m。先截断 q，再计算 a=(q+1)/2×(high−low)+low。因此 q=0.5 对应 0.05 m。若 q=1.3，截断到 1，得到 0.1 m，并记录越界。</p><pre><code>示例动作：[−1, 0.5, 1.3]
示例范围：Δx/Δy ∈ [−0.1,0.1] m，Δz ∈ [−0.05,0.05] m
解码结果：[−0.1, 0.05, 0.05] m
必须附带：shape、单位、参考系、采样时间、动作类型</code></pre><p>这里的上下界只是教学统计。真实数据集可能用分位数 p1/p99、其他归一化方法或未归一化维度，夹爪等离散量也不能盲目按米解码。分位数范围更不等于机械设备安全极限。</p></details>
<div id="action-explorer"></div>
<details><summary><span class="eye" aria-hidden="true"></span>展开：为什么预测一整块动作，却可能执行不了？</summary><p>动作块一次输出多步动作，减少反复调用开销，但观测越旧，动作越可能与当前状态不一致。本例 H=4，计划时刻为 0/100/200/300 ms；推理结果 300 ms 才到达，前三个动作时刻已过，只剩最后一个按本例约定尚未过期。网页不定义真实控制策略。</p><p>工程上不能简单“把剩余动作继续发出去”：动作可能是依赖前一步的增量，跳过后含义改变。应明确是重算、重定时、拒绝、插值还是安全回退。要测从采集时间到动作生效的端到端延迟，而不仅是 GPU 前向耗时。</p><p>面试追问：增加 batch 能否改善实时控制？可能提高吞吐，却因排队提高单次动作时延。机器人策略通常更关心新鲜度和尾延迟；不能把服务器吞吐优化直接搬过来。</p></details>
<details><summary><span class="eye" aria-hidden="true"></span>展开：为什么跨机器人数据不能直接混在一起？</summary><p>不同设备有不同关节数、末端坐标、夹爪语义、频率和相机视角。相同长度向量可能完全不是相同动作。数据标准化、任务描述、本体状态和设备差异处理，是跨机器人学习的重要难点。Open X-Embodiment 提供相关数据与研究入口，不意味着所有数据天然共用一个动作空间。<sup><a href="#cite-9">[9]</a></sup></p><p>部署岗位需要做：输入相机排序与预处理、tokenizer 与模型版本匹配、动作维度/解码统计校验、异步队列时效、失败日志、离线回放。是否使用 KV Cache、怎样复用，也必须依模型结构和输入变化分析，不能认为所有 VLA 都等同聊天 LLM。</p><p>验收分层：数值对齐只证明部署结果接近参考模型；任务成功率需环境和任务定义；安全约束需独立设计与验证。本实验只覆盖第一层之前的接口契约检查。</p></details>
<ul class="checks"><li>拿到一个动作数组前，我会先问清单位和参考系。</li><li>我能解释 action chunk 的效率收益与时效风险。</li><li>我不会把解码正确等同任务成功或安全认证。</li></ul>`},
{id:'world-case',title:'17 · 世界模型：为什么一步准，不代表未来准？',subtitle:'从预测误差积累，理解状态、动作条件与闭环评测。',priority:'P2',body:`
<div class="callout"><strong>世界模型学习或近似环境如何变化；策略选择做什么。</strong>两者可以结合，但“生成未来画面”不自动等于“理解可控物理世界”。</div>
<h3>先问它要解决什么问题</h3><p>机器人想比较几种动作可能导致的未来，直接在现实中试错成本高。于是用预测器根据观测历史与候选动作，推演未来状态或图像，再用于训练、规划或评估。不同研究的目标差异很大，不能用一张“未来很像”的视频替代行动效果验证。</p>
<figure class="diagram"><pre class="mermaid">flowchart TD
 A[历史观测] --> B[状态或潜变量表示]
 C[候选动作] --> D[动力学预测]
 B --> D
 D --> E[未来状态或观测]
 E --> F[奖励风险或任务评价]
 E --> D
 F --> G[比较候选行为]</pre><figcaption>图 17-1：动作条件预测与滚动评估；不意味着可放心替代真实验证</figcaption></figure>
<details><summary><span class="eye" aria-hidden="true"></span>展开：先用一条运动方程理解本质</summary><p>参考状态初始 x=0 m、v=1 m/s，加速度 a=0.2 m/s²，步长 dt=0.1 s。参考更新 x'=x+vdt+0.5adt²，v'=v+adt。故意错误的预测器忽略 a，只做 x'=x+vdt，v'=v。</p><p>如果每一步都给错误预测器真实 x、v，它的单步位置误差只有 0.5×0.2×0.1²=0.001 m。可连续用自己的输出滚动 20 步后，预测 x=2 m，参考 x=2.4 m，误差变成 0.4 m。</p><p>这不是训练的世界模型，只是有已知真值的机制实验。故意漏掉一项，是为了让你能完整追踪误差的来源，而不是把复杂模型的误差归结成一句“泛化不好”。</p></details>
<div id="world-explorer"></div>
<details><summary><span class="eye" aria-hidden="true"></span>展开：teacher forcing 与自由滚动区别在哪里？</summary><p>teacher forcing 用真实历史作为下一步输入，自由滚动用自己的预测结果作为历史。一步预测面对的是干净数据分布，滚动会逐步走进自己制造的偏差。因此训练损失低、一步验证漂亮，仍可能在长时域失稳。</p><p>要区分：单步误差、多步误差曲线、不同动作条件的预测差异、长期漂移、任务相关错误。图像指标可以检测外观差异，但不一定捕捉碰撞、几何穿透和行动后果。若模型用于候选动作评价，还要看模型偏差是否把规划引向“模型喜欢但现实失败”的行为。</p><p>面试追问：只用过去视频预测未来，会自然理解动作吗？不保证。没有明确动作输入或可辨识的干预信息时，模型可能只学到观测相关性，不能可靠回答“如果换一种动作会怎样”。</p></details>
<details><summary><span class="eye" aria-hidden="true"></span>展开：真实研究与你的推理工作怎样连接？</summary><p>DreamerV3 是学习潜在世界模型并用于行为学习的研究入口；UniSim 探索用交互视频预测构建模拟器。两者目标、输入输出及评估不同。这里只提供阅读方向，未运行它们，也没有实现可替代真实道路的仿真器。<sup><a href="#cite-10">[10]</a></sup><sup><a href="#cite-11">[11]</a></sup></p><p>部署工作仍落在具体问题：多帧缓存的生命周期、视频/状态张量布局、递归状态重置、长序列显存、采样步骤与延迟、不同精度下滚动误差。量化后一步输出接近，还要检验多步传播的误差，不能只拿一层最大差值宣称效果不变。</p><p>学习顺序上，先完成几何和部署基础，再决定深入这类模型；求职可以解释原理与接口，不应把本章算例写成“大规模世界模型研发经验”。</p></details>
<ul class="checks"><li>我能区分策略、动力学预测器与视频生成器。</li><li>我知道一步误差与自由滚动误差为何不同。</li><li>我能给出模型性能之外的任务验收条件。</li></ul>`},
{id:'case-lab',title:'18 · 动手：制造问题，再用证据修复',subtitle:'四组零依赖算例与一个真实 KITTI 文件入口；先跑通，再逐步替换为真实模型。',priority:'P0',body:`
<div class="callout"><strong>现在只需 Python 标准库，不必安装 CUDA、下载大模型或购买算力。</strong>这一步训练的是数据契约、几何与验证方法；不是声称已经完成融合网络、VLA 或世界模型部署。</div>
<p><a href="./assets/scenario_lab.py" download>下载场景实验脚本 scenario_lab.py</a>。在当前工作区可直接复制下列命令。产物位于 /tmp/inference-case，不写回你的业务数据。</p>
<figure class="diagram"><pre class="mermaid">flowchart TD
 A[运行基准] --> B[故意增加时间或表示误差]
 B --> C[读JSON与中间结果]
 C --> D[用公式定位原因]
 D --> E[修改一个变量]
 E --> F[重复测试并说明边界]</pre><figcaption>图 18-1：每个实践都形成可追溯的问题闭环</figcaption></figure>
<h3>实验 1：雷达投影慢了 100 ms</h3><pre><code>LAB="/home/sti/Documents/Trae-CN/default/llm-basics/inference-handbook/assets/scenario_lab.py"
python3 "$LAB" fusion --yaw-deg 0 --delay-ms 0 --work-dir /tmp/inference-case
python3 "$LAB" fusion --yaw-deg 0 --delay-ms 100 --work-dir /tmp/inference-case
python3 "$LAB" fusion --yaw-deg 2 --delay-ms 100 --work-dir /tmp/inference-case</code></pre><p>成功信号：基准中心 (680,360)，100 ms 旧点中心 (660,360)。第二次结果会覆盖同名 fusion.json，比较前请另存。观察补偿前后及 ROI 统计，不要只看“程序退出码为零”。</p>
<details><summary><span class="eye" aria-hidden="true"></span>展开：怎样读结果，怎样回答面试？</summary><p>先找到参考投影，再看扰动投影；两者差值能与公式相互校验。加入旋转后，单独修复时间差仍有残余误差。回答应包括坐标约定、时间约定、变量控制和证据，不是“我调参把图调对了”。输出光轴深度中位数不是可靠检测距离，演示还混有背景点。</p></details>
<h3>实验 2：改变体素边长</h3><pre><code>python3 "$LAB" voxel --size 0.5 --work-dir /tmp/inference-case
python3 "$LAB" voxel --size 0.25 --work-dir /tmp/inference-case</code></pre><p>比较接受/排除点数、体素索引、占用与内存算例。脚本与网页使用不同的明确标注的教学点集，不能比较两者非空格子的绝对数量；应比较边界处理和尺寸变化规律。脚本 voxel.json 会覆盖，请保留实验配置。</p>
<details><summary><span class="eye" aria-hidden="true"></span>展开：为什么测试要包含负坐标与边界？</summary><p>因为正常的正坐标样例看不出 floor 与截断的区别，也发现不了上边界越界。真实问题常发生在范围裁剪、无效点、空输入或容量极限。单元测试是写清输入约定，而不是给每个函数机械凑覆盖率。</p></details>
<h3>实验 3：动作输出迟到了</h3><pre><code>python3 "$LAB" action --latency-ms 0 --work-dir /tmp/inference-case
python3 "$LAB" action --latency-ms 300 --work-dir /tmp/inference-case
python3 "$LAB" action --latency-ms 500 --work-dir /tmp/inference-case</code></pre><p>查看 action.json：归一化动作、截断、米制解码、动作元数据及过期状态。300 ms 时按计划采样时刻保留最后一步；500 ms 时全部过期。这里只验证契约，绝不能把这些值发给实际设备。</p>
<h3>实验 4：让错误预测滚动更久</h3><pre><code>python3 "$LAB" rollout --steps 1 --work-dir /tmp/inference-case
python3 "$LAB" rollout --steps 20 --work-dir /tmp/inference-case
python3 "$LAB" rollout --steps 40 --work-dir /tmp/inference-case</code></pre><p>比较 rollout.json 中 teacher-forced 单步与自由滚动误差。20 步位置误差 0.4 m。解释为什么单步误差小却不能代表长时域可靠，再考虑实际模型该怎样记录每个时域的任务误差。</p>
<h3>实验 5：替换为公开真实 KITTI 文件</h3><p>从 KITTI 官方渠道按许可准备 object 数据集一帧点云与配套标定。下面路径是待替换的用户数据位置，不会自动下载。width、height 必须读取对应原始图像尺寸；1242×375 仅为常见示例，不适用于所有帧。此脚本不读取图像。</p><pre><code>python3 "$LAB" kitti \
  --points /absolute/path/to/velodyne/000000.bin \
  --calib /absolute/path/to/calib/000000.txt \
  --width 1242 --height 375 \
  --work-dir /tmp/inference-case</code></pre><p>输出 kitti.json 与 kitti_visible.csv。校准必须包含 P2、R0_rect、Tr_velo_to_cam；读取 N×4 little-endian float32 点，使用完整齐次变换后过滤深度和视场。不要混用 raw 数据集的独立校准文件格式，也不要把 image_3 与 P2 随意混配。</p>
<details><summary><span class="eye" aria-hidden="true"></span>展开：真实数据入口已经证明了什么，还缺什么？</summary><p>本次用临时微型 bin 与标定文件验证了读取、完整矩阵投影、视场过滤和 CSV 输出，未下载或验证真实 KITTI 数据。拿到真实数据后，首先人工核对帧号、图像大小、标定及叠加位置；CSV 可以供后续可视化读取。本轮没有实现照片叠加、遮挡处理、实例关联或真实融合网络。</p><p>从这里走向完整岗位作品：增加真实图像叠加 → 接检测器与基线评测 → 评估点关联失败 → 模型导出及逐层对齐 → 分段计时 → 按瓶颈优化。每一步都记录真实结果，不能把完成了投影入口写成“实现多传感器自动驾驶系统”。</p></details>
<h3>每次实践只写这四句话</h3><ol><li>原始问题是什么，输入契约是什么？</li><li>哪个错误假设造成了失败？</li><li>改变哪个变量，有什么证据？</li><li>这次实验还不能证明什么？</li></ol>
<ul class="checks"><li>我独立运行了四组算例，保留配置与产物。</li><li>我能解释 −20 px、动作过期与 0.4 m 的来源。</li><li>我能区分教学算例、真实数据读取和真实模型验证。</li></ul>`}
];
