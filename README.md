# RoadSurf-3D

## Phase 0 和 Phase 1

### 数据与环境

本地 KITTI Object 根目录为 `E:/litianhong/kitti`。按六位帧号核对 `image_2/*.png`、`velodyne/*.bin` 和 `calib/*.txt`，`training` 有 7,481 帧，`testing` 有 7,518 帧。两组的文件均完整配对，所有 Velodyne 文件的字节数均符合四列 float32 的格式。

几何运行使用 `eiseg` 的 Python 3.8.19、NumPy 1.24.4、Pillow 10.3.0、PyYAML 6.0.1。测试使用本地 Visualization 虚拟环境的 Python 3.9.19 和 pytest 8.4.2。系统 GPU 为 RTX 3050 8 GiB，NVIDIA 驱动 591.86，`nvidia-smi` 报告 CUDA 13.1。当前 PATH 中未发现 `nvcc`，上述两个环境均未安装 PyTorch。本阶段在 CPU 上运行。

### 单帧几何验收

在 RoadSurf 根目录运行。默认使用 [configs/kitti.yaml](configs/kitti.yaml)，从三个输入目录中自动选取首个共有帧。数据根目录可通过配置或 `--kitti-root` 设置，`--split` 和 `--frame` 可指定数据划分与帧号。重复运行时使用 `--out` 指定一个新目录。

```powershell
& 'D:/ProgramData/Anaconda3/envs/eiseg/python.exe' -B run_kitti_geometry.py
& 'E:/litianhong/Visualization/.venv/Scripts/python.exe' -B -m pytest -q -p no:cacheprovider
```

[数据加载器](data/dataset_loader.py) 保留 RGB 原尺寸和原始点云的 `[x, y, z, intensity]` 四列。[KITTI 标定](data/kitti_calib.py) 使用完整 `P2 × R0_rect × Tr_velo_to_cam` 进行投影。射线起点为相机 2 光心，射线参数为其光轴深度（米），方向不做 L2 归一化。裁剪或缩放后的像素网格可通过 `image_transform` 传入原始 UV 到输出 UV 的 3 × 3 仿射矩阵。输出深度同像素保留最近的有效相机前方回波，图外点不参与。

真实数据自动选中了 `training/000000`。RGB 尺寸为 1224 × 370，Velodyne 共 115,384 点，其中 20,285 点投影到画幅内，得到 20,227 个有效深度像素。连续浮点 UV 的投影反投影最大误差为 3.36 × 10⁻¹⁴ m，验收阈值为 10⁻⁵ m。9 项测试通过，另用原始标定分步投影计算了每像素深度最小值，回读确认与导出数组一致。

- [原始 LiDAR 投影叠加图](outputs/kitti_geometry/training/000000/lidar_projection.png)
- [稀疏深度数组](outputs/kitti_geometry/training/000000/sparse_depth.npy) 为原图大小的 float64，单位为米，空像素为 NaN。
- [稀疏深度预览](outputs/kitti_geometry/training/000000/sparse_depth.png) 和 [验收统计](outputs/kitti_geometry/training/000000/metadata.json)

## Phase 2 路面观测

[run_roadsurf.py](run_roadsurf.py) 用 `--dav2-model` 开启单帧 DA-V2 Small 推理，用 `--relative-depth` 复用同帧、原图尺寸的浮点 `.npy` 预测。模型输出只插值回原始 RGB 尺寸，以浮点保存为 `dav2_relative_depth.npy`。灰度预览的归一化仅用于显示。

实测点沿用原始 Velodyne、road 语义、Patchwork++ 及边缘和局部残差筛选。拟合在实测道路点的连续像素坐标双线性采样相对预测 `q`，按 5 m 光轴深度桶均衡权重，用 Huber IRLS 拟合 `1/Z = a*q + b`。检查条件数和正斜率，视觉点仅使用道路内部像素、1–80 m 有效光轴深度和原 LiDAR XY ROI。缺少有效锚点或拟合失败时，结果为 `lidar_only`，诊断原因保存在 `metadata.json`。独立的 `lidar_road.ply` 和 `visual_road.ply` 分别保存实测 XYZ 与强度、视觉 XYZ。

### 实际单帧结果

本地使用 `training/000486` 的 BiSeNetV2 类别缓存和 Patchwork++ 缓存，在 WSL 的 RTX 3050 上实跑 PyTorch 2.3.0、torchvision 0.18.0、Transformers 4.46.3。权重为官方 DA-V2 Small HF revision `5426e4f0f36572d16453bbda7a8389317b1bef99`，`model.safetensors` 的 SHA-256 为 `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`。[官方模型版本](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/tree/5426e4f0f36572d16453bbda7a8389317b1bef99)

| 验收项 | 结果 |
| --- | ---: |
| 实测道路点 / 参与拟合锚点 | 6,649 / 6,609 |
| 校准系数 `a` / `b` | 0.023066 / 0.003037 |
| 锚点拟合内光轴深度 MAE | 0.351 m |
| 视觉道路伪点 | 93,327 |
| 视觉点光轴深度范围 | 5.89–64.48 m |
| 导出视觉 PLY 的重投影最大像素误差 | 0.000052 px 以下 |

40 个实测道路点的投影落在最后一行像素中心下方，像素标签有效，但连续坐标超出双线性采样网格，因此从拟合中排除。回读核对了伪点对应的 road 类别、XY ROI 和度量深度，并验证了缓存重跑和病态拟合时的 LiDAR-only 回退。纵剖面可见中远距视觉点存在带状分层，拟合内 MAE 只描述本帧锚点，独立几何精度留待定量验证。该帧用于开发检查，微调数据重叠见后文。

结果在 [Phase 2 产物目录](outputs/phase2/kitti_object/training/64beams/000486)。可直接查看 [原始相对预测预览](outputs/phase2/kitti_object/training/64beams/000486/dav2_relative_preview.png)、[度量道路深度叠加](outputs/phase2/kitti_object/training/64beams/000486/visual_road_projection.png)、[俯视与纵剖面对比](outputs/phase2/kitti_object/training/64beams/000486/road_observations.png) 和 [统计](outputs/phase2/kitti_object/training/64beams/000486/metadata.json)。

### 服务器复现

保留目录结构上传 `run_roadsurf.py`、`data/{dataset_loader,kitti_calib}.py`、`frontends/{semantic_loader,run_dav2}.py`、`core/{road_lidar_extract,depth_alignment,road_export}.py`、`visualization/visualize_road_observations.py` 和 `requirements.txt`。使用服务器已有的 PyTorch/torchvision 和 KITTI 数据。缺少 Transformers 时执行 `python -m pip install 'transformers==4.46.3'`。缺少 Patchwork++ 缓存时可使用安装了 `pypatchworkpp` 的环境，运行时省略 `--ground-cache`。

在服务器的 RoadSurf 项目根目录运行。下例读取已有的 19 通道概率缓存，按 Cityscapes trainId 取 argmax。

```bash
DATA=/mnt/sdb1/jiangjiazheng/kitti
FRAME=000486
python run_roadsurf.py \
  --kitti-root "$DATA" --split training --frame "$FRAME" \
  --semantic "$DATA/training/semantic_probabilities/$FRAME.npy" \
  --semantic-format probabilities \
  --ground-cache "$DATA/training/ground_patchworkpp/$FRAME.npz" \
  --dav2-model --device cuda:0 \
  --out "outputs/phase2/kitti_object/training/64beams/$FRAME"
```

`--dav2-model` 使用官方 HF 模型及服务器已有缓存。离线服务器可上传本地 `cache/models/dav2_small_hf/` 内的三个模型文件和 `revision.txt`，再传入该目录。GPU 不可用时用 `--device cpu`。已经有同帧相对预测时，可用 `--relative-depth /路径/000486.npy` 代替模型选项。

如果要把服务器上的其他帧带回本地复验，已在本地的 KITTI 帧只需下载同帧的类别图（单通道 uint8 trainId）或原始 19 通道概率文件；若本地缺少该帧 Patchwork++ 缓存，再取同帧 ground `.npz`。已有审核包的 `image_semantics.npz` 用 `--semantic-format trainId --semantic-key classes`。

## Phase 3 单级道路重建

[run_road_surface.py](run_road_surface.py) 读取 Phase 2 的实测 PLY、视觉伪点 PLY 和语义缓存。三条基线分别为 LiDAR RANSAC 高程平面、仅由实测点拟合的三次张量积 B 样条，以及实测点与视觉伪点联合拟合的单级 B 样条。

- RANSAC 使用 0.12 m 竖直残差阈值、256 次采样和固定种子 42，用 SciPy float64 在共识集上重估平面。
- 两种样条以该平面为基底。夹持节点目标间距 4 m，残差用 SciPy float64 稀疏 LSMR 解算，包含 3 次 Huber IRLS、按节点间距计算的二阶差分和小幅系数正则。视觉每 0.4 m BEV 单元取一个高度中位样本，实测和视觉数据项各自按占用单元归一化，视觉项乘 0.25。[SciPy B 样条设计矩阵](https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.interpolate.BSpline.design_matrix.html) · [SciPy LSMR](https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.sparse.linalg.lsmr.html)
- 按 0.2 m 网格采样，限定于实测点的纵向跨度，且到最近实测道路点的 XY 距离小于 1.25 m。每个曲面按实际高度投影，要求落在相机前方的 road 内部像素。最近实测点所在位置的拟合残差超过 0.30 m 时拒绝该网格。阈值约束局部残差，与全局高程无关。

每帧导出 `road_ransac.ply`、`road_lidar_bspline.ply` 和 `road_multimodal_bspline.ply`。PLY 使用 float32 XYZ，单位为米。 `road_surface.npz` 保存 float64 平面参数、样条系数和节点；`road_bev.npz` 保存各方法的网格高程、解析 XY 梯度和独立的有效域掩码。网格数组按方法、x、y 排列，无效单元为 NaN。远距缺测和遮挡位置可以留白。

先验证 `000486`，再以相同参数处理另外 10 帧，覆盖 9 条行驶序列。拟合使用本地 SciPy 1.10.1 CPU 环境，新增帧的 DA-V2 观测使用本地 RTX 3050 生成。18 项测试通过，33 个 PLY 均完成回读、重投影及有效域核查。

| 方法 | 单帧 `000486` 导出点 | 10 帧每帧导出点范围 | 10 帧 LiDAR 拟合内 MAE 均值 |
| --- | ---: | ---: | ---: |
| RANSAC 平面 | 6,982 | 4,019–14,068 | 3.48 cm |
| 纯 LiDAR 单级 B 样条 | 7,315 | 4,231–14,622 | 1.00 cm |
| 多模态单级 B 样条 | 7,329 | 4,222–14,349 | 1.36 cm |

10 帧多模态曲面的相邻有效网格最大高差为 3.53 cm。上坡帧 `005458` 中，局部残差门限拒绝了 RANSAC 平面的 2,790 个网格，两种样条均保留了上坡段。已知语义误差仍会影响边界，例如 `001466` 的部分停车铺装区被标为 road。拟合内 MAE 使用参与拟合的实测点，独立几何精度尚待后续评价。

[单帧高程对比](outputs/phase3/kitti_object/training/64beams/000486/road_height.png) · [单帧多模态 PLY](outputs/phase3/kitti_object/training/64beams/000486/road_multimodal_bspline.ply) · [10 帧统计](outputs/phase3/batch10_summary.csv) · [10 帧投影总览](outputs/phase3/batch10_projection.png)

在 RoadSurf 根目录运行单帧。重复运行可用 `--out` 指定新目录。

```powershell
& 'D:/ProgramData/Anaconda3/envs/eiseg/python.exe' -B run_road_surface.py `
  --kitti-root E:/litianhong/kitti `
  --phase2-dir outputs/phase2/kitti_object/training/64beams/000486
```

本地批量重建使用已生成的 10 帧 Phase 2 目录。复跑时指定新的输出目录。

```powershell
$phase2 = 'outputs/phase2/kitti_object/training/64beams'
$frames = '000005','000702','001466','001590','002586','002702','003066','005458','005999','007430'
foreach ($frame in $frames) {
  & 'D:/ProgramData/Anaconda3/envs/eiseg/python.exe' -B run_road_surface.py `
    --kitti-root E:/litianhong/kitti --phase2-dir (Join-Path $phase2 $frame) `
    --out (Join-Path 'outputs/phase3_recheck' $frame)
  if ($LASTEXITCODE -ne 0) { throw "Phase 3 failed for $frame" }
}
```

## Phase 4 独立评价与深度前端选择

**保留 DA-V2 Small 作为默认深度前端。** 两种视觉前端的共同覆盖高程 MAE 最大差约 0.15 mm，覆盖率最大差约 0.09 个百分点。DA-V2 每个 RGB 只预测一次，三个线束设置可以复用。本地单帧预测时间中位数约 0.275 s，Any2Full 各线束设置约 0.380–0.389 s。Any2Full 在 16 线的法向优势出现在全部 4 条评价序列，并在留出的实测点上将高程 MAE 从 DA-V2 的 1.080 cm 降至 1.033 cm，保留为低线束对照。

### 数据与口径

[锁定名单](outputs/phase4/manifest.json) 包含 KITTI DC `val` 的 12 帧、4 条 Raw drive。通过官方 Object 的 `train_rand` 和 `train_mapping` 将现有 Object RGB/Velodyne/标定与同一 Raw 帧的左相机 DC PNG 配对。满足至少 10 个可匹配 Raw 帧的序列，在排序后按三个四分位取样。排除了 Phase 3 开发序列和 BiSeNet 语义训练及验证的已知映射序列。Phase 6 用 KITTI 官方 OXTS 重新确认名单与 Virtual KITTI 2 的五条种子记录无交集。BiSeNet 语义数据的空映射行仍无法判断序列身份。[KITTI DC](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) · [Object 开发包](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip) · [Virtual KITTI 2](https://arxiv.org/abs/2001.10773) · [Phase 6 OXTS 核验](outputs/phase6/manifest.json)

- 本地没有这 12 帧的既有语义图。本轮使用 PointPainting-master 的 KITTI 微调权重及 [VirConv 已有概率导出器](../VirConv/tools/experiments/v13/export_semantic_probabilities.py) 补齐，结果保存在 `cache/phase4/semantic_probabilities`，各方法和线束设置共用。
- 通过原始点序的方位角回绕识别 64 个扫描段。对每个扫描段拟合带截距的 `z(r)` 估计非均匀仰角，并核验扫描段的方位跨度和倾角顺序。12 帧中，校正后各束的角度偏差中位数最大为 0.0039°。32/16 线按束编号每隔 2/4 束保留，平均保留 50.23%/25.15% 原始点。两组为嵌套的模拟退化。
- 每种线束均重新做投影、Patchwork++、道路筛选和样条拟合。DA-V2 使用当前实测道路点重做逆深度校准。Any2Full 使用当前线束的整帧稀疏深度作 prompt，沿用官方模型的度量输出。两者共享 Phase 3 的样条、视觉权重和有效域规则，另跑纯 LiDAR 样条。Any2Full Small 使用官方 `vits` 权重，SHA-256 为 `c797e3b93a1b520033cb7d11a9d6a313326dc48c1ead0c4b417ac6570c95a03b`。[Any2Full 源码版本](https://github.com/zhiyuandaily/Any2Full/tree/780edf6be2bf9260f1cce8e4ddbcb52d7bcdae9e) · [官方权重](https://huggingface.co/zhiyuandaily/Any2Full/blob/main/checkpoints/Any2Full_vits.pth.tar) · [训练与评价说明](https://arxiv.org/html/2603.05711v1)
- DC PNG 以 uint16 / 256 解码、零值为无效，在全部模型拟合后才加载。对齐抽查使用同像素深度，12 帧的中位绝对差为 0.89–6.83 cm。主参考 `dc_non_input` 排除完整 64 线投影像素的 3 × 3 邻域，使三种线束共用同一个 88,565 点的分母。另报告全部 DC 道路参考和抽稀时留出的实测点 `heldout_lidar`。DC 是多帧累积的半稠密代理参考。[KITTI DC 开发包格式](https://github.com/joseph-zhong/KITTI-devkit/blob/master/readme.md)
- 高程在相同参考 XY 查询，以米为单位报告 MAE/RMSE/Bias。法向由 1.5 m 邻域内具备二维覆盖的局部平面估计，以向上法向夹角统计。高程和法向误差同时提供各自覆盖和三个方法的共同覆盖口径。覆盖率按各方法实际接受点数计算。距离分桶为 5–20、20–40、40–70 m，另按帧和序列统计。
- 道路区域由各方法共用的 BiSeNet 掩码界定。对比线束退化时，额外计算同时被所有线束和方法覆盖的坐标交集，避免不同线束仅覆盖不同难度的点。

### 实测结果

下表为 `dc_non_input`。三个方法的高程和法向误差使用共同覆盖坐标，覆盖率为各方法自己的覆盖率。

| 线束 | 方法 | 高程 MAE（cm） | 法向 MAE（°） | 覆盖率 |
| --- | --- | ---: | ---: | ---: |
| 64 | 纯 LiDAR | 1.648 | 0.749 | 98.60% |
| 64 | DA-V2 | 1.694 | 0.801 | 98.61% |
| 64 | Any2Full | 1.709 | 0.769 | 98.57% |
| 32 | 纯 LiDAR | 1.631 | 0.742 | 97.79% |
| 32 | DA-V2 | 1.665 | 0.780 | 97.87% |
| 32 | Any2Full | 1.672 | 0.743 | 97.79% |
| 16 | 纯 LiDAR | 1.643 | 0.757 | 91.06% |
| 16 | DA-V2 | 1.651 | 0.757 | 91.09% |
| 16 | Any2Full | 1.650 | 0.673 | 91.06% |

本子集的纯 LiDAR 样条在三个线束设置的共同覆盖高程 MAE 均最低。跨线束交集共有 80,411 个主参考点，DA-V2 的高程 MAE 为 1.650/1.635/1.652 cm，Any2Full 为 1.668/1.646/1.651 cm（64/32/16 线）。两种口径支持相同的前端选择。主参考以近距为主，40–70 m 的同线束共同覆盖仅 108/56/24 点，跨线束交集为 24 点。前端选择适用于这次单级样条及其保守有效域，远距结论需要更多独立序列和观测。21 项单元测试通过，并回读核验了 12 套嵌套抽稀、48 张模型预测和汇总计数。

[汇总 CSV](outputs/phase4/results/summary.csv) · [跨线束共同覆盖 CSV](outputs/phase4/results/cross_beam_summary.csv) · [逐帧 CSV](outputs/phase4/results/per_frame.csv) · [逐序列 CSV](outputs/phase4/results/per_drive.csv) · [耗时 CSV](outputs/phase4/results/timings.csv) · [线束与模型审计](outputs/phase4/results/audit.json) · [对比图](outputs/phase4/results/frontend_comparison.png)

在同时具备 PyTorch/torchvision、Transformers、SciPy、OpenCV、PyPI `pypatchworkpp` 和 Any2Full 官方源码的环境运行。本地使用 WSL / CUDA、PyTorch 2.3.0、SciPy 1.13.1、Patchwork++ 1.4.1。下例使用已锁定的名单和已有语义缓存。`--out` 指向一个新目录。

```bash
python run_phase4.py \
  --manifest outputs/phase4/manifest.json --kitti-root /mnt/e/litianhong/kitti \
  --semantic-dir cache/phase4/semantic_probabilities --semantic-format probabilities \
  --dav2-model cache/models/dav2_small_hf \
  --any2full-repo cache/third_party/Any2Full-780edf6be2bf9260f1cce8e4ddbcb52d7bcdae9e \
  --any2full-checkpoint cache/models/any2full_vits/Any2Full_vits.pth.tar \
  --cache-dir cache/phase4/depth --out outputs/phase4/recheck
```

使用 `python -m data.kitti_dc --help` 可用官方 Object 和 Scene Flow 开发包及 Phase 3 输出重新构建名单。另一机器可以复用同帧 `uint8` trainId 缓存，改用 `--semantic-format trainId`。

## Phase 5 观测支持度引导的分级样条

在 RANSAC 基准平面上先拟合 4 m 三次 B 样条，再用 2 m 三次 B 样条拟合剩余高度残差。两级分别使用 SciPy 稀疏 float64 最小二乘和 3 次 Huber IRLS，LiDAR 与视觉数据项分别归一化，视觉模态系数为 0.25。两阶段依次固定前一级系数，得到顺序残差解。[SciPy LSMR](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.sparse.linalg.lsmr.html)

| 消融 | 在上一步基础上的变化 |
| --- | --- |
| A0 | 4 m 多模态单级样条，采用统一可见道路约束 |
| A1 | 加入 2 m 细尺度残差 |
| A2 | 用真实道路 LiDAR 支持度抑制低支持区的细尺度系数 |
| A3 | 粗、细两级加入视觉几何误差权重 |
| A4 | 使用 A3 的同一曲面，加入有界隐藏道路有效域 |

- LiDAR 支持度结合 2 m 邻域中的最近距离、0.2 m 网格占用密度和二维方向覆盖。方向因子来自中心化加权信息矩阵的空间特征值，单条共线扫描的方向支持为零。按细基函数足迹求 `c_k` 后，对细系数加 `0.10 / n × (1−c_k)²` 的二次惩罚，`n` 为细控制点数。支持度范围为 `[0,1]`，用于表达观测约束强弱。[SciPy 邻域查询](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.spatial.cKDTree.query.html)
- 视觉权重使用单级样条的解析梯度计算 `Jz = rz − Sx·rx − Sy·ry`，将光轴深度扰动及水平位置扰动传播到高程残差。先验为 `σZ = 0.08 + 0.025Z` m、水平 `σXY = 0.10` m、残差标准差下限 0.035 m、原始权重上限 250。视觉可靠性依据附近 LiDAR 单模态粗样条的一致性单独计算，缺少附近道路锚点时取 0.35。这里的误差参数是实验先验，可靠性是几何代理。
- 0.2 m 可见网格须处于真实道路 LiDAR 的测量包络，距最近道路 LiDAR 不超过 1.25 m，距最近保留观测不超过 0.45 m，回投落在腐蚀后的 road 区域，且邻近锚点的模型误差不超过 0.30 m。隐藏网格须被实测支持的可见区四向包围，距可见区不超过 1.8 m，空洞最大范围 6.4 m × 3.4 m、面积不超过 20 m²。真实遮挡还须投到前景语义，并有至少近 0.35 m 的 LiDAR 深度作为遮挡证据。[SciPy 有界空洞填充](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.ndimage.binary_fill_holes.html)
- `road_visible.ply`、`road_hidden.ply`、`road_complete.ply` 分别存可见点、推断点和二者并集。`road_surface.npz` 存 float64 系数和节点。`road_bev.npz` 分别记录可见性、有效域、LiDAR 支持度、视觉可靠性、高程、法向和坡度。`core.sgsr_export.load_road_surface` 提供高度、解析梯度、法向、支持度和有效域查询。坡度相对于当前 Velodyne 轴。

### 验证口径与复现

沿用 Phase 4 的锁定 12 帧、DA-V2 相对深度和 BiSeNet 缓存，逐线束重新提取道路与校准。合成空洞用全分辨率可见道路选定两种矩形规格（横向 × 纵向 2 × 4 m 和 3 × 6 m），同帧各线束共用位置。拟合前删除空洞内的原始 LiDAR 和视觉伪点，RGB road 仍可用。合成补全只接收空洞推断回投到 road 的网格，推断函数不接收留出矩形坐标。评价使用空洞内留出的实测道路点，也提供独立 DC 代理参考。DC 参考在本帧所有模型拟合后读取。

一共完成 96 组重建，包含 36 组原始场景、33 组 2 × 4 m 空洞和 27 组 3 × 6 m 空洞。两种空洞分别在满足可见道路包围条件的 11 帧和 9 帧构造。所有组完成逆深度校准。下表为原始场景独立 DC 参考的共同覆盖误差，每格为高程 MAE（cm）/ 法向 MAE（°）。A4 在共同坐标的曲面误差与 A3 相同。

| 线束 | A0 单级 | A1 分级 | A2 支持度 | A3 几何加权 | A4 覆盖率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 1.684 / 0.792 | 1.589 / 0.741 | 1.590 / 0.720 | 1.577 / 0.693 | 97.81% |
| 32 | 1.657 / 0.768 | 1.593 / 0.773 | 1.594 / 0.738 | 1.589 / 0.714 | 97.08% |
| 16 | 1.652 / 0.761 | 1.614 / 0.872 | 1.612 / 0.813 | 1.614 / 0.800 | 90.47% |

A1 降低了三种线束的高程 MAE，32 和 16 线的法向误差上升。A2 将 `c_k < 0.25` 的细系数 RMS 中位数从 2.64/2.73/2.97 cm 降至 0.95/1.14/1.28 cm（64/32/16 线），并降低三种线束的法向误差。A3 进一步改善三种线束的法向，64/32 线的高程 MAE 也下降。16 线最终法向 MAE 为 0.800°，高于单级的 0.761°。

下表以合成空洞中留出的真实 LiDAR 点为参考。覆盖率的分母为对应空洞中的全部留出点。最后一列仅对 A4 新增推断点计算误差。

| 空洞 | 线束 | A3 → A4 覆盖率 | 新增覆盖点 | 新增高程 MAE（cm） |
| --- | ---: | ---: | ---: | ---: |
| 2 × 4 m | 64 | 59.45% → 100.00% | 1381 | 0.543 |
| 2 × 4 m | 32 | 59.45% → 100.00% | 1381 | 0.485 |
| 2 × 4 m | 16 | 44.25% → 56.58% | 420 | 0.510 |
| 3 × 6 m | 64 | 37.83% → 100.00% | 3842 | 0.904 |
| 3 × 6 m | 32 | 37.41% → 100.00% | 3868 | 0.967 |
| 3 × 6 m | 16 | 24.32% → 27.01% | 166 | 0.690 |

在空洞独立 DC 参考上，新增推断点高程 MAE 为 1.245–1.543 cm。原始场景 12 帧的 64 线输出共接受 36 个隐藏网格，分布在 6 帧。真实遮挡下没有独立高程真值，量化结果限于满足筛选条件的合成留出区域。回读核对了 17 套 PLY 与网格、解析查询和投影证据，26 项测试通过。

[消融汇总](outputs/phase5/ablation/ablation_summary.csv) · [逐帧指标](outputs/phase5/ablation/ablation_per_frame.csv) · [输入与支持度审计](outputs/phase5/ablation/audit.json) · [真实道路投影](outputs/phase5/pilot/000486/road_projection.png) · [合成空洞与支持度](outputs/phase5/ablation/examples/hole_3x6_64beams/road_domain_support.png)

```bash
python run_sgsr.py --kitti-root /mnt/e/litianhong/kitti \
  --phase2-dir outputs/phase2/kitti_object/training/64beams/000486 \
  --out outputs/phase5/pilot/recheck

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run_phase5.py \
  --manifest outputs/phase4/manifest.json --kitti-root /mnt/e/litianhong/kitti \
  --semantic-dir cache/phase4/semantic_probabilities \
  --dav2-cache cache/phase4/depth/dav2 --out outputs/phase5/recheck
```

批量入口在已有 NumPy、SciPy、Pillow、OpenCV 和 Patchwork++ 的环境中运行，`--out` 使用新目录。可加 `--frames 000575` 复查单帧消融。

## Phase 6 独立验收

使用与 Phase 2–5 的已知开发序列隔离的 13 帧，运行 64/32/16 线、B0/B1 和 A0–A4，并评价 2 × 4 m 与 3 × 6 m 合成几何观测空洞。生成 13 帧可见/隐藏/完整 PLY 和结构化曲面。独立参考、实际指标、数据泄漏核查及适用边界见[验收报告](outputs/phase6/acceptance_report.md)。

`data/phase6_holdout.py` 用官方映射、开发帧列表和 OXTS 核查结果构建锁定名单。`data/check_vkitti_overlap.py` 对比 Virtual KITTI 种子的官方 GPS/IMU 位姿。`data/check_bisenet_overlap.py` 直接比对官方语义归档的训练 RGB。`run_phase6.py` 分为输入准备和独立验收，复用 Phase 5 重建入口。验收前验证缓存和输入哈希，分帧、分序列及分距离汇总。`tests/test_phase6.py` 检查留出点和参考排除规则。

本地 WSL 中，依赖沿用 Phase 4 的 PyTorch 2.3.0、Transformers 4.46.3、SciPy 1.13.1 和 Patchwork++ 1.4.1。现有 `cache/phase6/predictions` 已验证，可直接执行评价，`--out` 使用新目录。

```bash
export PYTHONPATH=cache/.runtime/dav2:cache/.runtime/phase4:cache/.runtime/phase3-test
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -B run_phase6.py evaluate --manifest outputs/phase6/manifest.json \
  --kitti-root /mnt/e/litianhong/kitti --cache-dir cache/phase6/predictions \
  --out outputs/phase6/recheck
python -B -m visualization.plot_phase6 outputs/phase6/recheck/summary.csv \
  outputs/phase6/recheck/acceptance_summary.png
python -B -m pytest -q -p no:cacheprovider tests
```

需要在另一机器准备同批缓存时，复用锁定名单及原有权重。`--cache-dir` 使用新目录。图表脚本另需 Matplotlib。

```bash
python -B run_phase6.py prepare --manifest outputs/phase6/manifest.json \
  --kitti-root /mnt/e/litianhong/kitti --cache-dir cache/phase6/new_predictions \
  --dav2-model cache/models/dav2_small_hf --device cuda:0 \
  --bisenet-repo /mnt/e/litianhong/PointPainting-master \
  --bisenet-weights /mnt/e/litianhong/PointPainting-master/BiSeNetv2/checkpoints/BiseNetv2_150.pth \
  --semantic-exporter /mnt/e/litianhong/VirConv/tools/experiments/v13/export_semantic_probabilities.py

python -B -m data.phase6_holdout --kitti-root /mnt/e/litianhong/kitti \
  --devkits cache/phase6/devkits --previous outputs \
  --vkitti-audit cache/phase6/devkits/vkitti_audit.json --out outputs/phase6/manifest_recheck.json
python -B -m data.check_bisenet_overlap --kitti-root /mnt/e/litianhong/kitti \
  --manifest outputs/phase6/manifest.json --devkits cache/phase6/devkits \
  --out outputs/phase6/bisenet_image_audit_recheck.json
```

## 既有单帧道路点验收

当前入口读取 KITTI Object 左相机 RGB、BiSeNetV2 语义缓存、原始 Velodyne 和标定。设置 `--kitti-root`、`--split` 和 `--frame` 可更换帧。依赖见 [requirements.txt](requirements.txt)。本地真实帧使用 Python 3.8.19、NumPy 1.24.4、Pillow 10.3.0。

## 本地缓存与运行

在 `E:/litianhong/VirConv/output/v14/g0d_semantic_audit/frames/<frame>/image_semantics.npz` 找到 30 帧。全部缓存中的 RGB 和标定副本与本地 KITTI Object 对应帧逐文件相同。29 帧为 1242 × 375，`001466` 为 1224 × 370。

[本地审核导出代码](../VirConv/tools/experiments/v14/g0d/export.py) 将原图尺寸的 19 类概率图保存为 `classes`（uint8 trainId argmax）及 `road`（已有 float16 道路通道）。[本地概率导出代码](../VirConv/tools/experiments/v13/export_semantic_probabilities.py) 使用上游 KITTI 预处理，将 logits 双线性插值到每帧原图尺寸后计算 softmax。验收读取 `classes`，无需再调整该帧的标签尺寸。服务器全概率缓存路径见审核目录的 `metadata.json`。

在 RoadSurf 项目根目录运行以下命令。

```powershell
& 'D:/ProgramData/Anaconda3/envs/eiseg/python.exe' -B run_roadsurf.py `
  --kitti-root E:/litianhong/kitti --frame 000486 `
  --semantic E:/litianhong/VirConv/output/v14/g0d_semantic_audit/frames/000486/image_semantics.npz `
  --semantic-key classes --semantic-format trainId
```

- 索引图用 `--semantic-format trainId` 或 `labelId`。二者像素值重叠时应依据导出代码显式指定。道路分别为 0 和 7，人行道分别为 1 和 8。输出统一为 uint8 trainId，忽略区为 255。
- 纯色图用 `--semantic-format color`，按 Cityscapes RGB 精确解码。若用 OpenCV 读入数组，先从 BGR 转为 RGB 再存入数组缓存。纯色图存在复用颜色的类别，按 19 类调色板解释。混合或经颜色插值的图像会在出现非调色板像素时被拒绝，实际使用应选原始预测缓存。
- 对上游整图缩放后的标签直接按实际 RGB 宽高做最近邻恢复。原预处理若为裁剪、缩放、填充，传入原图裁剪框 `--crop X Y W H` 和标签网格 padding `--padding L T R B`。恢复顺序为去填充、最近邻缩放到裁剪框、回填原图，其余像素为 255。
- 投影使用完整 `P2 × R0_rect × Tr_velo_to_cam`。先在全扫描上选取有效相机深度的同像素最近点，再应用 XY ROI 与语义、几何条件。默认复用 `<split>/ground_patchworkpp/<frame>.npz`，可用 `--ground-cache` 指定。缺少缓存时尝试已安装的 `pypatchworkpp`。
- 道路候选在扣除两像素语义边缘后，以局部鲁棒平面残差排除异常点。局部拟合采用 2 m 网格和周围支持点，支持不足时保留 Patchwork++ 判定。PLY 四列均源自原始 Velodyne，第四列是反射强度。筛选输入为 RGB 语义、完整标定和点云几何。

源码核对涵盖 [类别映射](https://github.com/AmrElsersy/PointPainting/blob/master/BiSeNetv2/utils/label.py)、[KITTI 预处理](https://github.com/AmrElsersy/PointPainting/blob/master/BiSeNetv2/utils/utils.py)、[纯色与混合图导出](https://github.com/AmrElsersy/PointPainting/blob/master/BiSeNetv2/visualization.py)、[投影与点云第四列](https://github.com/AmrElsersy/PointPainting/blob/master/pointpainting.py) 和 [BEV 裁剪](https://github.com/AmrElsersy/PointPainting/blob/master/bev_utils.py)。RoadSurf 使用实际尺寸、原始强度、指定 70.4 m ROI。

## 实际验收

输出目录为 [000486](outputs/kitti_object/training/64beams/000486)，包含六项验收文件和 [metadata.json](outputs/kitti_object/training/64beams/000486/metadata.json)。重复运行可用 `--out` 指定新目录。

| 项目 | 结果 |
| --- | ---: |
| RGB / 语义图 | 1242 × 375 / 1242 × 375 |
| 语义格式 | uint8 Cityscapes trainId |
| 道路 / 人行道像素 | 96,589 / 13,660 |
| 原始 Velodyne 点 | 108,786 |
| ROI 内有效最近投影点 | 17,661 |
| 语义道路候选 | 7,038 |
| 与 Patchwork++ 取交集后 | 6,943 |
| 加入语义边缘和局部残差筛选后 | 6,649 |

叠加图的绿色为 road，粉色为 sidewalk。在 `000486` 中，掩码与车道、车辆轮廓和两侧人行道整体对齐。投影点主要在车道上，车体、建筑立面和人行道主体未见成片混入。路缘邻近仍有少量点难以单凭图像判定。上坡试查帧 `003066` 的硬标签将右侧部分铺装区域误标为 road，几何上与道路相近的点仍可能通过。当前结论限于单帧开发验收。

回读验证了六项文件、原扫描四列数据、输出像素的 road 类别及同像素最近深度。真实帧连续坐标的投影反投影最大误差为约 1.93 × 10⁻¹⁴ m。保留了 38 个前向距离超过 50 m 的点。运行 `python -m pytest -q` 可复查坐标闭环、全扫描遮挡、坡道局部残差、类别映射和裁剪填充恢复。

## 权重来源及检测划分核对

PointPainting 作者记录了 CoinCheung 的 Cityscapes BiSeNetV2 初始化和 KITTI 微调，发布权重为 `BiseNetv2_150.pth.tar`。上游训练代码按排序后的 KITTI 2015 语义 training 前 180 张训练、后 20 张验证。训练 notebook 使用 `BiseNetv2.pth` 初始化，未给出该初始化文件的哈希。[PointPainting 权重说明](https://github.com/AmrElsersy/PointPainting#download-the-checkpoint) · [训练 notebook](https://github.com/AmrElsersy/PointPainting/blob/master/BiSeNetv2/BiseNet_Train.ipynb) · [训练划分](https://github.com/AmrElsersy/PointPainting/blob/master/BiSeNetv2/dataset.py) · [CoinCheung BiSeNet](https://github.com/CoinCheung/BiSeNet)

本地 `E:/litianhong/PointPainting-master/BiSeNetv2/checkpoints/BiseNetv2_150.pth` 的 SHA-256 为 `81d3d71c8d3605313d6bea34a6b423374a01731adb7074490f37e417b4d9e529`。审核缓存元数据记录了原概率目录，未记录生成时 checkpoint 哈希，因此本地文件无法单独证明与缓存所用权重逐字节一致。

KITTI 2015 语义图与 Stereo/Flow 2015 对应。使用两个官方开发包的映射，将语义训练图与 Object 检测帧都还原为 `(date, drive, raw_frame)`。Scene Flow 映射直接以语义编号查对应行，Object 映射以 `train_rand[object_id] - 1` 查 `train_mapping`。逐帧交集按三个字段比较，序列交集按日期和 drive 比较。[KITTI 语义说明](https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015) · [Scene Flow 官方开发包](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_scene_flow.zip) · [Object 官方开发包](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip)

| 上游语义开发数据 | 有 Raw 映射 | 与本地检测 val 同帧 | 检测 val 中同序列帧 | 重叠序列 |
| --- | ---: | ---: | ---: | ---: |
| 前 180 张微调图 | 141 | 31 | 2,032 / 3,769 | 9 |
| 全部 200 张，包括微调验证 | 142 | 32 | 2,328 / 3,769 | 10 |

例如检测帧 `000035` 和 `000216` 都与微调集精确同帧。验收帧 `000486` 和微调集有序列重叠，用于开发检查。语义训练数据的空映射行以及官方 Object testing 暂时无法用这两份公开映射判定，统计是已确认下界。正式实验应锁定检测评价名单，按 Raw drive 排除微调及其验证所用序列，或使用只经 Cityscapes 训练的权重作为独立对照，并披露映射缺失。

审计输入 SHA-256 如下。

| 输入 | SHA-256 |
| --- | --- |
| Scene Flow `train_mapping.txt` | `6cd7e3a60042a3cfa5e05a0b058e6872250ad28c88a704d03ac13a0764585ceb` |
| Object `train_mapping.txt` | `ca9a80524adbfbebc3179987e70a8dab1488a88ba368e77715d561c901196a54` |
| Object `train_rand.txt` | `0976b8b4f876f538010a9690249b60c0766d852e23acd5e07b4c3e0bd70f305f` |
| 本地 `ImageSets/val.txt` | `657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86` |

Phase 3 已用这些实测道路点与视觉伪点完成单级 B 样条拟合。
