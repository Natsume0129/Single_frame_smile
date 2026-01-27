下面只讲**怎么用、进什么、出什么、怎么看结果**，不讲背景理论。

---

## 一、怎么用

### 1️⃣ 命令行

```bash
python compare_fc7_pt.py \
  --bg_pt "E:\Matsuda_data\手动标注( vgg-face分析)\features\vggface_fc7_20251029_9-12-9-21_0_0.pt" \
  --rvm_pt p"E:\Matsuda_data\手动标注( vgg-face分析)\features\vggface_fc7_20251029_9-12-9-21_0_0_withbg.pt" \
  --out_dir compare_out \
  --k 20
```

### 2️⃣ 前提（不满足会直接报错）

* 两个 `.pt` **来自同一视频、同一时间片段**
* 帧数一致、顺序一致
* 每一帧是 **fc7 向量**
* 形状等价于：

  * `[T, 4096]`
  * 或 list(T) × `[4096]`
  * 或 dict 里包着上述 tensor

---

## 二、输入是什么

### 输入 1：`bg_fc7.pt`

* **原始背景**
* 内容示例（任一即可）：

```python
torch.Tensor [T, 4096]
# or
{'fc7': Tensor[T, 4096]}
# or
[List[Tensor[4096]]]
```

### 输入 2：`rvm_fc7.pt`

* **RVM + 绿幕**
* 格式 **必须与 bg 对齐**
* 第 `t` 行 = 同一视频第 `t` 帧

---

## 三、输出是什么

`compare_out/` 里会有两个文件：

```
compare_out/
 ├─ report.json        # 结论用（你主要看这个）
 └─ raw_metrics.pt     # 原始数值（画图 / 二次分析用）
```

---

## 四、report.json 怎么看（重点）

### 1️⃣ `cosine_distance`

```json
"dcos_mean"
"dcos_median"
"dcos_p95"
```

**这是最重要的一组**

解释：

* `0` → 两个特征几乎一样
* 越大 → 背景影响越强

**经验判断（fc7）：**

* `mean < 0.03` → 背景影响很小
* `0.03 ~ 0.08` → 有影响，但可能可接受
* `> 0.1` → 背景强烈改变语义表示 ⚠️

---

### 2️⃣ `relative_change`

```json
"rchg_mean"
```

看 **幅值是否被系统性改变**：

* 大，但 cosine 小 → 只是 scale 变了（normalize 可救）
* 大，cosine 也大 → 真语义变了

---

### 3️⃣ `delta_shift`

```json
"mean_delta_norm"
"delta_pca_pc1_explained"
```

**判断是不是 domain shift**

* `mean_delta_norm` 大
  → 绿幕引入了**稳定方向偏移**
* `pc1_explained > 0.3`
  → 不是噪声，是结构性差异 ⚠️

---

### 4️⃣ `knn_overlap_*`

```json
"knn_overlap_mean"
```

**极其重要（结构层面）**

* 1.0 → 邻居结构完全一致
* < 0.7 → 相似性结构已经变了

👉 **如果你后面要做聚类 / 检索 / ranking，这一项比 cosine 还重要**

---

### 5️⃣ `temporal_smoothness_*`

```json
"smooth_bg_mean"
"smooth_rvm_mean"
```

比较：

* rvm 明显更抖 → RVM 边界 / mask 抖动在污染特征
* rvm 更平滑 → 去背景在时间上是正向的

---

## 五、raw_metrics.pt 是干嘛的

里面是**逐帧向量**，给你画图用：

```python
data = torch.load("raw_metrics.pt")
data.keys()

# dcos[t]        每一帧的 cosine 差异
# rchg[t]        幅值变化
# knn_overlap[t] 每帧邻域一致性
```

你可以直接画：

* `dcos(t)` vs 时间
* `bg vs rvm` 平滑性对比

---

## 六、最终结论你该怎么下

直接按这个顺序：

1️⃣ **先看 cosine_distance**

* 小 → 不用纠结背景
* 大 → 继续看下面

2️⃣ **看 delta_shift**

* 有主方向 → 绿幕是新 domain

3️⃣ **看 knn_overlap**

* 低 → 下游任务一定受影响


