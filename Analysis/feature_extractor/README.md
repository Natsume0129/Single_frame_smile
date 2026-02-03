## for feature extractor fc7

## 这段代码做什么？

目标：

* 从一个文件夹里按顺序读取所有人脸图片（224×224）
* 用 **VGG-Face（conv + fc6 + fc7 + fc8）** 作为特征提取器
* 取 **fc7 层输出（4096 维）** 作为每张图的特征向量
* 最终得到一个二维矩阵：

  * `T = 图片数量`
  * `D = 4096`
  * 形状为 `(T, 4096)`
* 将结果保存到一个 `.pt` 文件里（包含文件名列表和特征矩阵）

---

## 代码结构解释（逐块）

### 1) 读取图片的数据集 `ImageFolderDataset`

作用：

* 给一个文件夹 `img_dir`
* 找出其中所有图片文件（jpg/png/…）
* 按文件名排序（保证时序一致）
* 每次返回：`(img_tensor, filename)`

关键点：

* `sorted(...)` 很重要：`DataLoader(shuffle=False)` + 排序 = 输出特征严格按顺序对齐。

---

### 2) 构建“完整 VGGFace”网络 `VGGFaceFull`

为什么要自己写这个类？

* 你原来的 `face_comp_torch.py` 只有 `VGGFace_conv`（只有卷积层）
* 但 `vggface.pth` 里有 `fc.fc6/fc.fc7/fc.fc8` 权重
  所以你必须补上 `fc` 部分，才能用上完整权重。

这个类做了两件事：

#### (1) 复用你已有的卷积定义

```python
self.features = FCmodel.VGGFace_conv().features
```

这会得到一套 `features.conv_1_1 ... conv_5_3` 的结构，与权重 key 完全一致。

#### (2) 新增 fc 层（fc6/fc7/fc8）

```python
self.fc = nn.ModuleDict({
    "fc6": nn.Linear(512*7*7, 4096),
    "fc7": nn.Linear(4096, 4096),
    "fc8": nn.Linear(4096, fc8_out),
})
```

> 注意：`fc8_out` 是从权重里推断出来的（你打印出的 2622），这保证维度完全匹配。

---

### 3) 为什么输出是 fc7？

你代码里用的是 `forward_fc7()`：

```python
x = F.relu(self.fc["fc6"](x))
x = F.relu(self.fc["fc7"](x))
return x
```

也就是说：

* conv → flatten 得到 25088
* fc6 → 4096
* fc7 → 4096
* **直接返回 fc7**
* fc8 不参与（fc8 是身份分类 logits，不适合作 embedding）

因此最终每张图输出就是 **4096 维向量**。

---

### 4) 为什么要做 normalize？

```python
feats = F.normalize(feats, dim=1)
```

作用：

* 把每个特征向量归一化成单位长度
* 这样后续用 cosine similarity、距离比较、PCA/聚类时更稳定
* 也更符合“embedding 空间”的常规使用方式

---

### 5) 主流程 `main()`

核心步骤：

1. 读取命令行参数：权重路径、图片文件夹、保存路径、mode、device等
2. 选择设备：

   * `cuda:0` 有就用 GPU
   * 没有就自动 fallback 到 CPU
3. 加载权重：

   ```python
   sd = torch.load(args.weights, map_location="cpu")
   ```
4. 从权重推断 fc8 输出维度：

   ```python
   fc8_out = sd["fc.fc8.weight"].shape[0]   # 2622
   ```
5. 构建网络并加载权重：

   ```python
   model_full = VGGFaceFull(fc8_out)
   model_full.load_state_dict(sd, strict=True)
   model_full.to(device).eval()
   ```
6. 构建预处理（Resize + ToTensor + Normalize）
7. DataLoader 逐 batch 推理，得到特征并保存到列表
8. 拼成 `(T, 4096)` 并保存：

   ```python
   torch.save({"names": all_names, "feats": feats_mat}, args.save)
   ```

---

## 怎么使用？

### 1) 文件放置（最简单）

建议目录同级：

```
VGG-Face_Analysis/
  feature_extractor_fc7.py
  face_comp_torch.py
```

### 2) 命令行运行（GPU）

```powershell
python feature_extractor_fc7.py `
  --weights "E:\Single_frame_smile\data\models\vggface.pth" `
  --img_dir  "E:\Matsuda_data\手动标注( vgg-face分析)\after_facetracking_withbg\DetectedFaces\20251029_9-12-9-21\0\0" `
  --save     "E:\Matsuda_data\手动标注( vgg-face分析)\features\vggface_fc7_20251029_9-12-9-21_0_0_withbg.pt" `
  --mode fc7 `
  --device "cuda:0"
```
```powershell
python feature_extractor_conv5_3.py `
  --weights "E:\Single_frame_smile\data\models\vggface_conv.pth" `
  --img_dir  "E:\Matsuda_data\手动标注( vgg-face分析)\after_facetracking\DetectedFaces\20251029_9-12-9-21\0\0" `
  --save     "E:\Matsuda_data\手动标注( vgg-face分析)\features\20251029_9-12-9-21_0_0_rvm" `
  --device "cuda:0"
```

输出示例（你已经看到）：

* `Num images: 329`
* `Feature shape: (329, 4096)`

---

## 结果文件怎么读？

```python
import torch

data = torch.load(r"E:\...\vggface_fc7_20251029_9-12-9-21_0_0.pt")
names = data["names"]   # list[str], 长度=329
feats = data["feats"]   # torch.Tensor, shape=[329,4096]

print(names[0], feats[0].shape)
```

解释：

* `names[i]` 对应第 i 张图片的文件名
* `feats[i]` 对应这张图片的 fc7 特征向量（4096维）

---
