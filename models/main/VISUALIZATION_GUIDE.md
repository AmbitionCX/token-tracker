# 训练结果可视化指南

## 概述

完整的可视化系统自动在训练完成后生成以下图表：

1. **损失曲线 (Loss Curves)**
   - 训练集损失和验证集损失随时间的变化
   - 帮助识别过拟合/欠拟合

2. **指标曲线 (Metrics Curves)**
   - Precision, Recall, F1, AUC 随时间的变化
   - 验证集上的指标

3. **混淆矩阵 (Confusion Matrix)**
   - 显示TP, FP, TN, FN
   - 计算精确度、精准率、召回率、F1

4. **ROC曲线 (ROC Curve)**
   - 展示不同阈值下的 TPR vs FPR
   - 计算 AUC-ROC 分数

5. **精准-召回曲线 (Precision-Recall Curve)**
   - 展示不同阈值下的精准率 vs 召回率
   - 特别适合不平衡数据集

6. **类别分布 (Class Distribution)**
   - 测试集中两类样本的分布
   - 显示百分比和绝对数量

7. **总结报告 (Summary Report)**
   - 文本格式的详细指标总结
   - 包含所有关键性能指标

---

## 自动生成（推荐）

训练脚本会在完成后**自动生成所有可视化**：

```bash
cd token-tracker/models/main
python scripts/train_main_model.py
```

**输出位置**：由 `config.yaml` 中的 `training.checkpoint.save_dir` 决定（默认：`./checkpoints`）

**输出文件**：
```
checkpoints/
├── loss_curves.png           # 损失曲线
├── metrics_curves.png        # 指标曲线
├── confusion_matrix.png      # 混淆矩阵
├── roc_curve.png             # ROC曲线
├── pr_curve.png              # 精准-召回曲线
├── class_distribution.png    # 类别分布
├── training_report.txt       # 文本总结
├── best_model.pt             # 最佳模型
└── history.json              # 训练历史
```

---

## 手动生成（可选）

如果已保存训练历史和结果，可单独运行可视化脚本：

```bash
python scripts/visualize_results.py [checkpoint_dir] [output_dir]
```

**示例**：
```bash
# 使用默认目录
python scripts/visualize_results.py

# 指定自定义目录
python scripts/visualize_results.py ./my_checkpoints ./my_results
```

---

## 可视化示例输出

### 损失曲线
```
显示：
- 蓝色曲线：训练集损失
- 橙色曲线：验证集损失
- X轴：Epoch
- Y轴：损失值
```

### 混淆矩阵
```
显示：
- 热力图格式矩阵
- Predicted vs True labels
- 包含：Accuracy, Precision, Recall, F1
```

### ROC曲线
```
显示：
- 橙色曲线：模型ROC曲线
- 蓝色曲线：随机分类器基线
- 标题显示 AUC 分数
```

---

## 文本报告示例

```
============================================================
Training and Evaluation Report
============================================================

TRAINING HISTORY
------------------------------------------------------------
Number of epochs: 50
Final train loss: 0.234567
Final val loss: 0.345678
Best val loss: 0.341234

TEST SET RESULTS
------------------------------------------------------------
Accuracy:  0.9234
Precision: 0.8956
Recall:    0.8765
F1 Score:  0.8860
AUC-ROC:   0.9512
Specificity: 0.9450

CONFUSION MATRIX
------------------------------------------------------------
True Negatives:  58234
False Positives: 1200
False Negatives: 350
True Positives:  2100

CLASS DISTRIBUTION (TEST SET)
------------------------------------------------------------
Class 0: 59434 (96.05%)
Class 1: 2450 (3.95%)
```

---

## 集成到模型中

### 在 `SequenceGNNModel` 的训练中使用：

```python
from training import TrainingVisualizer

# 训练完成后
visualizer = TrainingVisualizer(output_dir='./results')

# 生成所有可视化
plots = visualizer.plot_all_results(
    history=history,
    y_true_final=y_test_true,
    y_pred_final=y_test_pred,
    y_scores_final=y_test_scores
)

# 生成总结报告
report_path = visualizer.generate_summary_report(
    history, y_test_true, y_test_pred, y_test_scores
)
```

---

## API 文档

### `TrainingVisualizer` 类

#### 初始化
```python
visualizer = TrainingVisualizer(output_dir='./results')
```

#### 主要方法

**1. 损失曲线**
```python
visualizer.plot_loss_curves(
    history={'train_loss': [...], 'val_loss': [...]},
    title="Training and Validation Loss",
    save_path=None  # 若为None，使用默认路径
)
```

**2. 指标曲线**
```python
visualizer.plot_metrics_curves(
    history={'val_metrics': [...]},
    metrics=['precision', 'recall', 'f1', 'auc'],
    save_path=None
)
```

**3. 混淆矩阵**
```python
visualizer.plot_confusion_matrix(
    y_true=np.array([...]),
    y_pred=np.array([...]),
    class_names=['Normal', 'Suspicious'],
    save_path=None
)
```

**4. ROC曲线**
```python
visualizer.plot_roc_curve(
    y_true=np.array([...]),
    y_scores=np.array([...]),  # 正类概率
    save_path=None
)
```

**5. 精准-召回曲线**
```python
visualizer.plot_precision_recall_curve(
    y_true=np.array([...]),
    y_scores=np.array([...]),
    save_path=None
)
```

**6. 类别分布**
```python
visualizer.plot_class_distribution(
    y=np.array([...]),
    labels=['Normal', 'Suspicious'],
    title="Class Distribution",
    save_path=None
)
```

**7. 一键生成所有可视化**
```python
plots = visualizer.plot_all_results(
    history=training_history,
    y_true_final=test_true_labels,
    y_pred_final=test_predictions,
    y_scores_final=test_probabilities,
    output_dir=None
)
# 返回字典：{plot_name: file_path}
```

**8. 生成总结报告**
```python
report_path = visualizer.generate_summary_report(
    history=training_history,
    y_true_final=test_true_labels,
    y_pred_final=test_predictions,
    y_scores_final=test_probabilities,
    save_path=None
)
```

---

## 配置说明

在 `config.yaml` 中配置可视化相关设置：

```yaml
training:
  checkpoint:
    save_dir: './checkpoints'    # 保存图表和报告的目录
    keep_best: true              # 是否保存最佳模型
    
  num_epochs: 50
  learning_rate: 0.001
  # ... 其他设置
```

---

## 依赖包

已包含在 `requirements.txt` 中：
- `matplotlib>=3.5.0` - 绘图库
- `seaborn>=0.12.0` - 统计可视化
- `scikit-learn>=1.0.0` - 指标计算
- `numpy<2.0` - 数值计算

---

## 常见问题

**Q: 如何修改图表的样式？**
A: 修改 `visualization.py` 第12-15行的 `plt.rcParams`

**Q: 如何改变输出分辨率？**
A: 修改 `savefig()` 调用中的 `dpi` 参数（默认300）

**Q: 可以自定义文件名吗？**
A: 可以，每个 `plot_*` 方法都接受 `save_path` 参数

**Q: 支持其他文件格式吗？**
A: 支持，只需改变 `save_path` 的扩展名（.pdf, .svg, .jpg等）

---

## 下一步

训练完成后：

1. ✅ 自动生成可视化
2. ✅ 查看 `training_report.txt` 获取数值总结
3. ✅ 在 TensorBoard 中查看实时指标：
   ```bash
   tensorboard --logdir=./checkpoints
   ```
4. ✅ 与基线模型进行对比

