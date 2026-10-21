# 修复不等间隔电池遥测的 SOC 估计程序

一套 3.2 Ah 锂离子电池包的遥测处理程序在恒流工况下看似正常，但换成不等间隔采样并跨越不同温度后，SOC 和端电压残差会明显漂移。请检查 `/workspace/battery_estimator` 中的实现并修复它。不能更改输入数据的含义，也不能依赖网络服务。

程序采用一阶 Thevenin 等效电路和二状态扩展卡尔曼滤波。电流约定为正值放电、负值充电。状态为 `z`（SOC）和 `vp`（极化电压）。相邻样本间隔为秒，容量单位为 Ah：

```text
z(k|k-1) = clip(z(k-1) - eta * I(k) * dt / (3600 * C), 0, 1)
a         = exp(-dt / (R1(T) * C1))
vp(k|k-1)= a * vp(k-1) + R1(T) * (1-a) * I(k)
Vhat(k)   = OCV(z,T) - R0(T) * I(k) - vp(k)
```

放电时 `eta=1`，充电时 `eta=0.995`。`R0(T)` 和 `R1(T)` 均按 `R_ref * exp(0.018 * (25-T))` 计算，`C1` 不随温度变化。OCV 必须使用 `model.py` 内给出的节点做分段线性插值，区间外钳位到端点，并加上 `0.00035 * (T-25)` 的温度修正；测量雅可比中的 OCV 斜率取当前分段的解析斜率。

预测协方差应使用状态转移矩阵 `diag(1,a)`。过程噪声为 `diag(q_soc*dt, q_vp*(1-a*a))`，参数取现有类属性；电压测量雅可比为 `[dOCV/dz, -1]`。更新后要钳位 SOC，并用 Joseph 形式更新对称的 2x2 协方差。第一行只用初始状态做一次电压测量更新，不做时间传播；之后每行都必须使用该行与前一行的实际时间差。

命令行入口保持为：

```bash
python -m battery_estimator.cli INPUT.csv OUTPUT.csv --summary SUMMARY.json
```

输入必须包含 `timestamp_s,current_a,voltage_v,temp_c`，时间戳要严格递增，所有数值必须有限；列缺失、重复或倒序时间、非法数字都要以非零状态退出。输出 CSV 的列顺序固定为 `timestamp_s,soc,polarization_v,voltage_estimate_v,residual_v`。摘要 JSON 必须包含 `rows`、`final_soc`、`rms_residual_v`、`max_abs_residual_v`，数值不得为了显示而提前取整。仓库中的 `data/sample.csv` 可用于本地检查。
