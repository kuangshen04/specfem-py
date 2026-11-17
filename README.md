# my-specfem

一个单进程的 3D 弹性谱元教学项目。v0.1 从内置长方体网格出发，计算点力源激发的三分量位移地震图。

```powershell
uv run python main.py
uv run python -m unittest discover -s tests -v
```

`output/seismogram.csv` 每行是一座台站在一个物理时刻的位移，列为 `time_s, station_id, ux_m, uy_m, uz_m`。内置算例有 201 个采样时刻和两座台站，Ricker 峰值时刻为 0 秒。盒子的六个外表面均为零牵引，波形包含边界反射。算例参数在 `main.py` 的 `homogeneous_box()` 里。

## 沿一次正演读代码

| 模块 | 输入和输出 |
| --- | --- |
| `mesh.py` | 从盒子边长和单元数生成六面体连通性、材料区域和外边界面。 |
| `sem.py` | 构造 GLL 求积和全局节点映射，提供梯度、弱式积分、装配和插值。源与台站坐标需在单元内部。 |
| `elastic.py` | 从区域材料装配集中质量，应用刚度算子。每个全局节点有 XYZ 三个位移分量，共用同一个标量质量。 |
| `acquisition.py` | 把点力投影成全局载荷，定义台站及地震图的分量和单位。 |
| `time_integration.py` | 用显式 Newmark 推进状态。步进器通过函数求加速度。 |
| `forward.py` | 编排初始状态、推进和采样。 |
| `io.py` | 按地震图中的通道名和单位写 CSV。 |

全局场按 `(node, component)` 存储。单元场按 `(element, i, j, k, component)` 存储，`i, j, k` 分别沿三个参考坐标轴。梯度末轴是物理空间方向。共享面的节点拥有同一个全局编号，单元贡献在装配时累加。

测试中另有一个双单元、双材料盒子，检查共享面的 25 个 GLL 点、全局质量、应变能和点力与采样的功配对。时间推进用振子校验，完整正演用时间步减半检查波形。
