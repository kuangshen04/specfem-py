# my-specfem

一个单进程的 3D 谱元教学项目，包含弹性和纯声学两个长方体算例。

```powershell
uv run python main.py
uv run python main.py --case acoustic
uv run python -m unittest discover -s tests -v
```

`output/seismogram.csv` 每行是一座台站在一个物理时刻的位移，列为 `time_s, station_id, ux_m, uy_m, uz_m`。内置算例有 201 个采样时刻和两座台站，Ricker 峰值时刻为 0 秒。盒子的六个外表面均为零牵引，波形包含边界反射。算例参数在 `main.py` 的 `homogeneous_box()` 里。

纯声学沿用同一套网格和台站。外表面是刚壁，法向通量为零。算例推进单分量势 `chi`，在每个完整时间步采样 `-chi_ddot`，写成 `output/acoustic_seismogram.csv` 的 `pressure_Pa` 列。点源的振幅是势方程中空间 Dirac 项的系数，单位为 m³。参数在 `main.py` 的 `acoustic_box()` 里。

## 沿一次正演读代码

| 模块 | 输入和输出 |
| --- | --- |
| `mesh.py` | 从盒子边长和单元数生成六面体连通性、材料区域和外边界面。 |
| `sem.py` | 构造 GLL 求积和全局节点映射，提供梯度、弱式积分、装配和插值。源与台站坐标需在单元内部。 |
| `elastic.py` | 从区域材料装配集中质量，应用刚度算子。每个全局节点有 XYZ 三个位移分量，共用同一个标量质量。 |
| `acoustic.py` | 用 `1/kappa` 装配集中质量，用 `grad(chi)/rho` 计算通量。 |
| `acquisition.py` | 把弹性点力或声学标量源投影成全局载荷，定义台站及地震图的分量和单位。 |
| `time_integration.py` | 用显式 Newmark 推进状态。步进器通过函数求加速度。 |
| `forward.py` | 编排初始状态、推进和采样。 |
| `io.py` | 按地震图中的通道名和单位写 CSV。 |

全局场按 `(node, component)` 存储。单元场按 `(element, i, j, k, component)` 存储，`i, j, k` 分别沿三个参考坐标轴。梯度末轴是物理空间方向。共享面的节点拥有同一个全局编号，单元贡献在装配时累加。

测试中另有一个双单元、双材料盒子，检查共享面的 25 个 GLL 点、全局质量、应变能和点力与采样的功配对。时间推进用振子校验，完整弹性正演用时间步减半检查波形。纯声学用刚壁盒子的解析驻波检查频率和压力符号。
