# my-specfem

一个单进程的 3D 谱元教学项目，包含弹性、纯声学和流固耦合三个算例。

```powershell
uv run python main.py
uv run python main.py --case acoustic
uv run python main.py --case coupled
uv run python -m unittest discover -s tests -v
```

`output/seismogram.csv` 每行是一座台站在一个物理时刻的位移，列为 `time_s, station_id, ux_m, uy_m, uz_m`。内置算例有 201 个采样时刻和两座台站，Ricker 峰值时刻为 0 秒。盒子的六个外表面均为零牵引，波形包含边界反射。算例参数在 `main.py` 的 `homogeneous_box()` 里。

纯声学沿用同一套网格和台站。外表面是刚壁，法向通量为零。算例推进单分量势 `chi`，在每个完整时间步采样 `-chi_ddot`，写成 `output/acoustic_seismogram.csv` 的 `pressure_Pa` 列。点源的振幅是势方程中空间 Dirac 项的系数，单位为 m³。参数在 `main.py` 的 `acoustic_box()` 里。

流固耦合把两个长方体沿平直共形面相接。流体中的初始高斯平面脉冲沿界面法向传播，流体台站记录入射和反射压力，固体台站记录透射位移。输出为 `output/coupled_pressure.csv` 和 `output/coupled_displacement.csv`。固体取零第一 Lamé 参数，使平面纵波满足侧壁零牵引，便于在外端反射返回前与一维解比较。

## 沿一次正演读代码

| 模块 | 输入和输出 |
| --- | --- |
| `mesh.py` | 从盒子边长和单元数生成六面体连通性、材料区域和外边界面。 |
| `sem.py` | 构造 GLL 求积和全局节点映射，提供梯度、弱式积分、装配和插值。源与台站坐标需在单元内部。 |
| `elastic.py` | 从区域材料装配集中质量，应用刚度算子。每个全局节点有 XYZ 三个位移分量，共用同一个标量质量。 |
| `acoustic.py` | 用 `1/kappa` 装配集中质量，用 `grad(chi)/rho` 计算通量。 |
| `coupling.py` | 配对两侧界面 GLL 节点，把固体法向位移和流体压力分别装配成另一侧的载荷。 |
| `acquisition.py` | 把弹性点力或声学标量源投影成全局载荷，定义台站及地震图的分量和单位。 |
| `time_integration.py` | 用显式 Newmark 推进状态。步进器通过函数求加速度。 |
| `forward.py` | 编排初始状态、推进和采样。 |
| `io.py` | 按地震图中的通道名和单位写 CSV。 |

耦合保留两侧各自的场编号。在 Newmark 给出两侧的预测场后，`run_coupled()` 先把固体位移的界面贡献交给声学算子，再把本步的声压贡献交给弹性算子。界面算子使用指向流体的固体外法向。场变量推进后，两侧按同一时间轴分别采样压力和位移。

全局场按 `(node, component)` 存储。单元场按 `(element, i, j, k, component)` 存储，`i, j, k` 分别沿三个参考坐标轴。梯度末轴是物理空间方向。共享面的节点拥有同一个全局编号，单元贡献在装配时累加。

测试中另有一个双单元、双材料盒子，检查共享面的 25 个 GLL 点、全局质量、应变能和点力与采样的功配对。时间推进用振子校验，完整弹性正演用时间步减半检查波形。纯声学用刚壁盒子的解析驻波检查频率和压力符号。耦合测试检查面权重、共享边的载荷累加、两个界面方向的转置配对，以及法向入射的反射压力和透射位移。
