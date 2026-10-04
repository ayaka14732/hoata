# Hoata

Hoata is the compiler explorer for TPU

左侧编辑 JAX/Pallas Python，右侧查看从实际机器映像反汇编得到的 `.tpuasm`，点击源码或指令查看双向对应关系。

首版作为本地应用运行，支持 TPU v4 TensorCore 和 TPU v6e TensorCore 的离线编译，不需要 TPU 硬件。仅通过按钮或 Ctrl/⌘+Enter 编译，不自动提交编辑内容。

## 许可证

MPL-2.0.

## 发音与名字来源

Hoata 读作 **HO-a-ta**，重音在第一个音节 **HO**，中文可近似记作“霍阿塔”；`o` 和 `a` 连贯发音。元音读法可参考[毛利语发音指南](https://www.maorilanguage.net/how-to-pronounce-maori/)。

名字来自作者 **Ayaka Mikazuki** 的姓氏：**Mikazuki（三日月）** 的意象用毛利语 **Hoata** 表达，取“阴历第三夜的月亮”这一义项，见 [Te Aka 毛利语词典](https://maoridictionary.co.nz/word/1306)。这也呼应了 [Godbolt（Compiler Explorer）](https://godbolt.org/) 以作者 **Matt Godbolt** 的姓氏为人所知的命名方式。

## 启动

需要 Linux x86-64、g++、CPython 3.14t、Node.js 22.12+、pnpm 12.6.0 和 Rust。pnpm 版本由 `package.json` 的 `packageManager` 固定，可按 [pnpm 安装说明](https://pnpm.io/installation)准备；前端依赖由 `pnpm-lock.yaml` 锁定。Rust 1.98.1 及 Wasm 编译目标由 `rust-toolchain.toml` 固定。

在已配置 JAX/tpuasm 的工作区内：

```sh
export PATH="$HOME/.cargo/bin:$PATH"
pnpm install --frozen-lockfile
cargo install wasm-bindgen-cli --version 0.2.114 --locked
export HOATA_PYTHON=/srv/workspace/venv/bin/python
pnpm dev
```

打开 <http://127.0.0.1:5173>。开发命令同时启动 Vite 与 Rust 服务；退出时停止两者。修改 Rust 或 Python 代码后重新启动开发命令。

使用正式构建，由 Rust 服务同时提供网页与 API：

```sh
pnpm build
pnpm start
```

打开 <http://127.0.0.1:8000>。`HOATA_PYTHON` 默认指向 `/srv/workspace/venv/bin/python`，也可指向另一套兼容的 Python 环境。

新环境应先创建 CPython 3.14t 的 venv，然后安装 `python/requirements.txt`。该文件固定 JAX 的 commit，以及 jaxlib、libtpu 和 tpuasm 的版本；编译来源捕获依赖这套适配关系。

首版直接执行用户的 Python，仅供可信代码的本地使用。服务固定监听 `127.0.0.1`，检查 Host 与浏览器 Origin；临时进程、超时和输出限制不构成沙箱。本版不包含 Docker、公网部署或域名配置。

## 编写代码

```python
import jax
import jax.numpy as jnp
import hoata

def square(x: jax.Array) -> jax.Array:
    return x * x

hoata.compile(square, jax.ShapeDtypeStruct((8, 128), jnp.float32))
```

网页执行单个 `kernel.py`；每个脚本调用一次 `hoata.compile`。提供的 `clamp`、`matmul`、`top_k` 和 `double_buffer` 示例使用 `pl.kernel`，通过 CPU 上的 JAX/libtpu 编译链路生成所选目标的真实机器映像，不执行 TPU 程序，也不使用 Pallas interpret mode；其中 `top_k` 来自 tpuasm 的复杂样例，展示跨两个 128-lane tile 的 Top-8 值与索引选择；`double_buffer` 同样来自 tpuasm，用 `pl.loop`、`pl.when` 和辅助函数组成多层调用链，适合查看指令来源的调用方与编译器位置。

- `hoata.compile(fn, *abstract_args, mesh=None, compiler_options=None)`：支持参数 pytree；没有 sharding 的 `ShapeDtypeStruct` 默认分配到目标的第一个离线 device。静态参数通过闭包固定；编译参数可以通过 `compiler_options` 传入。
- `hoata.devices(count=1)`：返回所选参考拓扑中的离线 device，最多四个。显式分片时，用这些设备构造 mesh 和输入 sharding，并把 mesh 传给 `compile`。
- v4 使用 tpuasm 捕获的 `v4_2x2x1_megacore.topology`，每个 device 为一颗 Megacore 芯片；v6e 使用 `v6e:2x2`。网页展示的是 TensorCore 指令。

编译成功后默认展示 executable 元数据中 Pallas custom-call symbol 给出的函数区间，包括其中没有来源的指令；函数区间之间的空洞和范围外代码被折叠。“视图”选择器可以切回完整程序。这里的 kernel 视图只是对完整、可逆清单的折叠，不另造一份不能独立回灌的片段。下载始终保存完整程序清单。程序包含多个映像时可切换；函数选择器定位其首条指令，不把不同函数合并。汇编左侧为十六进制 bundle PC，表示映像中的 bundle 序号，不是设备 IMEM 地址。

源码行的颜色对应有关联的指令；融合指令可能对应多行，点击后在双栏中直接高亮全部关联。关联分两级：指令直接对应其调用链中最内层的源码帧（以及编译器 `loc(...)` 给出的位置），高亮到具体列；调用链外层的帧（例如 `pl.loop`、`pl.when` 的装饰器行或辅助函数的调用处）只作为调用方，以斜纹显示。因此点击 `@pl.loop` 一行时，实线高亮循环控制指令，斜纹标出循环体中经由它调用的指令。汇编每条有来源的指令行尾以暗色标出源码行号和第一个 primitive，斜体表示只有编译器位置。点击汇编指令后，底部的“指令来源”卡片逐条列出调用链：最内层帧在上并附 primitive 与 LLO ordinal，调用方逐级缩进，编译器位置和编译器说明（如 `smod.u32 w/div 2`）各占一行；点击任一帧跳到源码。诊断和编译器输出仅在有内容时提供展开入口，编译错误自动展开。未知来源保持未知；普通 JAX 或部分 MXU 指令缺少 Pallas 来源时，汇编仍完整展示。不把缺失映射解释为“编译器生成”。

修改源码或目标后，旧结果标记为过期，暂停源码联动。编译成功后恢复。页面记住本地草稿、目标和语言；下载按钮分别导出当前源码和编译结果。没有服务端作品库或分享链接。

## 实现

- **前端：** TypeScript、React、Tailwind CSS 与 Monaco。Monaco 虚拟化长代码列表；TPU token 规则参考 vscode-tpuasm。布局结合 Godbolt 的双栏编辑体验和 Profiling Explorer 的中性深色面板、圆角控件与红色主要操作。
- **Rust/WASM：** `hoata-engine` 在专用 Web Worker 中按 tpuasm 的结构化坐标建立源码→指令、指令→源码索引，执行关联查询、函数范围定位和汇编搜索。源码索引读取 tpuasm 归一后的 `source_frames`，kernel 视图读取编译器的 `functions[*].ranges`，包括区间中没有来源的指令；前端不按源码覆盖率裁切函数，也不自行推断指令来源。
- **Rust 服务：** Axum/Tokio 提供 API 和静态资源，管理单任务并发、取消、120 秒默认超时以及进程组回收。每个任务有独立工作目录，结束后清理；取消也回收 native helper 的编译子进程。
- **Python 适配层：** 仅负责现有 JAX/tpuasm API 调用。禁用 JAX 编译缓存，在来源捕获上下文中编译；用 tpuasm 导出 `exact` 汇编并解析汇编行坐标。它不提供 HTTP 服务。

`python/hoata/v4_2x2x1_megacore.topology` 是 192 字节的硬件拓扑描述，来自 tpuasm 的实机捕获，不是编译中间产物。其他编译结果和 Wasm 产物均不纳入 Git。

API：

| 接口 | 用途 |
| --- | --- |
| `GET /api/config` | 目标、示例源码、已安装工具链版本与限制 |
| `POST /api/compile` | JSON `{id, source, target}`；返回 `ok`、`programs`、`diagnostics`、`versions`、`elapsed_ms` 和有界日志 |
| `DELETE /api/compile/{id}` | 取消相同 ID 的当前任务 |

每个 program 包含 `.tpuasm` 文本、tpuasm 解析出的逐指令行/PC/槽、bundle 坐标及来源映射。来源映射沿用 tpuasm schema v3 的字段，源码关联消费逐槽 `source_frames`，kernel 视图消费 `functions[*].ranges`；`source_kind` 保留 `captured`、`compiler_location` 或 `unknown`。引擎合并显示区间时只合并重叠或相邻区间，保留编译器给出的空洞。传往浏览器的 64 位 module/program/compilation ID 转为十进制字符串，用户文件路径归一化为 `kernel.py`。

请求源码最多 256 KiB，结果最多 8 MiB，stdout/stderr 各保留前 128 KiB。任务繁忙返回 429；普通源码编译错误以 `ok: false` 和源码诊断返回。`HOATA_TIMEOUT_SECONDS` 调整超时。`HOATA_PORT` 调整正式服务端口，开发代理仍默认使用 8000。服务日志不记录提交的源码，浏览器收到的原始编译器输出保持原文。

## 检查

```sh
pnpm build
pnpm check
cargo fmt --all --check
MYPYPATH=python:../tpuasm/src "$HOATA_PYTHON" -m mypy python examples
```

真实链路验证包括三个示例分别编译为 v4/v6e；源码融合映射、逐指令坐标与 tpuasm 输出一致；浏览器中检查三语言、RTL、双向定位、查找、过期结果、草稿、下载、取消与错误。浏览器截图及临时编译结果存放在 `/tmp`，不进入仓库。
