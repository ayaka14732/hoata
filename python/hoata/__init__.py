"""Hoata 的 JAX/Pallas 离线编译入口；由 Rust 服务在独立进程中调用。"""
from __future__ import annotations

from collections.abc import Callable
import functools
import os
from pathlib import Path
from typing import Any, cast

import jax
from jax.experimental import topologies
from jaxlib.xla_client import LoadedExecutable
import numpy as np

from tpuasm import compiler_source_mapping, executable_programs, executable_source_maps, format_assembly, parse_assembly

_result: dict[str, Any] | None = None
_calls = 0

@functools.cache
def _topology() -> list[Any]:
    target = os.environ['HOATA_TARGET']
    if target == 'tpu-v6e-tc':
        return list(topologies.get_topology_desc(platform='tpu', topology_name='v6e:2x2').devices)
    if target != 'tpu-v4-tc':
        raise ValueError(f'Unsupported target: {target}')
    topologies.get_topology_desc(platform='tpu', topology_name='v4:2x2x1')
    return list(topologies.TopologyDescription.deserialize(Path(__file__).with_name('v4_2x2x1_megacore.topology').read_bytes()).devices)

def devices(count: int = 1) -> list[Any]:
    """返回所选目标的离线设备；参考拓扑最多提供四个 device。"""
    available = _topology()
    if not 1 <= count <= len(available):
        raise ValueError(f'Expected 1..{len(available)} devices, got {count}')
    return available[:count]

def _browser_values(value: Any, source: str, key: str = '') -> Any:
    """将来源路径归一化；64 位身份作为字符串传给 JavaScript。"""
    if isinstance(value, dict):
        return {name: _browser_values(item, source, name) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_browser_values(item, source) for item in value]
    if key == 'path' and value == source:
        return 'kernel.py'
    if key in ('compilation_id', 'metadata_program_id', 'hlo_module_id') and value is not None:
        return str(value)
    return value

def compile(fn: Callable[..., Any], *abstract_args: Any, mesh: jax.sharding.Mesh | None = None, compiler_options: dict[str, Any] | None = None) -> None:
    """编译一次函数并收集所有机器程序；数组输入使用 ShapeDtypeStruct。"""
    global _result, _calls
    _calls += 1
    if _calls != 1:
        raise ValueError('Call hoata.compile exactly once per script.')
    target = os.environ['HOATA_TARGET']
    source = os.environ['HOATA_SOURCE']
    jax.config.update('jax_enable_compilation_cache', False)
    jax.clear_caches()
    if mesh is None:
        mesh = jax.sharding.Mesh(np.array(devices()), ('device',))
    default_sharding = jax.sharding.SingleDeviceSharding(mesh.devices.flat[0])

    def with_sharding(argument: Any) -> Any:
        if isinstance(argument, jax.ShapeDtypeStruct) and argument.sharding is None:
            return jax.ShapeDtypeStruct(argument.shape, argument.dtype, sharding=default_sharding, weak_type=argument.weak_type)
        return argument

    arguments = jax.tree.map(with_sharding, abstract_args)
    options = {'xla_msa_enable': 'false', 'xla_tpu_vmem_scavenging_mode': 'NONE'}
    if target == 'tpu-v4-tc':
        options['xla_mosaic_unsafe_allow_multicore_remote_dma'] = 'true'
    options.update(compiler_options or {})
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh), compiler_source_mapping():
        compiled = jax.jit(fn, compiler_options=options).lower(*arguments).compile()
    serialized = bytes(cast(LoadedExecutable, compiled.runtime_executable()).serialize())
    programs = []
    maps = executable_source_maps(serialized)
    for (record, index, image), mapping in zip(executable_programs(serialized), maps, strict=True):
        assembly = format_assembly(image, encoding='exact', target=target)
        parsed = parse_assembly(assembly)
        instructions = [
            {'pc': pc, 'slot': instruction.slot, 'line': instruction.location.line, 'mnemonic': instruction.mnemonic}
            for pc, bundle in enumerate(parsed.bundles) for instruction in bundle.instructions
        ]
        programs.append({
            'id': f'{record}:{index}',
            'target': target,
            'assembly': assembly,
            'instructions': instructions,
            'bundles': [{'pc': pc, 'line': bundle.location.line} for pc, bundle in enumerate(parsed.bundles)],
            'source_map': _browser_values(mapping.to_dict(), source),
        })
    _result = {'programs': programs}

__all__ = ['compile', 'devices']
