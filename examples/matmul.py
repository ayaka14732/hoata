import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import hoata

tile = (128, 128)
tc = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

@pl.kernel(
    out_type=jax.ShapeDtypeStruct(tile, jnp.float32),
    mesh=tc,
    scratch_types=(pltpu.VMEM(tile, jnp.bfloat16), pltpu.VMEM(tile, jnp.bfloat16), pltpu.VMEM(tile, jnp.float32), pltpu.SemaphoreType.DMA),
    name='matmul',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def matmul(lhs_hbm: Ref, rhs_hbm: Ref, out_hbm: Ref, lhs_vmem: Ref, rhs_vmem: Ref, out_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(lhs_hbm, lhs_vmem, sem).wait()
    pltpu.async_copy(rhs_hbm, rhs_vmem, sem).wait()
    out_vmem[...] = jnp.dot(lhs_vmem[...], rhs_vmem[...], preferred_element_type=jnp.float32)
    pltpu.async_copy(out_vmem, out_hbm, sem).wait()

spec = jax.ShapeDtypeStruct(tile, jnp.bfloat16)
hoata.compile(matmul, spec, spec)
