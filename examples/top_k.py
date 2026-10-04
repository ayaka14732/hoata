import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import hoata

tile = (8, 256)
output = (8, 8)
tc = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

@pl.kernel(
    out_type=(jax.ShapeDtypeStruct(output, jnp.float32), jax.ShapeDtypeStruct(output, jnp.int32)),
    mesh=tc,
    scratch_types=(pltpu.VMEM(tile, jnp.float32), pltpu.VMEM(output, jnp.float32), pltpu.VMEM(output, jnp.int32), pltpu.SemaphoreType.DMA),
    name='top_k',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def top_k(x_hbm: Ref, values_hbm: Ref, indices_hbm: Ref, x_vmem: Ref, values_vmem: Ref, indices_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()

    # Track the eight largest values and their indices across two 128-lane tiles.
    values, indices = jax.lax.top_k(x_vmem[...], 8, is_stable=False)
    values_vmem[...] = values
    indices_vmem[...] = indices

    pltpu.async_copy(values_vmem, values_hbm, sem).wait()
    pltpu.async_copy(indices_vmem, indices_hbm, sem).wait()

hoata.compile(top_k, jax.ShapeDtypeStruct(tile, jnp.float32))
