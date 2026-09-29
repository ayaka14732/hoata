import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import hoata

tile = (8, 128)
tc = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

@pl.kernel(
    out_type=jax.ShapeDtypeStruct(tile, jnp.float32),
    mesh=tc,
    scratch_types=(pltpu.VMEM(tile, jnp.float32), pltpu.SemaphoreType.DMA),
    name='clamp',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def clamp(x_hbm: Ref, out_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
    # Load the tile into TC VMEM.
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()

    # Two operations can become one machine instruction.
    lower = jnp.maximum(x_vmem[...], -1.0)
    x_vmem[...] = jnp.minimum(lower, 1.0)

    pltpu.async_copy(x_vmem, out_hbm, sem).wait()

hoata.compile(clamp, jax.ShapeDtypeStruct(tile, jnp.float32))
