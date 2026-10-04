import jax
from jax import Ref
from jax._src.pallas.mosaic.primitives import AsyncCopyDescriptor
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import hoata

tc = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

@pl.kernel(
    out_type=jax.ShapeDtypeStruct((32, 128), jnp.float32),
    mesh=tc,
    scratch_types=(
        pltpu.VMEM((2, 8, 128), jnp.float32),
        pltpu.VMEM((2, 8, 128), jnp.float32),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
    ),
    name='double_buffer',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def double_buffer(x_hbm: Ref, out_hbm: Ref, x_vmem: Ref, out_vmem: Ref, load_sem: Ref, store_sem: Ref) -> None:
    def load(index: int | jax.Array, buffer: int | jax.Array) -> AsyncCopyDescriptor:
        return pltpu.make_async_copy(x_hbm.at[pl.ds(index * 8, 8), :], x_vmem.at[buffer], load_sem.at[buffer])

    def store(index: int | jax.Array, buffer: int | jax.Array) -> AsyncCopyDescriptor:
        return pltpu.make_async_copy(out_vmem.at[buffer], out_hbm.at[pl.ds(index * 8, 8), :], store_sem.at[buffer])

    # Prefetch block 0; each step computes one buffer while the other receives the next block.
    load(0, 0).start()

    @pl.loop(0, 4)
    def body(index: jax.Array) -> None:
        buffer = index % 2
        load(index, buffer).wait()

        @pl.when(index < 3)
        def prefetch() -> None:
            load(index + 1, 1 - buffer).start()

        @pl.when(index >= 2)
        def release_output() -> None:
            store(index - 2, buffer).wait()

        out_vmem[buffer] = x_vmem[buffer] * 2.0 + 1.0
        store(index, buffer).start()

    # The last two write-backs must finish before the kernel exits.
    store(2, 0).wait()
    store(3, 1).wait()

hoata.compile(double_buffer, jax.ShapeDtypeStruct((32, 128), jnp.float32))
