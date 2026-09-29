import jax
import jax.numpy as jnp
import hoata

# A plain JAX function, compiled for the selected TPU.
def square(x: jax.Array) -> jax.Array:
    return x * x

hoata.compile(square, jax.ShapeDtypeStruct((8, 128), jnp.float32))
