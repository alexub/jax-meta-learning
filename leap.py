"""Simple, extensible Jax implementation of Leap.

The algorithm is from Transferring Knowledge Across Learning Processes,
by Flennerhag, Moreno, Lawrence, and Damianou, ICLR 2019.
https://arxiv.org/abs/1812.01054

I used the author's code as reference: https://github.com/amzn/metalearn-leap

This code assumes you've created your model with Flax, but it should be easy to modify
for other frameworks.
"""


from functools import partial
from collections import namedtuple

import jax
import jax.numpy as np

import flax


# LeapDef contains algorithm-level parameters.
# Think of constructing LeapDef as akin to passing args to the __init__ of a Trainer
# class, except here we're using an (immutable) namedtuple instead of a class instance,
# and functions rather than class instance methods, to be more Jax-like
# and to be more obvious about the functions having no side effects.
LeapDef = namedtuple(
    "LeapDef",
    [
        "make_inner_opt",  # fn: Flax model -> Flax optimizer
        "make_task_loss_fn",  # fn: PRNGKey -> loss_fn which defines a task
        # note Leap does not have a concept of inner-loop vs outer-loop (meta) loss
        # loss_fn may or may not be stochastic (i.e. use or do not use the PRNGKey)
        # loss_fn is a fn: PRNGkey, model -> loss
        "inner_steps",  # int: num inner-loop optimization steps
        "n_batch_tasks",  # int: number of 'tasks' in a batch for a meta-train step
        "norm",  # bool: whether to normalize Leap gradients
        "loss_in_distance",  # bool: whether to use loss to compute distances on task manifold
        "stabilize",  # bool: whether to use a stabilizer for Leap grads
    ],
)  # See the paper and reference code to understand the last three args.


def leap_inner_step(leap_def, key, opt, loss_fn, meta_grad_accum):
    """Inner step of Leap single-task rollout.

    This function does two things:
        (i) do a step of the inner optimization (e.g. SGD)
        (ii) update a meta-grad accumulator to (iteratively) compute the Leap gradient

    Args:
        maml_def: MamlDef namedtuple
        key: PRNGKey (Jax random key)
        opt: Flax optimizer (which carries the model in opt.target)
        loss_fn: the loss function which defines a given task
        meta_grad_accum: the running accumulated Leap gradient

    Returns:
        new_opt: a new Flax optimizer (with updated model and optimizer accumulators)
        loss: the inner-loop loss value at this step
    """

    k1, k2 = jax.random.split(key, 2)

    # differentiate w.r.t arg1 because args to loss_fn are (key, model)
    loss_and_grad_fn = jax.value_and_grad(loss_fn, argnums=1)

    loss, grad = loss_and_grad_fn(k1, opt.target)
    new_opt = opt.apply_gradient(grad)

    # inner optimization step done. now we update meta_grad_accum
    new_loss = loss_fn(k2, new_opt.target)

    meta_grad_increment = get_meta_grad_increment(
        leap_def, new_opt.target, opt.target, new_loss, loss, grad
    )

    meta_grad_accum = jax.tree_util.tree_multimap(
        lambda x, y: x + y, meta_grad_accum, meta_grad_increment
    )
    return new_opt, meta_grad_accum, new_loss


def single_task_rollout(leap_def, key, initial_model, loss_fn):

    """Roll out meta learned model on one task. Use for both training and deployment.

    Computes the final model, and the per-inner-loop step losses.
    Also accumulates Leap gradients.

    Args:
        leap_def: LeapDef namedtuple
        key: PRNGKey
        initial_model: a Flax model to use as initialization
        loss_fn: the loss fn which defines a task

    Returns:
        final_opt.target: trained model of same type as initial_model
        meta_grad_accum: accumulated Leap gradient w.r.t. the initial_model
        losses: [n_steps + 1] array of inner-loop losses at each inner step
    """
    loss0_key, inner_key = jax.random.split(key, 2)
    inner_keys = jax.random.split(inner_key, leap_def.inner_steps)

    loss0 = loss_fn(loss0_key, initial_model)

    inner_opt = leap_def.make_inner_opt(initial_model)

    meta_grad_accum = jax.tree_util.tree_map(lambda x: x * 0, inner_opt.target)

    def body_fn(carry, key):
        opt, meta_grad_accum = carry
        opt, meta_grad_accum, loss = leap_inner_step(
            leap_def, key, opt, loss_fn, meta_grad_accum
        )
        return (opt, meta_grad_accum), loss

    (final_opt, meta_grad_accum), losses = jax.lax.scan(
        body_fn, (inner_opt, meta_grad_accum), inner_keys, length=leap_def.inner_steps
    )

    losses = np.concatenate((np.array([loss0]), losses))

    return final_opt.target, meta_grad_accum, losses


@partial(jax.jit, static_argnums=0)
def single_task_grad_and_losses(leap_def, key, initial_model):
    """Make the task loss, do the rollout, and return the Leap gradient and losses.

    Args:
        key: terminal PRNGKey
        initial_model: a Flax model

    Returns:
        grads: gradient, of same type/treedef as the Flax model
        losses: [n_steps] array of losses at each inner step
    """
    loss_fn_key, rollout_key = jax.random.split(key, 2)
    loss_fn = leap_def.make_task_loss_fn(loss_fn_key)
    _, meta_grad, losses = single_task_rollout(
        leap_def, rollout_key, initial_model, loss_fn
    )
    return meta_grad, losses


@partial(jax.jit, static_argnums=0)
def multitask_rollout(leap_def, key, initial_model):
    """Roll out meta learner across multiple tasks, collecting Leap gradients.

    Args:
        key: terminal PRNGKey
        initial_model: a Flax model

    Returns:
        grads: gradient, of same type/treedef as the Flax model
        losses: [n_tasks, n_steps] array of losses at each inner step of each task
    """
    keys = jax.random.split(key, leap_def.n_batch_tasks)
    grads, losses = jax.vmap(single_task_grad_and_losses, in_axes=(None, 0, None))(
        leap_def, keys, initial_model
    )
    grads = jax.tree_util.tree_map(lambda g: g.mean(axis=0), grads)
    return grads, losses


@partial(jax.jit, static_argnums=0)
def get_meta_grad_increment(leap_def, new_model, model, new_loss, loss, grad):
    """Get Leap meta-grad increment. See paper/author code for details."""
    d_loss = new_loss - loss
    if leap_def.stabilize:
        d_loss = -np.abs(d_loss)

    if leap_def.norm:
        norm = compute_global_norm(leap_def, new_model, model, d_loss)
    else:
        norm = 1.0

    meta_grad_increment = jax.tree_util.tree_multimap(
        lambda x, y: x - y, model, new_model
    )

    if leap_def.loss_in_distance:
        meta_grad_increment = jax.tree_util.tree_multimap(
            lambda x, y: x - d_loss * y, meta_grad_increment, grad
        )

    meta_grad_increment = jax.tree_util.tree_map(
        lambda x: x / norm, meta_grad_increment
    )

    return meta_grad_increment


def compute_global_norm(leap_def, new_model, old_model, d_loss):
    """Compute norm within task manifold. See paper for details."""
    model_sq = jax.tree_util.tree_multimap(
        lambda x, y: np.sum((x - y) ** 2), new_model, old_model
    )
    sum_sq = jax.tree_util.tree_reduce(lambda x, y: x + y, model_sq)
    if leap_def.loss_in_distance:
        sum_sq = sum_sq + d_loss ** 2

    norm = np.sqrt(sum_sq)
    return norm


def run_sinusoid():
    """Test the code on a simple sinusiod problem, a la MAML."""

    # Simple definition of an MLP with Swish activations
    @flax.nn.module
    def MLP(x):
        for _ in range(3):
            x = flax.nn.Dense(x, 64)
            x = flax.nn.swish(x)
        x = flax.nn.Dense(x, 1)
        return x

    # Create a base model and the meta-model optimizer
    _, initial_params = MLP.init_by_shape(jax.random.PRNGKey(0), [((1, 1), np.float32)])

    model = flax.nn.Model(MLP, initial_params)
    meta_opt = flax.optim.Adam(learning_rate=1e-3).create(model)

    # Create helper functions needed for the LeapDef

    # Sinusoid loss with different phase
    # For Leap, we demonstrate having a stochastic loss fn
    def sinusoid_loss_fn(key, model, phase):
        x = jax.random.uniform(key, shape=(32, 1))
        y = np.sin(x + phase)
        yhat = model(x)
        return np.mean((y - yhat) ** 2)

    # Fn which makes a loss fn for a task (by sampling a phase)
    def make_task_loss_fn(key):
        phase = jax.random.uniform(key, shape=(1, 1), minval=0.0, maxval=2.0 * np.pi)
        return lambda key, model: sinusoid_loss_fn(key, model, phase)

    # Fn to make an inner optimizer from an initial model
    make_inner_opt = flax.optim.Momentum(learning_rate=0.1, beta=0.0).create

    # Specify the Leap algorithm-level parameters
    leap_def = LeapDef(
        make_inner_opt=make_inner_opt,
        make_task_loss_fn=make_task_loss_fn,
        inner_steps=10,
        n_batch_tasks=32,
        norm=True,
        loss_in_distance=True,
        stabilize=True,
    )

    # Run the meta-train loop
    key = jax.random.PRNGKey(1)
    for i in range(1000):
        key, subkey = jax.random.split(key)
        grad, losses = multitask_rollout(leap_def, subkey, meta_opt.target)
        print(
            "meta-step {}, per-inner-step avg losses {}".format(
                i, np.mean(losses, axis=0)
            )
        )
        meta_opt = meta_opt.apply_gradient(grad)


if __name__ == "__main__":
    run_sinusoid()
