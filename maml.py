"""Simple, extensible Jax implementation of MAML.

The algorithm is from Model-Agnostic Meta Learning for Fast Adaptation of Deep
Networks, by Finn, Abbeel, and Levine, ICML 2017. https://arxiv.org/abs/1703.03400

This code assumes you've created your model with Flax, but it should be easy to modify
for other frameworks.
"""

from functools import partial
from collections import namedtuple

import jax
import jax.numpy as np

import flax


# MamlDef contains algorithm-level parameters.
# Think of constructing MamlDef as akin to passing args to the __init__ of a Trainer
# class, except here we're using an (immutable) namedtuple instead of a class instance,
# and functions rather than class instance methods, to be more Jax-like
# and to be more obvious about the functions having no side effects.
MamlDef = namedtuple(
    "MamlDef",
    [
        "make_inner_opt",  # fn: Flax model -> Flax optimizer
        "make_task_loss_fns",  # fn: PRNGKey -> (inner_loss, outer_loss) which define a task
        # both inner_loss and outer_loss are fn: PRNGkey, model -> loss
        # (e.g. for few-shot classification, inner loss is the loss on support data
        #  and outer loss is the loss on query data)
        "inner_steps",  # int: num inner-loop optimization steps
        "n_batch_tasks",  # int: number of 'tasks' in a batch for a meta-train step
    ],
)


def maml_inner_step(key, opt, inner_loss_fn):
    """Inner step of MAML single-task rollout. It's just SGD or some other optim.

    Args:
        maml_def: MamlDef namedtuple
        key: PRNGKey (Jax random key)
        opt: Flax optimizer (which carries the model in opt.target)
        inner_loss_fn: the loss fn defining a given task within the task distribution

    Returns:
        new_opt: a new Flax optimizer (with updated model and optimizer accumulators)
        loss: the inner-loop loss at this step
    """

    # differentiate w.r.t arg1 because args to inner loss are (key, model)
    loss_and_grad_fn = jax.value_and_grad(inner_loss_fn, argnums=1)

    loss, grad = loss_and_grad_fn(key, opt.target)
    new_opt = opt.apply_gradient(grad)
    return new_opt, loss


def single_task_rollout(maml_def, rollout_key, initial_model, inner_loss_fn):
    """Roll out meta learned model on one task. Use for both training and deployment.

    Computes the final model, and the per-inner-loop step losses.

    Args:
        maml_def: MamlDef namedtuple
        key: PRNGKey
        initial_model: a Flax model to use as initialization
        inner_loss_fn: the loss fn with which to train the model in the inner loop

    Returns:
        final_opt.target: trained model of same type as initial_model
        meta_grad: accumulated MAML gradient w.r.t. the initial_model
        losses: [n_steps + 1] array of inner-loop losses at each inner step
        meta_loss: the outer loss, e.g. the "test loss" on the given task
    """

    inner_key, loss_final_key = jax.random.split(rollout_key, 2)
    inner_keys = jax.random.split(inner_key, maml_def.inner_steps)

    body_fn = lambda opt, key: maml_inner_step(key, opt, inner_loss_fn)

    inner_opt = maml_def.make_inner_opt(initial_model)

    # Loop over the body_fn for each inner_step, carrying opt and stacking losses
    final_opt, losses = jax.lax.scan(
        body_fn, inner_opt, inner_keys, length=maml_def.inner_steps
    )

    # Cat the final loss to loss array (to have losses before and after each grad step)
    loss_final = inner_loss_fn(loss_final_key, final_opt.target)
    losses = np.concatenate((losses, np.array([loss_final])))

    return final_opt.target, losses


@partial(jax.jit, static_argnums=0)
def single_task_grad_and_losses(maml_def, key, initial_model):
    """Make the task losses, do rollout, and compute the meta-gradient"""
    task_key, rollout_key, outer_loss_key = jax.random.split(key, 3)
    inner_loss_fn, outer_loss_fn = maml_def.make_task_loss_fns(task_key)

    def task_rollout_and_eval(initial_model):
        final_model, losses = single_task_rollout(
            maml_def, rollout_key, initial_model, inner_loss_fn
        )
        outer_loss = outer_loss_fn(outer_loss_key, final_model)
        return outer_loss, losses

    (meta_loss, losses), meta_grad = jax.value_and_grad(
        task_rollout_and_eval, has_aux=True
    )(initial_model)

    return meta_grad, losses, meta_loss


@partial(jax.jit, static_argnums=0)
def multi_task_grad_and_losses(maml_def, key, initial_model):
    """Roll out meta learner across *multiple* tasks, collecting MAML gradients.

    Args:
        maml_def: MamlDef namedtuple
        key: terminal PRNGKey
        initial_model: a Flax model

    Returns:
        grads: gradient, of same type/treedef as the Flax model
        losses: [n_tasks, n_steps] array of losses at each inner step of each task
    """
    keys = jax.random.split(key, maml_def.n_batch_tasks)

    # Get the grads and losses for each task in parallel
    grads, losses, meta_losses = jax.vmap(
        single_task_grad_and_losses, in_axes=(None, 0, None)
    )(maml_def, keys, initial_model)

    # Average the gradient over the tasks
    grads = jax.tree_util.tree_map(lambda g: g.mean(axis=0), grads)

    return grads, losses, meta_losses


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

    # Create helper functions required by the MamlDef

    # For MAML, we demonstrate having a deterministic inner loss, and training to
    # generalize to the outer loss

    # Sinusoid loss with different phase
    def sinusoid_loss_fn(model, x, phase):
        y = np.sin(x + phase)
        yhat = model(x)
        return np.mean((y - yhat) ** 2)

    # Fn which makes inner/outer loss fns for a task (by sampling a phase and data)
    def make_task_loss_fns(key):
        k1, k2, k3 = jax.random.split(key, 3)
        x_train = jax.random.uniform(k1, shape=(32, 1))
        x_test = jax.random.uniform(k2, shape=(32, 1))
        phase = jax.random.uniform(k3, shape=(1, 1), minval=0.0, maxval=2.0 * np.pi)

        # Here we show using a deterministic, but different, inner loss and outer loss
        # similar to few-shot learning
        inner_loss = lambda key, model: sinusoid_loss_fn(model, x_train, phase)
        outer_loss = lambda key, model: sinusoid_loss_fn(model, x_test, phase)

        return inner_loss, outer_loss

    # Fn to make an inner optimizer from an initial model
    make_inner_opt = flax.optim.Momentum(learning_rate=0.1, beta=0.0).create

    # Specify the MAML algorithm-level parameters
    maml_def = MamlDef(
        make_inner_opt=make_inner_opt,
        make_task_loss_fns=make_task_loss_fns,
        inner_steps=10,
        n_batch_tasks=32,
    )

    # Run the meta-train loop
    key = jax.random.PRNGKey(1)
    for i in range(1000):
        key, subkey = jax.random.split(key)
        grad, losses, meta_losses = multi_task_grad_and_losses(
            maml_def, subkey, meta_opt.target
        )
        print(
            "meta-step {}, meta_loss {}, per-inner-step avg losses {}".format(
                i, np.mean(meta_losses), np.mean(losses, axis=0)
            )
        )
        meta_opt = meta_opt.apply_gradient(grad)


if __name__ == "__main__":
    run_sinusoid()
