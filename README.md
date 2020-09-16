# jax-meta-learning
Simple, flexible implementations of some meta-learning algorithms in Jax.

The goal is that you should be able to just specify hyperparameters and "drop in" your choice of model, gradient-based optimizers, and distribution over tasks, and these implementations should work out-of-the-box with minimal code overhead, whether your tasks are classification, regression, reinforcement learning, or something weird and wonderful.

The caveats are that you need to use Flax models/optimizers (or write classes with similar API), and your "tasks" must be written as functions which map from a random seed and a model to a scalar loss. The MAML implementation also does not include improvements added by subsequent papers, such as trainable inner-loop learning rates.

# Requires:
- Jax, https://github.com/google/jax
- Flax, https://github.com/google/flax 

# Done
- MAML implementation, https://arxiv.org/abs/1703.03400
- LEAP implementation, https://arxiv.org/abs/1812.01054
- Examples on toy sinusoid problem 

# Todo
- Usage guide, incl. adding code snippets to README
- Examples on Omniglot 
- Migrate examples from "maml.py" etc to their own folder
