import jax
import numpy as onp
import jax
import jax.numpy as np
from jax import jit
from .dataset import create_generate_trials

fudge = 1


def default_hp(n_hidden,
               train_batch_size,
               eval_batch_size,
               seed=42,
               max_steps=1_000_000,
               display_every=1000,
               save_every=1000,
               head='single',
               learning_rate=1e-3,
               mass=0.9,
               ruleset=['0_1', '2_3', '4_5', '6_7', '8_9'],
               lambd=None,
               data_size=None):
    '''Get a default hp. '''

    if head == 'multi':
        n_output = 30  #Multi head
        n_input = 4  #4 input channels
    else:
        n_output = 2  #single head
        n_input = 4 + len(ruleset)  #4 input channels, n tasks (one-hot)

    hp = {
        # train batch_size
        'train_batch_size': train_batch_size,
        # eval batch_size
        'eval_batch_size': eval_batch_size,
        # Type of loss functions
        'loss_type': 'lsq',
        # discretization time step (ms)
        'dt': 20,
        # discretization time step/time constant
        'tau': 0.2,
        # recurrent noise
        'sigma_rec': 0.05,
        # input noise
        'sigma_x': 1.,  #0.5,
        # number of input units
        'n_input': n_input,
        # number of output units
        'n_output': n_output,
        # number of hidden units
        'n_hidden': n_hidden,
        # data size
        'data_size': (50000 if data_size is None else data_size),
        # max steps
        'max_steps': max_steps,
        # random seed
        'seed': seed,
        # display_every
        'display_every': display_every,
        # save_every
        'save_every': save_every,
        #multi head or single head
        'smnist_head_type': head,
        #length of reporting period
        'npad': 5,
        # learning rate
        'learning_rate': learning_rate,
        # mass
        'mass': mass,
        #task indices
        'task_ind': dict(zip(ruleset, range(len(ruleset)))),
        #regularization strength c.f. EWC (the proper Bayesian treatment has lambda=1)
        'lambda': (1 if lambd is None else lambd),
    }
    return hp


def performance(y_hat, trial):
    """

    Args:
      y_hat: Actual output. Numpy array (Batch, Time, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Batch, Time)
      Ts: length of each batch (Batch,)

    Returns:
      perf: Numpy array (Batch,)
    """
    y_loc = trial['y_loc']
    Ts = trial['tdim']
    y_loc = y_loc[Ts - 1]
    y_hat = y_hat[Ts - 1, :]

    y_hat_loc = np.argmax(y_hat, axis=-1)  # final output for each batch
    perf = (np.abs((y_hat_loc - y_loc)) <= 0.1)

    return perf


generate_trials = create_generate_trials()


@jit
def negloglik(ypreds, ytarget, c_mask, mask, Rinv, jitter=1e-6):
    """ypreds is time x nout
    y is logits"""

    inner = mask[..., None] * c_mask * ytarget * (
        ypreds - jax.scipy.special.logsumexp(ypreds, -1, keepdims=True))
    ll = -np.sum(inner, axis=-1)  #sum over classes
    return np.sum(ll)


@jit
def sample_target(key, y, c_mask, sigma_mask, mask, R):
    """y are un-normalized logits"""
    targets = jax.random.categorical(key, y, axis=-1)  #sample indices
    ytarget = jax.nn.one_hot(targets, y.shape[-1])  #convert to one-hot
    return sigma_mask * mask[..., None] * c_mask * ytarget
