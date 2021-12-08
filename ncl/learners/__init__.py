from .base import Base
from .projected import (Orthogonal, LOrthogonal, ROrthogonal)
from .laplace import (KFAC, KFACAdam, NCL, LaplaceOWM, LaplaceDOWM)

__all__ = [
    'available',
    'build_learner_mapping',
    'Base',
    'Orthogonal',
    'LOrthogonal',
    'ROrthogonal',
    'KFAC',
    'KFACAdam',
    'LaplaceOWM',
    'LaplaceDOWM',
]

available = [
    'base', 'orthogonal', 'lorthogonal', 'rorthogonal', 'ncl', 'kfac',
    'kfac-adam', 'laplace-owm', 'laplace-dowm'
]


def build_learner_mapping(task, save_dir, l2_w, l2_h, projection_alpha, phi):
    learner_mapping = {
        "base":
        lambda hp, task: Base(
            hp, save_dir, task, phi=phi, l2_w=l2_w, l2_h=l2_h),
        "orthogonal":
        lambda hp, task: Orthogonal(hp,
                                    save_dir,
                                    task,
                                    phi=phi,
                                    projection_alpha=projection_alpha,
                                    l2_w=l2_w,
                                    l2_h=l2_h),
        "lorthogonal":
        lambda hp, task: LOrthogonal(hp,
                                     save_dir,
                                     task,
                                     phi=phi,
                                     projection_alpha=projection_alpha,
                                     l2_w=l2_w,
                                     l2_h=l2_h),
        "rorthogonal":
        lambda hp, task: ROrthogonal(hp,
                                     save_dir,
                                     task,
                                     phi=phi,
                                     projection_alpha=projection_alpha,
                                     l2_w=l2_w,
                                     l2_h=l2_h),
        "ncl":
        lambda hp, task: NCL(
            hp, save_dir, task, phi=phi, projection_alpha=projection_alpha),
        "laplace-owm":
        lambda hp, task: LaplaceOWM(
            hp, save_dir, task, phi=phi, projection_alpha=projection_alpha),
        "laplace-dowm":
        lambda hp, task: LaplaceDOWM(
            hp, save_dir, task, phi=phi, projection_alpha=projection_alpha),
        "kfac":
        lambda hp, task: KFAC(hp, save_dir, task, phi=phi),
        "kfac-adam":
        lambda hp, task: KFACAdam(hp, save_dir, task, phi=phi),
    }
    return learner_mapping
