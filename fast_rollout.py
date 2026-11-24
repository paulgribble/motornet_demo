"""A flat, fused re-implementation of my_utils.run_episode for ONE specific configuration:

    ExperimentEnv + RigidTendonArm26 + RigidTendonHillMuscle + Euler (1 ministep)
    + ModularPolicyGRU without inter-module (connectivity) delays, output_delay 0 or 1.

It computes the same thing as run_episode() (same order of operations, same noise
distributions entering at the same places, same returned dict) but as one flat step with
far fewer tensor operations, no dicts / python-list delay buffers / numpy noise, and it can
optionally run the rollout as torch.compile'd K-step chunks.

    from fast_rollout import run_episode_fast, FAST_PRESET
    data = run_episode_fast(env, task, policy, batch_size, n_t, device, k=0)                  # eager
    data = run_episode_fast(env, task, policy, batch_size, n_t, device, k=0, **FAST_PRESET)   # compiled

Eager mode is NOT faster than the stock code (it is for checking / as a reference); the speed-up
comes from compiling K-step segments (FAST_PRESET: ~3.5x on the go_3module.py workload, after about
1.5 min of compilation, which is repeated for every new batch size / force-field on-off).
Run check_fast_rollout.py after changing the model code, to confirm the two still agree.
Not covered: connectivity delays between modules, output_delay > 1, other effectors / muscles /
skeletons / integrators, env.obs_noise, action frame stacking (a NotImplementedError is raised).

All physical constants are read from the env / effector / muscle / skeleton objects, and the
network weights are read from the policy object's parameters on every call (so saved weights,
masks and gradient hooks keep working).

Timeline reproduced from the stock code (s_t = arm state before step t, h_t = hidden before step t):
    obs_t    = [vision, proprioception] with
               vision  = fingertip(s_{t-vd}) + noise   (vd = env.vision_delay, in steps; while t < vd the
                         network sees independent noisy samples of the initial state, fingertip(s_0))
               proprio = [muscle_len/l0_ce, muscle_vel/vmax](s_{t-pd}) + noise   (pd = env.proprioception_delay)
               (noise is independent for every trial, channel and sample; a sample is drawn once,
               when the state is sensed, not when it reaches the network)
    u_t      = sigmoid(Y h_t + bY)                 (output_delay=1: read out from h BEFORE the update)
    h_{t+1}  = GRU([obs_t, inp_t], h_t)            (hidden noise: independent per trial, unit and step)
    a_t      = clip(u_t + action_noise)
    s_{t+1}  : activation' = clip(act + dt * (a - act)/tau)
               muscle state (length, velocity, force) from activation' and the geometry of s_t
               joint: vel' = vel + dt*acc(s_t, force_t), pos' = pos + dt*vel, then limits
    logged at index t: cartesian(s_{t+1}), joint(s_{t+1}), h_{t+1}, u_t, activation_{t+1}, force_{t+1}
"""

import torch as th
import torch.nn.functional as F

import motornet as mn
from my_policy import ModularPolicyGRU


class _Consts:
    """Constants read once from the env / effector objects."""

    def __init__(self, env):
        eff, mus, sk = env.effector, env.effector.muscle, env.effector.skeleton
        if not isinstance(eff, mn.effector.RigidTendonArm26) or isinstance(eff, mn.effector.CompliantTendonArm26):
            raise NotImplementedError("fast rollout: effector must be RigidTendonArm26")
        if type(mus) is not mn.muscle.RigidTendonHillMuscle:
            raise NotImplementedError("fast rollout: muscle must be RigidTendonHillMuscle")
        if type(sk) is not mn.skeleton.TwoDofArm:
            raise NotImplementedError("fast rollout: skeleton must be TwoDofArm")
        if eff.integration_method != 'euler' or eff.n_ministeps != 1:
            raise NotImplementedError("fast rollout: Euler integration with 1 ministep only")
        if env.action_frame_stacking != 0 or not env.differentiable:
            raise NotImplementedError("fast rollout: no action frame stacking, differentiable env only")
        if any(float(x) != 0.0 for x in env.obs_noise):
            raise NotImplementedError("fast rollout: env.obs_noise must be 0 (use vision/proprioception noise)")
        if len(env.proprioception_noise) != 1 or len(env.vision_noise) != 1:
            raise NotImplementedError("fast rollout: scalar vision / proprioception noise only")

        self.dt = float(eff.minidt)
        self.vd = int(env.vision_delay)            # delays, in steps (0 = current state)
        self.pd = int(env.proprioception_delay)
        self.read_noise(env)
        self.n_muscles = int(eff.n_muscles)

        # muscle (per-muscle constants as (1, n_muscles))
        nm = self.n_muscles
        self.min_act = float(mus.min_activation)
        self.tau_act = float(mus.tau_activation)
        self.tau_deact = float(mus.tau_deactivation)
        self.l0_se = mus.l0_se.detach().reshape(1, nm).clone()
        self.l0_ce = mus.l0_ce.detach().reshape(1, nm).clone()
        self.l0_pe = mus.l0_pe.detach().reshape(1, nm).clone()
        self.k_pe = mus.k_pe.detach().reshape(1, nm).clone()
        self.vmax = mus.vmax.detach().reshape(1, nm).clone()
        self.max_iso = mus.max_iso_force.detach().reshape(1, nm).clone()
        self.f_iso_n_den = float(mus.f_iso_n_den)
        self.min_flce = float(mus.min_flce)
        self.q_crit = float(mus.q_crit)
        self.s_as = float(mus.s_as)

        # musculotendon geometry (a0: (1,6); a1, a2: (1,2,6); a3: (1,2))
        self.a0 = eff.a0.detach().reshape(1, nm).clone()
        self.a1 = eff.a1.detach().clone()
        self.a2 = eff.a2.detach().clone()
        self.a2x2 = self.a2 * 2
        self.a3 = eff.a3.detach().reshape(1, 2).clone()

        # skeleton
        self.L1, self.L2 = float(sk.L1), float(sk.L2)
        ic, im = sk.inertia_c.detach(), sk.inertia_m.detach()
        self.ic11, self.ic12, self.ic21, self.ic22 = (float(ic[0, 0, 0]), float(ic[0, 0, 1]),
                                                      float(ic[0, 1, 0]), float(ic[0, 1, 1]))
        self.im11, self.im12, self.im21, self.im22 = (float(im[0, 0, 0]), float(im[0, 0, 1]),
                                                      float(im[0, 1, 0]), float(im[0, 1, 1]))
        self.cor1, self.cor2 = float(sk.coriolis_1), float(sk.coriolis_2)
        self.visc = float(sk.c_viscosity)
        self.damping = float(eff.damping)
        self.pos_lb = sk.pos_lower_bound.detach().reshape(1, 2).clone()
        self.pos_ub = sk.pos_upper_bound.detach().reshape(1, 2).clone()
        self.vel_lb = sk.vel_lower_bound.detach().reshape(1, 2).clone()
        self.vel_ub = sk.vel_upper_bound.detach().reshape(1, 2).clone()


    def read_noise(self, env):
        """Noise levels are re-read from the env at every rollout (they are only used outside the
        compiled code), so changing e.g. env.vision_noise between episodes works as in the stock code."""
        if len(env.proprioception_noise) != 1 or len(env.vision_noise) != 1:
            raise NotImplementedError("fast rollout: scalar vision / proprioception noise only")
        self.vis_std = float(env.vision_noise[0])
        self.prop_std = float(env.proprioception_noise[0])
        self.act_std = th.tensor([float(x) for x in env.action_noise], dtype=th.float32).reshape(1, 1, -1)


# ----------------------------------------------------------------------------------------------
# Pure functions (these are what gets compiled)
# ----------------------------------------------------------------------------------------------

def _geometry(c, pos, vel):
    """RigidTendonArm26._get_geometry: musculotendon length, velocity (batch, 6), moment arms (batch, 2, 6)."""
    q = (pos - c.a3).unsqueeze(-1)
    marm = q * c.a2x2 + c.a1
    glen = th.sum((c.a1 + q * c.a2) * q, dim=1) + c.a0
    gvel = th.sum(vel.unsqueeze(-1) * marm, dim=1)
    return glen, gvel, marm


def _muscle(c, act, glen, gvel):
    """RigidTendonHillMuscle._integrate after the activation update.
    Returns normalised muscle length & velocity (the proprioceptive signals) and force."""
    mlen = th.clamp(glen - c.l0_se, min=0.0)
    strain = th.clamp((mlen - c.l0_pe) / c.l0_ce, min=0.0)
    mlen_n = mlen / c.l0_ce
    mvel_n = gvel / c.vmax
    flpe = c.k_pe * strain ** 2
    flce = th.clamp(1 + (-mlen_n ** 2 + 2 * mlen_n - 1) / c.f_iso_n_den, min=c.min_flce)
    a_rel_st = th.where(mlen_n > 1.0, 0.41 * flce, 0.41)
    b_rel_st = th.where(act < c.q_crit, 5.2 * (1 - 0.9 * ((act - c.q_crit) / (0.005 - c.q_crit))) ** 2, 5.2)
    dfdvcon0 = act * (flce + a_rel_st) / b_rel_st
    f_x_a = flce * act
    tmp_p_nom = f_x_a * 0.5
    tmp_p_den = c.s_as - dfdvcon0 * 2.0
    p1 = -tmp_p_nom / tmp_p_den
    p2 = tmp_p_nom ** 2 / tmp_p_den
    p3 = -1.5 * f_x_a
    shortening = mvel_n < 0
    nom = th.where(shortening,
                   mvel_n * act * a_rel_st + f_x_a * b_rel_st,
                   -p1 * p3 + p1 * c.s_as * mvel_n + p2 - p3 * mvel_n + c.s_as * mvel_n ** 2)
    den = th.where(shortening, b_rel_st - mvel_n, p1 + mvel_n)
    active = th.clamp(nom / den, min=0.0)
    force = (active + flpe) * c.max_iso
    return mlen_n, mvel_n, force


def _trig(pos):
    """sin and cos of [shoulder, elbow, shoulder + elbow], each (batch,)."""
    p0, p1 = pos.unbind(1)
    p01 = p0 + p1
    return th.sin(p0), th.sin(p1), th.sin(p01), th.cos(p0), th.cos(p1), th.cos(p01)


def _cartesian(c, trig, vel):
    """TwoDofArm._joint2cartesian -> (batch, 4) [x, y, vx, vy]."""
    s1, s2, s12, c1, c2, c12 = trig
    v0, v1 = vel.unbind(1)
    x = c.L1 * c1 + c.L2 * c12
    y = c.L1 * s1 + c.L2 * s12
    L2s12 = c.L2 * s12
    L2c12 = c.L2 * c12
    vx = -y * v0 - L2s12 * v1
    vy = x * v0 + L2c12 * v1
    return th.stack([x, y, vx, vy], dim=1)


class _AddmmConst(th.autograd.Function):
    """e + x @ WT for a constant weight matrix given both as WT (in, out) and W = WT.t() (out, in),
    each contiguous: the backward product g @ W then runs on contiguous memory like the forward
    one (BLAS is ~2x slower on the transposed view autograd would otherwise use)."""

    @staticmethod
    def forward(ctx, e, x, WT, W):
        ctx.save_for_backward(W)
        return th.addmm(e, x, WT)

    @staticmethod
    def backward(ctx, g):
        W, = ctx.saved_tensors
        return g, g @ W, None, None


def _segment(c, K, use_load, out_delay, deferred, fast_tanh, zeros_h, net,
             h, pos, vel, act, glen, gvel, marm, mlen_n, mvel_n, force, cart,
             vis_pre, prop_pre, inp, n_act, n_hid, n_vis, n_prop, ff_gain):
    """Simulate K consecutive steps.

    Carried state (all for the state s_t at entry): h, pos, vel, act, the geometry (glen, gvel, marm)
    of the current joint state, the current muscle state (mlen_n, mvel_n, force) and cart.
    Per-step inputs, stacked on dim 0 (length K): inp (task inputs), n_act (action noise),
    n_hid (hidden noise, (K, batch, hidden)), n_vis / n_prop (noise for the sensory samples taken from
    the K new states), ff_gain ((K, batch) force-field strength, or None).
    vis_pre / prop_pre: (n, batch, .) the already-noisy delayed sensory signals for the first n
    steps of this segment (those that come from states simulated before the segment); later
    steps use states simulated inside the segment.

    net: the network weights.
    deferred=False: (Wz, bz, Wr, br, Wh, bh, Y, bY), used exactly as in ModularPolicyGRU.forward.
    deferred=True: (WzrT, Wzr, WhT, Wh, Ezr, Eh, Y, bY) -- see _DeferredLinear. The gate weights are
    constants here, laid out for inputs ordered [vision, proprioception, task, 1, hidden] (the bias
    is the column fed by the 1; `inp` then carries that extra column of ones), and Ezr / Eh are
    zero tensors (K, batch, .) added to the pre-activations, whose gradients are collected instead.

    Returns (carry, logs, aux). logs are stacked on dim 1 (batch, K, .) except the hidden states,
    which are returned as (K + 1, batch, hidden) including the entry state.
    """
    dt = c.dt
    n_vis_pre, n_prop_pre = vis_pre.shape[0], prop_pre.shape[0]
    vis_new, prop_new = [], []          # noisy sensory samples of the new states
    log_cart, log_joint, log_h, log_u, log_act, log_force, log_prop = [], [], [h], [], [], [], []

    if deferred:
        WzrT, Wzr, WhT, Wh, Ezr, Eh, Y, bY = net
        Ezr, Eh = Ezr.unbind(0), Eh.unbind(0)
        H = WhT.shape[1]
        log_obs, log_rh = [], []
    else:
        Wz, bz, Wr, br, Wh, bh, Y, bY = net

    trig = _trig(pos)
    for i in range(K):
        # ---- observation (delayed, noisy) -------------------------------------------------
        vis = vis_pre[i] if i < n_vis_pre else vis_new[i - n_vis_pre]
        prop = prop_pre[i] if i < n_prop_pre else prop_new[i - n_prop_pre]

        # ---- network ---------------------------------------------------------------------
        if deferred:
            obs = th.cat((vis, prop, inp[i]), dim=1)                  # [vision, proprio, task, 1]
            zr = th.sigmoid(_AddmmConst.apply(Ezr[i], th.cat((obs, h), dim=1), WzrT, Wzr))
            z, r = zr[:, :H], zr[:, H:]
            rh = r * h
            pre = _AddmmConst.apply(Eh[i], th.cat((obs, rh), dim=1), WhT, Wh) + n_hid[i]
            log_obs.append(obs.detach())
            log_rh.append(rh.detach())
        else:
            concat = th.cat((vis, prop, inp[i], h), dim=1)
            z = th.sigmoid(F.linear(concat, Wz, bz))
            r = th.sigmoid(F.linear(concat, Wr, br))
            concat_hidden = th.cat((vis, prop, inp[i], r * h), dim=1)
            pre = F.linear(concat_hidden, Wh, bh) + n_hid[i]
        if fast_tanh:
            # tanh(x) = -expm1(-2x) / (2 + expm1(-2x)): the same function (agrees with th.tanh to a
            # few float32 ulp), but the compiler can vectorise expm1 and not tanh
            em = th.expm1(th.clamp(pre, min=-10.0) * -2.0)
            h_tilda = -em / (em + 2.0)
        else:
            h_tilda = th.tanh(pre)
        if zeros_h is not None:
            h_tilda = th.maximum(zeros_h, h_tilda)
        h_new = (1 - z) * h + z * h_tilda
        u = th.sigmoid(F.linear(h_new if out_delay == 0 else h, Y, bY))

        # ---- muscle activation dynamics ------------------------------------------------------
        a = th.clamp(u + n_act[i], min=c.min_act, max=1.0)
        tmp = 0.5 + 1.5 * act
        tau = th.where(a > act, c.tau_act * tmp, c.tau_deact / tmp)
        act_new = th.clamp(act + ((a - act) / tau) * dt, min=c.min_act, max=1.0)

        # ---- arm dynamics (uses force, moment arms and joint state of s_t) ---------------------
        s1, s2, s12, c1, c2, c12 = trig
        v0, v1 = vel.unbind(1)
        gen = -th.sum(force.unsqueeze(1) * marm, dim=-1)
        if c.damping != 0.0:
            gen = gen - c.damping * vel
        g0, g1 = gen.unbind(1)
        i11 = c.ic11 + c2 * c.im11
        i12 = c.ic12 + c2 * c.im12
        i21 = c.ic21 + c2 * c.im21
        i22 = c.ic22 + c2 * c.im22 if c.im22 != 0.0 else c.ic22
        cor_1 = c.cor1 * s2 * (2 * v0 + v1) * v1
        cor_2 = c.cor2 * s2 * v0 * v0
        if c.visc != 0.0:
            cor_1 = cor_1 + c.visc * v0
            cor_2 = cor_2 + c.visc * v1
        if use_load:
            # curl field: load = k * [vy, -vx] of the current hand velocity (applied_load, mode 'CW')
            lx = ff_gain[i] * cart[:, 3]
            ly = -(ff_gain[i] * cart[:, 2])
            L2s12 = c.L2 * s12
            L2c12 = c.L2 * c12
            j11 = -c.L1 * s1 - L2s12
            j21 = c.L1 * c1 + L2c12
            g0 = g0 + (j11 * lx + j21 * ly)
            g1 = g1 + (-L2s12 * lx + L2c12 * ly)
        rhs0 = g0 - cor_1
        rhs1 = g1 - cor_2
        denom = 1 / (i11 * i22 - i12 * i21)
        acc0 = (denom * i22) * rhs0 + (denom * -i12) * rhs1
        acc1 = (denom * -i21) * rhs0 + (denom * i11) * rhs1
        acc = th.stack([acc0, acc1], dim=1)

        # ---- muscle state of s_{t+1}: new activation, geometry of s_t ---------------------------
        mlen_n, mvel_n, force = _muscle(c, act_new, glen, gvel)
        prop_f = th.cat([mlen_n, mvel_n], dim=1)

        # ---- Euler step of the joints + limits --------------------------------------------------
        vel_new = vel + acc * dt
        pos_new = pos + vel * dt
        vel_new = th.clamp(vel_new, min=c.vel_lb, max=c.vel_ub)
        stop = ((vel_new < 0) & (pos_new <= c.pos_lb)) | ((vel_new > 0) & (pos_new >= c.pos_ub))
        vel_new = th.where(stop, 0.0, vel_new)
        pos_new = th.clamp(pos_new, min=c.pos_lb, max=c.pos_ub)

        h, pos, vel, act = h_new, pos_new, vel_new, act_new
        glen, gvel, marm = _geometry(c, pos, vel)
        trig = _trig(pos)
        cart = _cartesian(c, trig, vel)

        # ---- sensory samples of the new state (consumed later through the delays) ----------------
        if (i + n_vis_pre) < K:      # only if used inside this segment
            vis_new.append(cart[:, :2] + n_vis[i])
        if (i + n_prop_pre) < K:
            prop_new.append(prop_f + n_prop[i])

        log_cart.append(cart); log_joint.append(th.cat([pos, vel], dim=1)); log_h.append(h)
        log_u.append(u); log_act.append(act); log_force.append(force); log_prop.append(prop_f)

    carry = (h, pos, vel, act, glen, gvel, marm, mlen_n, mvel_n, force, cart)
    logs = (th.stack(log_cart, 1), th.stack(log_joint, 1), th.stack(log_h, 0), th.stack(log_u, 1),
            th.stack(log_act, 1), th.stack(log_force, 1), th.stack(log_prop, 1))
    aux = (th.stack(log_obs, 0), th.stack(log_rh, 0)) if deferred else ()
    return carry, logs, aux


_PATCHED = False


def _ban_transcendental_recompute():
    """Optional tweak of torch internals (the AOT-autograd partitioner): by default the compiled
    backward pass re-computes sigmoid / tanh / sin / cos / expm1 from their saved inputs instead of
    saving their outputs; in this workload those functions are a large part of the remaining cost."""
    global _PATCHED
    if _PATCHED:
        return
    import torch._functorch.partitioners as part
    import torch._functorch.config as fconfig
    orig = part.get_default_op_list
    banned = [getattr(th.ops.aten, n) for n in ('sigmoid', 'tanh', 'sin', 'cos', 'expm1', 'exp')]

    def patched():
        ops = orig()
        for op in banned:
            ops.recomputable_ops.discard(op)
        return ops

    part.get_default_op_list = patched
    # Inert change of a config value (999 behaves like the default 1000), only so that torch's
    # on-disk compile cache does not mix graphs partitioned with and without this patch.
    fconfig.max_dist_from_bw = 999
    _PATCHED = True


class _Holder:
    obs = None      # (K, batch, n_obs + 1)  [vision, proprio, task, 1] fed to the network at each step
    h = None        # (K + 1, batch, hidden) hidden states, including the one at entry
    rh = None       # (K, batch, hidden)     reset gate * previous hidden


class _DeferredLinear(th.autograd.Function):
    """Computes the gradients of the GRU gate weights with a few big matrix products per segment.

    In the rollout the gate weights are used as constants: pre_t = [x_t, 1] @ [W, b]^T + E_t, where
    E (the output of this Function) is identically zero. Backpropagation delivers dLoss/dE_t
    (= dLoss/dpre_t) for every step here, and then dLoss/d[W, b] = sum_t dLoss/dpre_t^T [x_t, 1],
    exactly what autograd would have accumulated one step at a time (up to summation order).
    """

    @staticmethod
    def forward(ctx, Wz, bz, Wr, br, Wh, bh, holder, zeros_zr, zeros_h):
        ctx.holder = holder
        ctx.H = Wh.shape[0]
        # fresh tensor objects sharing the storage of persistent all-zero buffers (never written)
        return zeros_zr.view_as(zeros_zr), zeros_h.view_as(zeros_h)

    @staticmethod
    def backward(ctx, g_zr, g_h):
        H, hd = ctx.H, ctx.holder
        n = hd.obs.shape[0] * hd.obs.shape[1]
        obs = hd.obs.reshape(n, -1)
        out = [None] * 6
        if g_zr is not None:
            g = g_zr.reshape(n, 2 * H).t()
            Go, Gh = g @ obs, g @ hd.h[:-1].reshape(n, H)
            out[0], out[1] = th.cat((Go[:H, :-1], Gh[:H]), 1), Go[:H, -1]
            out[2], out[3] = th.cat((Go[H:, :-1], Gh[H:]), 1), Go[H:, -1]
        if g_h is not None:
            g = g_h.reshape(n, H).t()
            Go, Gh = g @ obs, g @ hd.rh.reshape(n, H)
            out[4], out[5] = th.cat((Go[:, :-1], Gh), 1), Go[:, -1]
        return (*out, None, None, None)


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------

class FastRollout:
    """Holds the constants and (optionally compiled) segment function for one (env, policy) pair.

    chunk:    None = the whole episode is one segment; K = the rollout is executed as n_t / K calls
              of a K-step segment.
    compile:  torch.compile the segment function (compile_kwargs are passed to torch.compile).
    deferred: gate-weight gradients via _DeferredLinear (faster) instead of step by step.
    fast_tanh: compute tanh through expm1 (compiled code vectorises it; differs from th.tanh by
              float32 round-off only).
    ban_recompute: see _ban_transcendental_recompute (only matters when compiling).
    cpp_wrapper: have the compiler emit the glue between its kernels in C++ instead of Python.
    """

    def __init__(self, env, policy, chunk=None, compile=False, compile_kwargs=None, deferred=True,
                 fast_tanh=False, ban_recompute=False, cpp_wrapper=False):
        if not isinstance(policy, ModularPolicyGRU):
            raise NotImplementedError("fast rollout: policy must be ModularPolicyGRU")
        if policy.max_connectivity_delay != 0:
            raise NotImplementedError("fast rollout: connectivity delays are not supported")
        if policy.output_delay not in (0, 1):
            raise NotImplementedError("fast rollout: output_delay must be 0 or 1")
        self.c = _Consts(env)
        self.env, self.policy = env, policy
        self.chunk = chunk
        self.deferred = deferred
        self.fast_tanh = fast_tanh
        self.rect = policy.activation is not th.tanh
        self.keep_obs = False
        self.last_obs = None
        self._const = {}
        self._seg = _segment
        self._compiled = bool(compile)
        if compile:
            if ban_recompute:
                _ban_transcendental_recompute()
            compile_kwargs = dict(compile_kwargs or {})
            if cpp_wrapper and 'mode' in compile_kwargs:
                import torch._inductor.config as iconfig     # torch.compile refuses mode= with options=
                iconfig.cpp_wrapper = True
            elif cpp_wrapper:
                compile_kwargs['options'] = {**compile_kwargs.get('options', {}), 'cpp_wrapper': True}
            self._seg = th.compile(_segment, **compile_kwargs)

    def _cached(self, kind, *shape):
        key = (kind, *shape)
        if key not in self._const:
            self._const[key] = th.ones(*shape) if kind == 'ones' else th.zeros(*shape)
        return self._const[key]

    def _initial_state(self, init_states):
        c = self.c
        pos, vel = init_states[:, :2], init_states[:, 2:]
        glen, gvel, marm = _geometry(c, pos, vel)
        act = th.full((pos.shape[0], c.n_muscles), c.min_act)
        act = th.clamp(act, min=c.min_act, max=1.0)
        mlen_n, mvel_n, force = _muscle(c, act, glen, gvel)
        cart = _cartesian(c, _trig(pos), vel)
        return pos, vel, act, glen, gvel, marm, mlen_n, mvel_n, force, cart

    def rollout(self, inp, init_states, k=0.0):
        """inp: (batch, n_t, n_inputs) task inputs, init_states: (batch, 4). Returns dict of logs."""
        c, policy = self.c, self.policy
        c.read_noise(self.env)
        B, T = inp.shape[0], inp.shape[1]
        H = policy.hidden_size
        K = T if self.chunk is None else self.chunk
        Dv, Dp = c.vd, c.pd

        # -- all the noise for the episode, in a few big draws ------------------------------------
        n_act = th.randn(T, B, c.n_muscles) * c.act_std
        n_hid = th.randn(T, B, H) * float(policy.hidden_noise)
        n_vis = th.randn(T + 1, B, 2) * c.vis_std                   # [j]: noise of the sample of state s_j
        n_prop = th.randn(T + 1, B, 2 * c.n_muscles) * c.prop_std
        n_vis0 = th.randn(Dv, B, 2) * c.vis_std                     # [t]: noise of the sample of s_0 seen
        n_prop0 = th.randn(Dp, B, 2 * c.n_muscles) * c.prop_std     #      at step t < delay
        inp_t = inp.transpose(0, 1)                     # (T, B, n_inputs)
        use_load = k != 0
        ff_gain = (k * (inp_t[:, :, 2] > 0.5).float()) if use_load else None

        # -- reset ---------------------------------------------------------------------------------
        with th.no_grad():
            state0 = self._initial_state(init_states)
        if self._compiled:
            # The initial arm state does not depend on the weights, so the first segment would be
            # compiled a second time (different requires_grad pattern). Adding a zero that requires
            # grad makes every segment look the same to the compiler (values are unchanged).
            anchor = th.zeros((), requires_grad=True)
            state0 = tuple(x + anchor for x in state0)
        pos, vel, act, glen, gvel, marm, mlen_n, mvel_n, force, cart = state0
        h0 = policy.activation(policy.h0)
        h = th.tile(h0, (B, 1))
        zeros_h = self._cached('zeros', B, H) if self.rect else None

        # noisy sensory sample histories: X[j] = sample of state s_j
        vis0 = cart[:, :2]
        prop0 = th.cat([mlen_n, mvel_n], dim=1)
        vis_hist = [vis0 + n_vis[0]]
        prop_hist = [prop0 + n_prop[0]]

        def vis_at(t):      # what the network sees at step t
            return vis0 + n_vis0[t] if t < Dv else vis_hist[t - Dv]

        def prop_at(t):
            return prop0 + n_prop0[t] if t < Dp else prop_hist[t - Dp]

        if self.deferred:
            n_in = policy.Wz.shape[1] - H            # sensory + task inputs
            with th.no_grad():
                # rows: [z; r], columns reordered to [inputs, bias, hidden]
                Wzr = th.cat((policy.Wz, policy.Wr), 0)
                Wzr = th.cat((Wzr[:, :n_in], th.cat((policy.bz, policy.br))[:, None], Wzr[:, n_in:]), 1)
                WzrT = Wzr.t().contiguous()
                Wh = th.cat((policy.Wh[:, :n_in], policy.bh[:, None], policy.Wh[:, n_in:]), 1)
                WhT = Wh.t().contiguous()
                inp_t = th.cat((inp_t, self._cached('ones', T, B, 1)), dim=2)
            gate_params = (policy.Wz, policy.bz, policy.Wr, policy.br, policy.Wh, policy.bh)
        else:
            P = (policy.Wz, policy.bz, policy.Wr, policy.br, policy.Wh, policy.bh, policy.Y, policy.bY)

        carry = (h, pos, vel, act, glen, gvel, marm, mlen_n, mvel_n, force, cart)
        logs = []
        for t0 in range(0, T, K):
            Kc = min(K, T - t0)
            # the first delay + 1 steps of the segment see states simulated before it
            vis_pre = th.stack([vis_at(t) for t in range(t0, t0 + min(Dv + 1, Kc))])
            prop_pre = th.stack([prop_at(t) for t in range(t0, t0 + min(Dp + 1, Kc))])
            sl = slice(t0, t0 + Kc)
            nv, npr = n_vis[t0 + 1:t0 + Kc + 1], n_prop[t0 + 1:t0 + Kc + 1]   # for states s_{t0+1} .. s_{t0+Kc}
            if self.deferred:
                holder = _Holder()
                Ezr, Eh = _DeferredLinear.apply(*gate_params, holder, self._cached('zeros', Kc, B, 2 * H),
                                                self._cached('zeros', Kc, B, H))
                P = (WzrT, Wzr, WhT, Wh, Ezr, Eh, policy.Y, policy.bY)
            carry, lg, ax = self._seg(c, Kc, use_load, policy.output_delay, self.deferred, self.fast_tanh,
                                      zeros_h, P, *carry,
                                      vis_pre, prop_pre, inp_t[sl], n_act[sl], n_hid[sl], nv, npr,
                                      ff_gain[sl] if use_load else None)
            logs.append(lg)
            if self.deferred:
                holder.obs, holder.rh = ax
                holder.h = lg[2].detach()
                if self.keep_obs:      # for tests: the (noisy, delayed) inputs the network received
                    self.last_obs = ax[0] if t0 == 0 else th.cat((self.last_obs, ax[0]))
            if t0 + Kc < T:   # extend the sensory histories for the next segment
                cart_l, prop_l = lg[0], lg[6]
                need_v = min(Kc, Dv + 1)      # only the most recent samples can still be needed
                for m in range(Kc):
                    j = t0 + m + 1            # state index of this sample
                    if m >= Kc - need_v:
                        vis_hist.append(cart_l[:, m, :2] + n_vis[j])
                    else:
                        vis_hist.append(None)
                    if m >= Kc - min(Kc, Dp + 1):
                        prop_hist.append(prop_l[:, m] + n_prop[j])
                    else:
                        prop_hist.append(None)

        # hidden states come back as (K + 1, batch, hidden) per segment -> (batch, T, hidden)
        hidden = th.cat([lg[2][1:].transpose(0, 1) for lg in logs], dim=1)
        if len(logs) == 1:
            xy, joint, _, actions, muscle, force_l, _ = logs[0]
        else:
            xy, joint, actions, muscle, force_l = (th.cat([lg[n] for lg in logs], dim=1) for n in (0, 1, 3, 4, 5))
        return {'xy': xy, 'hidden': hidden, 'actions': actions, 'muscle': muscle, 'force': force_l, 'joint': joint}


# The fastest measured configuration (see the benchmark notes): compiled 25-step segments.
FAST_PRESET = dict(chunk=25, compile=True, fast_tanh=True, ban_recompute=True, cpp_wrapper=True)

_CACHE = {}


def get_fast_rollout(env, policy, **kw):
    key = (id(env), id(policy), repr(sorted(kw.items())))
    if key not in _CACHE:
        _CACHE[key] = FastRollout(env, policy, **kw)
    return _CACHE[key]


def run_episode_fast(env, task, policy, batch_size, n_t, device, k=0, dmax=0.30, **kw):
    """Drop-in replacement for my_utils.run_episode (same arguments, same returned dict).
    Extra keyword arguments select the execution mode (see FastRollout)."""
    inputs, targets, init_states, delay_go_times, delay_tg_times = task.generate(batch_size, n_t, dmax=dmax)
    targets = th.tensor(targets[:, :, 0:2], device=device, dtype=th.float)
    inp = th.tensor(inputs, device=device, dtype=th.float)
    init_states = th.tensor(init_states, device=device, dtype=th.float)

    out = get_fast_rollout(env, policy, **kw).rollout(inp, init_states, k=k)
    out.update({
        'targets': targets,
        'inp': inp,
        'l1': env.skeleton.l1,
        'l2': env.skeleton.l2,
        'dt': env.dt,
        'delay_go_times': delay_go_times,
        'delay_tg_times': delay_tg_times,
    })
    return out
