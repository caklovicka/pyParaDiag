using KitBase

st = KB.Setup(space = "1d1f3v", collision = "fsm", interpOrder = 1, boundary = "period", maxTime = 1.0)
ps = KB.PSpace1D(0.0, 1.0, 100, 1)
vs = KB.VSpace3D(-8, 8, 48, -8, 8, 28, -8, 8, 28)

knudsen = 1e-2
muref = KB.ref_vhs_vis(knudsen, 1.0, 0.5)
fsm = KB.fsm_kernel(vs, muref, 5, 1.0)
gas = KB.Gas(Kn=knudsen, K=0.0, fsm=fsm)
phi, psi, chi = KB.kernel_mode(
    5,
    vs.u1,
    vs.v1,
    vs.w1,
    vs.du[1, 1, 1],
    vs.dv[1, 1, 1],
    vs.dw[1, 1, 1],
    vs.nu,
    vs.nv,
    vs.nw,
    1.0,
)

function fw(x, p)
    ρ = 1 + 0.1 * sin(2 * pi * x)
    u = 1.0
    λ = ρ
    return KB.prim_conserve([ρ, u, 0, 0, λ], gas.γ)
end

function ff(x, p)
    w = fw(x, p)
    prim = KB.conserve_prim(w, gas.γ)
    return KB.maxwellian(vs.u, vs.v, vs.w, prim)
end

w = zeros((ps.nx, 5))
f = zeros((ps.nx, vs.nu, vs.nv, vs.nw))
df = deepcopy(f)
Q = deepcopy(f)
for i in 1:100
    f[i, :, :, :] = ff(ps.x[i+1], ())
    w[i, :] = KB.moments_conserve(f[i, :, :, :], vs.u, vs.v, vs.w, vs.weights)
end

w0 = deepcopy(w)

function compute_Q(Q, f, ps, gas, phi, psi, chi, dt)
    # f = (x, u, v, w)
    Base.Threads.@threads for i in 1:ps.nx
        @inbounds Q[i, :, :, :] = dt * KB.boltzmann_fft(f[i, :, :, :], gas.fsm.Kn, gas.fsm.nm, phi, psi, chi)
    end
end

dt = 1e-3
compute_Q(Q, f, ps, gas, phi, psi, chi, dt)

for iter in 1:10
    #compute_df(df, f, ps, vs)
    print(iter)
    @time compute_Q(Q, f, ps, gas, phi, psi, chi, dt)
    #compute_Qsimple(Q, f, ps, vs, gas, muref, dt)
    #step(f, df, Q, ps, vs, dt)
end
