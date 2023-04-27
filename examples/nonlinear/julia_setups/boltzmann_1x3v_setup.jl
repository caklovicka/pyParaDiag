using KitBase

params = [384, 72, 36, 36]
knudsen = 5e-2

Nx = params[1] - 1
Nu = params[2]
Nv = params[3]
Nw = params[4]

st = KB.Setup(space = "1d1f3v", collision = "fsm", interpOrder = 1, boundary = "period", maxTime = 1.0)
ps = KB.PSpace1D(0.0, 1.0, Nx, 1)
vs = KB.VSpace3D(-8, 8, Nu, -8, 8, Nv, -8, 8, Nw)

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

# write gas.fsm.Kn
file = open("gas_fsm_kn.txt", "w")
print(file, gas.fsm.Kn)
close(file)

# write gas.fsm.nm
file = open("gas_fsm_nm.txt", "w")
print(file, gas.fsm.nm)
close(file)

#write gas.γ
file = open("gas_gamma.txt", "w")
print(file, gas.γ)
close(file)

# write phi
file = open("phi.txt", "w")
print(file, size(phi, 1), " ", size(phi, 2), " ", size(phi, 3), " ", size(phi, 4))
print(file, "\n")
print(file, vec(phi))
close(file)

# write psi
file = open("psi.txt", "w")
print(file, size(psi, 1), " ", size(psi, 2), " ", size(psi, 3), " ", size(psi, 4))
print(file, "\n")
print(file, vec(psi))
close(file)

# write chi
file = open("chi.txt", "w")
print(file, size(chi, 1), " ", size(chi, 2), " ", size(chi, 3))
print(file, "\n")
print(file, vec(chi))
close(file)

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

# compute initial condition
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
for i in 1:Nx
    f[i, :, :, :] = ff(ps.x[i+1], ())
    w[i, :] = KB.moments_conserve(f[i, :, :, :], vs.u, vs.v, vs.w, vs.weights)
end

# write f0
#file = open("f0.txt", "w")
#print(file, size(f, 1), " ", size(f, 2), " ", size(f, 3), " ", size(f, 4))
#print(file, "\n")
#print(file, vec(f))
#close(file)

