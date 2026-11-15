# Numerical method and conventions

## Free band and constrained screening

The eight orbitals are `(A1,B1,A2,B2,A3,B3,A4,B4)`. Energies are in eV internally, while the user-facing displacement is in meV. ΔD is the outer-layer potential difference: the four layers have potentials +ΔD/2, +ΔD/6, −ΔD/6, −ΔD/2, with the onsite offset and hopping parameters defined in `free_fermion_gamma/model.py`. Momenta are dimensionless k a, with lattice constant a=0.246 nm. The mesh is an open centered triangular hexagon with radius M, spacing 0.12/M by default, and 1+3M(M+1) points. There is no periodic wrap at the domain boundary.

A single spin–valley flavor (`K up`) is used, with active conduction-band index 4 (zero based). The supplied density is the excess carrier density of that flavor relative to the four filled valence bands. The chemical potential is determined from the free eight-band occupations. No additional spin/valley degeneracy factor is applied in screening. No Hartree–Fock or normal self-energy update is used.

Let u_k be the active-band Bloch eigenvector. The static constrained interaction is

```math
W_c(Q)=\frac{V_0(Q)}{1-V_0(Q)\Pi_{\rm rest}(Q)},\qquad
\Pi_{\rm rest}=\sum_{(m,n)\ne(a,a)}\Pi_{mn}.
```

The sum retains 63 of the 64 ordered eight-band transitions, including mixed active/remote terms. The gate-screened bare interaction and its finite Q=0 limit are implemented in the bundled free-band model. The same W_c screens both direct and exchange source terms.

## Transfer blocks and the parquet solve

The four legs are `(q-Q,q;p-Q,p)`. A matrix row is q and a column is p, restricted to the domain where both k and k−Q are in the mesh. In this convention the projected, antisymmetrized source is

```math
\Lambda_c^Q(q,p)=
W_c(Q)\langle u_{q-Q}|u_q\rangle\langle u_p|u_{p-Q}\rangle
-W_c(p-q)\langle u_{q-Q}|u_{p-Q}\rangle\langle u_p|u_q\rangle.
```

The diagonal static particle-hole bubble, including the quadrature weight, is

```math
B_Q(p)=w\frac{f(\xi_p)-f(\xi_{p-Q})}{\xi_p-\xi_{p-Q}},\qquad
w=\frac{|\det({\rm basis})|}{(2\pi)^2},
```

with the Fermi-derivative limit at coincident energies. In this sign convention B_Q≤0. With K_Q=Γ^ph_Q, solve

```math
(I-K_QB_Q)\Phi_Q=K_QB_QK_Q,\qquad
R_Q[K]=K_Q-g\Lambda_c^Q+(C\Phi)_Q=0,
```

where crossing couples transfer sectors through

```math
(C\Phi)^Q(q,p)=\Phi^{p-q}(p-Q,p).
```

Newton–Krylov iterations solve the coupled residual; the Phi equations are blockwise linear solves. Coupling continuation scales the already constrained-screened source by g, keeping W_c fixed, until g=1. The default policy uses only preceding converged states and reduces the continuation step after failures. Physical Hermiticity and pair-interchange symmetries are preserved. Large arrays and Jacobian actions use the streamed-memory implementation.

The full vertex used for pairing is reconstructed explicitly as

```math
\Gamma=g\Lambda_c-C\Phi+\Phi.
```

This preserves crossing algebraically. Away from an exact root it differs from K+Phi by −R[K], so the full-vertex BSE defect is saved separately from the parquet equation residual. Neither a small algebraic symmetry defect nor a loose acceptance tolerance establishes mesh convergence.

## Pairing and the number constraint

The Cooper slice is

```math
\Gamma_C(k,k')=\Gamma^{k'-k}(k',-k),\qquad
V_{\rm pair}=\tfrac12\Gamma_C.
```

The factor 1/2 is required for this antisymmetrized interaction on the full mesh; form factors and the mesh weight are not inserted twice. The odd projector is P_-Δ(k)=[Δ(k)−Δ(−k)]/2. The linearized equation is

```math
\lambda\Delta=-wP_-V_{\rm pair}P_-\chi_{pp}\Delta,\qquad
\chi_{pp}(k)=\frac{1-f(\xi_k)-f(\xi_{-k})}{\xi_k+\xi_{-k}}.
```

The denominator's zero limit is handled explicitly. The Hermitian susceptibility-weighted problem is solved in the independent odd subspace. A leading eigenvalue above one signals a linear instability in this channel. This **Cooper eigenvalue** is distinct from the particle-hole kernel eigenvalue monitored in parquet continuation.

For an unstable normal state, the nonlinear gap and number equations are solved together. The free normal state's active-band population is held fixed by adjusting the chemical potential; remote occupations are frozen. The intravalley asymmetry ξ_k≠ξ_−k is retained. A converged normal result is returned when the linearized instability is absent; this does not exclude a first-order transition. Since k is measured relative to K, intravalley pairs carry physical total momentum 2K.

The optional second-temperature calculation reuses the same static vertex. It is a fixed-vertex approximation, not a full recalculation at the lower temperature. The parquet approximation includes particle-hole renormalization but omits vertex frequency dependence and the particle-particle parquet channel.

## Phase convention and winding

The saved gap is the active-band gap in the eigensolver's Bloch gauge. Re/Im plots fix only one overall superconducting phase. For winding, define θ_k=arg u_A1(k) and

```math
\Delta_{A_1+}(k)=e^{i[\theta_k+\theta_{-k}]}\Delta(k).
```

This expresses the same band gap in a gauge with positive-real A1 components. Its phase equals that of u_A1(k)u_A1(−k)Δ(k); it is not an independently solved orbital gap. The reference component must remain nonzero. One constant phase rotation fixes the plotting origin.

On each closed normal-state Fermi contour, traverse counterclockwise and interpolate the **complex** gap piecewise linearly. Sum the phase increments arg(Δ_{j+1}Δ_j*)/(2π), including the closing segment. The code checks minimum amplitude along segments and phase-step resolution; unresolved contours return no integer winding. Gap nodes, an open contour, or a vanishing reference orbital invalidate that assignment. A winding of −3 in this convention describes an f-wave-like texture; it alone is not a crystal representation assignment or a BdG Chern number.
