using  ApproxOperator

import ApproxOperator.WeightedResidual: ∫kNṄdxdt, ∫kNNdxdt, ∫NṄdxdt, ∫c²B₁B₁dxdt
import ApproxOperator.Heat: ∫vtdΓ, ∫vgdΓ, ∫vbdΩ, L₂, ∫∫∇v∇udxdy

using GLMakie
using WriteVTK

# ps = MKLPardisoSolver()
# set_matrixtype!(ps,2)

include("import_hmd.jl")

ndiv= 4
# elements,nodes = import_hmd_Tri6("./msh/b=2/Tri6非均布"*string(ndiv)*".msh")
# elements,nodes = import_hmd_Tri3("./msh/b=2/Tri3反向"*string(ndiv)*".msh")
# elements,nodes = import_hmd_Tri6("./msh/b=2/"*string(ndiv)*".msh")
elements,nodes = import_hmd_Tri6("./msh/b=2/Tri6反向"*string(ndiv)*".msh")

nₚ = length(nodes)
nₑ = length(elements["Ω"])

set∇𝝭!(elements["Ω"])
set𝝭!(elements["Γ₁"])
set𝝭!(elements["Γ₂"])
set𝝭!(elements["Γ₃"])
set𝝭!(elements["Γ₄"])
set∇𝝭!(elements["Ωᵍ"])

ρA = 1.0
EA = 1.0
α = 1e15
L = 1.0
v₀ = 1.0
c = √(EA/ρA)

prescribe!(elements["Ω"],:k=>(x,y,z)->EA)
prescribe!(elements["Ω"],:c=>(x,y,z)->c)
prescribe!(elements["Γ₁"],:α=>(x,y,z)->α)
prescribe!(elements["Γ₃"],:α=>(x,y,z)->α)
prescribe!(elements["Γ₄"],:α=>(x,y,z)->α)
prescribe!(elements["Γ₁"],:g=>(x,y,z)->0.0)
prescribe!(elements["Γ₃"],:g=>(x,y,z)->0.0)
prescribe!(elements["Γ₄"],:g=>(x,y,z)->0.0)
prescribe!(elements["Γ₁"],:t=>(x,y,z)->1.0)

Kuu = zeros(nₚ,nₚ)
Kuv = zeros(nₚ,nₚ)
Kvv = zeros(nₚ,nₚ)
Kvu = zeros(nₚ,nₚ)

a₁ = ∫kNṄdxdt => elements["Ω"]
a₂ = ∫kNNdxdt => elements["Ω"]
a₃ = ∫NṄdxdt => elements["Ω"]
a₄ = ∫c²B₁B₁dxdt => elements["Ω"]

a₁(Kuu)
a₂(Kuv)
a₃(Kvv)
a₄(Kvu)
𝑓 = ∫vtdΓ=>elements["Γ₁"]

𝑎ᵅ = ∫vgdΓ=>elements["Γ₁"]∪elements["Γ₄"]
𝑎ᵝ = ∫vgdΓ=>elements["Γ₃"]∪elements["Γ₄"]

k = zeros(nₚ,nₚ)
f = zeros(nₚ)
kᵅ = zeros(nₚ,nₚ)
fᵅ = zeros(nₚ)
kᵝ = zeros(nₚ,nₚ)
fᵝ = zeros(nₚ)

𝑓(f)
𝑎ᵅ(kᵅ,fᵅ)
𝑎ᵝ(kᵝ,fᵝ)

A = [Kuu + kᵅ    Kuv        ;
     Kvu         Kvv + kᵝ   ]
b = [fᵅ; fᵝ - f]
uv = A\b
d = uv[1:nₚ]
push!(nodes,:d=>d)

xs = [node.x for node in nodes]'
ys = [node.y for node in nodes]'
zs = [node.z for node in nodes]'
points = [xs; ys; zs]
cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE_STRIP, [xᵢ.𝐼 for xᵢ in elm.𝓒]) for elm in elements["Ω"]]

σ = zeros(nₑ)
for (j,p) in enumerate(elements["Ω"])
    σ_ = 0.0
    𝑤_ = 0.0
    for ξ in p.𝓖
        B₁ = ξ[:∂𝝭∂x]
        ε = 0.0
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(p.𝓒)
            ε += B₁[i]*xᵢ.d
        end
        σ_ += EA*ε*𝑤
        𝑤_ += 𝑤
    end
    σ[j] = σ_/𝑤_
end

 vtk_grid("./vtk/impact_bar/Tri6_反向_"*string(ndiv)*"_"*string(nₚ), points, cells) do vtk
     vtk["位移"] = d
     vtk["应力"] = σ
 end
