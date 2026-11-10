using  ApproxOperator

import ApproxOperator.WeightedResidual: ∫∫kNṄdxdt, ∫∫kNNdxdt, ∫∫NṄdxdt, ∫∫c²B₁B₁dxdt
import ApproxOperator.Heat: ∫vtdΓ, ∫vgdΓ, ∫vbdΩ, L₂, ∫∫∇v∇udxdy

using ApproxOperator.GmshImport: getPhysicalGroups, getElements, get𝑿ᵢ
import Gmsh: gmsh
using WriteVTK

# ps = MKLPardisoSolver()
# set_matrixtype!(ps,2)

ρA = 1.0
EA = 1.0
α = 1e15
L = 1.0
v₀ = 1.0
c = √(EA/ρA)

integrationorder = 2
integrationorder_Ωᵍ = 10
ndiv= 8
filename = "Tri3反向"*string(ndiv)
gmsh.initialize()
gmsh.open("./msh/b=2/"*filename*".msh")
entities = getPhysicalGroups()
nodes = get𝑿ᵢ()
nₚ = length(nodes)
kᵘᵘ = zeros(nₚ,nₚ)
kᵘᵛ = zeros(nₚ,nₚ)
kᵛᵛ = zeros(nₚ,nₚ)
kᵛᵘ = zeros(nₚ,nₚ)

elements = getElements(nodes, entities["Ω"], integrationorder)
prescribe!(elements,:k=>EA,:c=>c)
set∇𝝭!(elements)
𝑎₁ = ∫∫kNṄdxdt => elements
𝑎₂ = ∫∫kNNdxdt => elements
𝑎₃ = ∫∫NṄdxdt => elements
𝑎₄ = ∫∫c²B₁B₁dxdt => elements
𝑎₁(kᵘᵘ)
𝑎₂(kᵘᵛ)
𝑎₃(kᵛᵛ)
𝑎₄(kᵛᵘ)

kᵅ = zeros(nₚ,nₚ)
fᵅ = zeros(nₚ)
elements_Γ¹ = getElements(nodes, entities["Γ¹"], integrationorder)
elements_Γ⁴ = getElements(nodes, entities["Γ⁴"], integrationorder)
prescribe!(elements_Γ¹,:α=>α,:g=>0.0)
prescribe!(elements_Γ⁴,:α=>α,:g=>0.0)
set𝝭!(elements_Γ¹)
set𝝭!(elements_Γ⁴)
𝑎 = ∫vgdΓ=>elements_Γ¹∪elements_Γ⁴
𝑎(kᵅ,fᵅ)


kᵝ = zeros(nₚ,nₚ)
fᵝ = zeros(nₚ)
elements_Γ¹ = getElements(nodes, entities["Γ¹"], integrationorder)
prescribe!(elements_Γ¹,:α=>α,:g=>1.0)
set𝝭!(elements_Γ¹)
𝑎 = ∫vgdΓ=>elements_Γ¹
𝑎(kᵝ,fᵝ)

d = [kᵘᵘ+kᵅ kᵘᵛ;kᵛᵘ kᵛᵛ+kᵝ]\[fᵅ;fᵝ]
push!(nodes,:d=>d)

xs = [node.x for node in nodes]'
ys = [node.y for node in nodes]'
zs = [node.z for node in nodes]'
points = [xs; ys; zs]
cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, [xᵢ.𝐼 for xᵢ in elm.𝓒]) for elm in elements]

nₑ = length(elements)
σ = zeros(nₑ)
for (j,p) in enumerate(elements)
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

 vtk_grid("./vtk/impact_bar/"*filename, points, cells) do vtk
     vtk["位移"] = d
     vtk["应力"] = σ
 end
