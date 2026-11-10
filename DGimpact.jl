using ApproxOperator, LinearAlgebra
import ApproxOperator.WeightedResidual: ∫∫kNṄdxdt, ∫∫kNNdxdt, ∫∫NṄdxdt, ∫∫c²B₁B₁dxdt
import ApproxOperator.Heat: ∫vgdΓ
using ApproxOperator.GmshImport: getPhysicalGroups, getElements, get𝑿ᵢ
using WriteVTK
import Gmsh: gmsh

# ==================== 参数 ====================
ρA = 1.0; EA = 1.0; c = sqrt(EA/ρA)
L  = 1.0
α  = 1e6                       # 固定端/初值/顶边 罚系数
k_jump = 1e10                  # 时间层间跳变罚系数
u₀(x) = 0.0
v₀(x) = 1.0

integrationorder = 2
ndiv = 32
filename = "DG_impact_"*string(ndiv)

gmsh.initialize()
gmsh.open("./msh/DG/DG_impact_bar/"*filename*".msh")
entities = getPhysicalGroups()
nodes = get𝑿ᵢ()
nₚ = length(nodes)

# ==================== 单元 ====================
elements_Ω   = getElements(nodes, entities["Ω"],   integrationorder)
elements_Γ¹  = getElements(nodes, entities["Γ¹"],  integrationorder)
elements_Γ²  = getElements(nodes, entities["Γ²"],  integrationorder)
elements_Γ³⁻ = getElements(nodes, entities["Γ³⁻"], integrationorder)
elements_Γ³⁺ = getElements(nodes, entities["Γ³⁺"], integrationorder)
elements_Γ⁴  = getElements(nodes, entities["Γ⁴"],  integrationorder)
elements_Γ⁵  = getElements(nodes, entities["Γ⁵"],  integrationorder)

# 按质心 y 坐标拆分时间层
Ω₀  = [e for e in elements_Ω  if sum(x.y for x in e.𝓒)/length(e.𝓒) <= L]
Ω₁  = [e for e in elements_Ω  if sum(x.y for x in e.𝓒)/length(e.𝓒) >  L]
Γ⁴₀ = [e for e in elements_Γ⁴ if sum(x.y for x in e.𝓒)/length(e.𝓒) <= L]
Γ⁴₁ = [e for e in elements_Γ⁴ if sum(x.y for x in e.𝓒)/length(e.𝓒) >  L]

println("nₚ = ", nₚ, "  Ω₀ = ", length(Ω₀), "  Ω₁ = ", length(Ω₁))

# ==================== slab0（初始层0） ====================
kᵘᵘ = zeros(nₚ,nₚ); kᵘᵛ = zeros(nₚ,nₚ); kᵛᵛ = zeros(nₚ,nₚ); kᵛᵘ = zeros(nₚ,nₚ)
prescribe!(Ω₀, :k=>EA, :c=>c); set∇𝝭!(Ω₀)
(∫∫kNṄdxdt    => Ω₀)(kᵘᵘ)
(∫∫kNNdxdt    => Ω₀)(kᵘᵛ)
(∫∫NṄdxdt     => Ω₀)(kᵛᵛ)
(∫∫c²B₁B₁dxdt => Ω₀)(kᵛᵘ)

kᵅ = zeros(nₚ,nₚ); kᵝ = zeros(nₚ,nₚ); fᵅ = zeros(nₚ); fᵝ = zeros(nₚ)
prescribe!(elements_Γ¹, :α=>EA,  :g=>(x,y,z)->u₀(x)); set𝝭!(elements_Γ¹)
(∫vgdΓ => elements_Γ¹)(kᵅ, fᵅ)
prescribe!(elements_Γ¹, :α=>1.0, :g=>(x,y,z)->v₀(x)); set𝝭!(elements_Γ¹)
(∫vgdΓ => elements_Γ¹)(kᵝ, fᵝ)

prescribe!(Γ⁴₀, :α=>α, :g=>0.0); set𝝭!(Γ⁴₀)
(∫vgdΓ => Γ⁴₀)(kᵅ, zeros(nₚ))
(∫vgdΓ => Γ⁴₀)(kᵝ, zeros(nₚ))

dofs₀ = sort(collect(getDOFs(Ω₀)))
n₀ = length(dofs₀)
k₀ = [kᵘᵘ[dofs₀,dofs₀]+kᵅ[dofs₀,dofs₀]   kᵘᵛ[dofs₀,dofs₀];
      kᵛᵘ[dofs₀,dofs₀]                   kᵛᵛ[dofs₀,dofs₀]+kᵝ[dofs₀,dofs₀]]
dt₀ = k₀\[fᵅ[dofs₀]; fᵝ[dofs₀]]
d₀ = zeros(nₚ); δd₀ = zeros(nₚ)
d₀[dofs₀] = dt₀[1:n₀]; δd₀[dofs₀] = dt₀[n₀+1:end]

println("slab0: u∈[", round(minimum(d₀),digits=4), ", ", round(maximum(d₀),digits=4), "]")

# ==================== slab1（时间层1） ====================
kᵘᵘ = zeros(nₚ,nₚ); kᵘᵛ = zeros(nₚ,nₚ); kᵛᵛ = zeros(nₚ,nₚ); kᵛᵘ = zeros(nₚ,nₚ)
prescribe!(Ω₁, :k=>EA, :c=>c); set∇𝝭!(Ω₁)
(∫∫kNṄdxdt    => Ω₁)(kᵘᵘ)
(∫∫kNNdxdt    => Ω₁)(kᵘᵛ)
(∫∫NṄdxdt     => Ω₁)(kᵛᵛ)
(∫∫c²B₁B₁dxdt => Ω₁)(kᵛᵘ)

kᵅ = zeros(nₚ,nₚ); kᵝ = zeros(nₚ,nₚ); fᵅ = zeros(nₚ); fᵝ = zeros(nₚ)

# 跳变罚（Γ³⁺）：K^α = k_jump·∫NN、K^β = k_jump·∫NN
prescribe!(elements_Γ³⁺, :α=>k_jump, :g=>0.0); set𝝭!(elements_Γ³⁺)
(∫vgdΓ => elements_Γ³⁺)(kᵅ, zeros(nₚ))
prescribe!(elements_Γ³⁺, :α=>k_jump, :g=>0.0); set𝝭!(elements_Γ³⁺)
(∫vgdΓ => elements_Γ³⁺)(kᵝ, zeros(nₚ))

# 跳变荷载：f^α = k_jump·∫N u⁻、f^β = k_jump·∫N v⁻
set𝝭!(elements_Γ³⁻)
for (ec, ep) in zip(elements_Γ³⁺, elements_Γ³⁻)
    for (ξc, ξp) in zip(ec.𝓖, ep.𝓖)
        Nc = ξc[:𝝭]; Np = ξp[:𝝭]; w = ξc.𝑤
        u⁻ = 0.0; v⁻ = 0.0
        for (j,xj) in enumerate(ep.𝓒)
            u⁻ += Np[j]*d₀[xj.𝐼]
            v⁻ += Np[j]*δd₀[xj.𝐼]
        end
        for (i,xi) in enumerate(ec.𝓒)
            I = xi.𝐼
            fᵅ[I] += k_jump*Nc[i]*u⁻*w
            fᵝ[I] += k_jump*Nc[i]*v⁻*w
        end
    end
end

# 固定端 x=0
prescribe!(Γ⁴₁, :α=>α, :g=>0.0); set𝝭!(Γ⁴₁)
(∫vgdΓ => Γ⁴₁)(kᵅ, zeros(nₚ))
(∫vgdΓ => Γ⁴₁)(kᵝ, zeros(nₚ))

# 顶边 y=2
prescribe!(elements_Γ⁵, :α=>α, :g=>0.0); set𝝭!(elements_Γ⁵)
(∫vgdΓ => elements_Γ⁵)(kᵅ, zeros(nₚ))

dofs₁ = sort(collect(getDOFs(Ω₁)))
n₁ = length(dofs₁)
k₁ = [kᵘᵘ[dofs₁,dofs₁]+kᵅ[dofs₁,dofs₁]   kᵘᵛ[dofs₁,dofs₁];
      kᵛᵘ[dofs₁,dofs₁]                   kᵛᵛ[dofs₁,dofs₁]+kᵝ[dofs₁,dofs₁]]
println("slab1 cond = ", round(cond(k₁), digits=2))
dt₁ = k₁\[fᵅ[dofs₁]; fᵝ[dofs₁]]
d₁ = zeros(nₚ); δd₁ = zeros(nₚ)
d₁[dofs₁] = dt₁[1:n₁]; δd₁[dofs₁] = dt₁[n₁+1:end]

println("slab1: u∈[", round(minimum(d₁),digits=4), ", ", round(maximum(d₁),digits=4), "]")

# ==================== 合并 ====================
d  = zeros(nₚ)
δd = zeros(nₚ)
for i in dofs₀; d[i] = d₀[i]; δd[i] = δd₀[i]; end
for i in dofs₁; d[i] = d₁[i]; δd[i] = δd₁[i]; end

# ==================== 界面跳变 ====================
jump = let j = 0.0
    for (ec, ep) in zip(elements_Γ³⁺, elements_Γ³⁻)
        for (ξc, ξp) in zip(ec.𝓖, ep.𝓖)
            u⁺ = 0.0; u⁻ = 0.0
            for (i,xi) in enumerate(ec.𝓒); u⁺ += ξc[:𝝭][i]*d[xi.𝐼]; end
            for (j,xj) in enumerate(ep.𝓒); u⁻ += ξp[:𝝭][j]*d[xj.𝐼]; end
            j = max(j, abs(u⁺-u⁻))
        end
    end
    j
end
println("界面跳变 [[u]] = ", round(jump, digits=6))

# ==================== 诊断 ====================
using Printf
println("\n关键位置:")
for (xval,yval) in [(0.0,0.0),(1.0,0.0),(1.0,1.0),(1.0,2.0),(0.0,2.0)]
    for i in 1:nₚ
        if abs(nodes[i].x-xval)<1e-3 && abs(nodes[i].y-yval)<1e-3
            println("  x=$(round(xval,digits=1)), y=$(round(yval,digits=1)): u=", round(d[i],digits=6))
        end
    end
end

# ==================== 应力 ====================
Ωall = vcat(Ω₀, Ω₁)
set∇𝝭!(Ωall)
nₑ = length(Ωall)
σ = zeros(nₑ)
for (j, p) in enumerate(Ωall)
    σ_ = 0.0; w_ = 0.0
    for ξ in p.𝓖
        B₁ = ξ[:∂𝝭∂x]; ε = 0.0; w = ξ.𝑤
        for (i, xᵢ) in enumerate(p.𝓒); ε += B₁[i]*d[xᵢ.𝐼]; end
        σ_ += EA*ε*w; w_ += w
    end
    σ[j] = σ_/w_
end
println("\n应力: [", round(minimum(σ),digits=4), ", ", round(maximum(σ),digits=4), "]")

# ==================== VTK 输出 ====================
points = zeros(3, nₚ)
for (i, node) in enumerate(nodes)
    points[1,i] = node.x; points[2,i] = node.y; points[3,i] = node.z
end
cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, [x.𝐼 for x in elm.𝓒]) for elm in Ωall]
mkpath("./VTK/DG_impact_bar")
vtk_grid("./VTK/DG_impact_bar/"*filename*".vtu", points, cells) do vtk
    vtk["u"] = d
    vtk["v"] = δd
    vtk["应力"] = σ
end
println("完成。VTU: ./VTK/DG_impact_bar/"*filename*".vtu")
gmsh.finalize()