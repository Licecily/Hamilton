using ApproxOperator, LinearAlgebra
import ApproxOperator.WeightedResidual: ∫∫kNṄdxdt, ∫∫kNNdxdt, ∫∫NṄdxdt, ∫∫c²B₁B₁dxdt
import ApproxOperator.Heat: ∫vgdΓ
using ApproxOperator.GmshImport: getPhysicalGroups, getElements, get𝑿ᵢ
using WriteVTK
import Gmsh: gmsh

# ==================== 参数 ====================
ρA = 1.0
EA = 1.0
c  = sqrt(EA/ρA)               # 波速
L  = 2.0                       # 空间长度（x∈[0,2]，t∈[0,2]）
α  = 1e6                       # 固定端 u=0、v=0 罚系数
φ(x) = sin(π*x/L)              # 初始位移 u(x,0)
ψ(x) = 0.0                     # 初始速度 v(x,0)
u_exact(x,t) = cos(π*c*t/L)*sin(π*x/L)   # 精确解（固定-固定，一阶模态）

integrationorder = 2
ndiv = 32
filename = "DG_"*string(ndiv)

gmsh.initialize()
gmsh.open("./msh/DG/"*filename*".msh")
entities = getPhysicalGroups()
nodes = get𝑿ᵢ()
nₚ = length(nodes)

# ==================== 单元 ====================
elements_Ω  = getElements(nodes, entities["Ω"],  integrationorder)
elements_Γ¹ = getElements(nodes, entities["Γ¹"], integrationorder)   # t=0（初值）
elements_Γ² = getElements(nodes, entities["Γ²"], integrationorder)   # x=L（右）
elements_Γ³⁻ = getElements(nodes, entities["Γ³⁻"], integrationorder) # 界面下侧
elements_Γ³⁺ = getElements(nodes, entities["Γ³⁺"], integrationorder) # 界面上侧
elements_Γ⁴ = getElements(nodes, entities["Γ⁴"], integrationorder)   # x=0（左）

# 按质心 y 坐标拆成下、上两个时间层（分界 y = L/2 = 1）
Ω₀  = [e for e in elements_Ω  if sum(x.y for x in e.𝓒)/length(e.𝓒) <= L/2]
Ω₁  = [e for e in elements_Ω  if sum(x.y for x in e.𝓒)/length(e.𝓒) >  L/2]
Γ²₀ = [e for e in elements_Γ² if sum(x.y for x in e.𝓒)/length(e.𝓒) <= L/2]
Γ²₁ = [e for e in elements_Γ² if sum(x.y for x in e.𝓒)/length(e.𝓒) >  L/2]
Γ⁴₀ = [e for e in elements_Γ⁴ if sum(x.y for x in e.𝓒)/length(e.𝓒) <= L/2]
Γ⁴₁ = [e for e in elements_Γ⁴ if sum(x.y for x in e.𝓒)/length(e.𝓒) >  L/2]

# ==================== slab0（初值：u(t₀⁻)=φ，v(t₀⁻)=ψ） ====================
kᵘᵘ = zeros(nₚ,nₚ); kᵘᵛ = zeros(nₚ,nₚ); kᵛᵛ = zeros(nₚ,nₚ); kᵛᵘ = zeros(nₚ,nₚ)
prescribe!(Ω₀, :k=>EA, :c=>c)
set∇𝝭!(Ω₀)
𝑎₁ = ∫∫kNṄdxdt    => Ω₀
𝑎₂ = ∫∫kNNdxdt    => Ω₀
𝑎₃ = ∫∫NṄdxdt     => Ω₀
𝑎₄ = ∫∫c²B₁B₁dxdt => Ω₀
𝑎₁(kᵘᵘ)
𝑎₂(kᵘᵛ)
𝑎₃(kᵛᵛ)
𝑎₄(kᵛᵘ)

# 初值罚（Γ¹），矩阵与荷载一起装：K^α=k∫NN、f^α=k∫N φ；K^β=∫NN、f^β=∫N ψ
kᵅ = zeros(nₚ,nₚ); kᵝ = zeros(nₚ,nₚ)
fᵅ = zeros(nₚ);     fᵝ = zeros(nₚ)
prescribe!(elements_Γ¹, :α=>EA,  :g=>(x,y,z)->φ(x))
set𝝭!(elements_Γ¹)
𝑎ᵅ = ∫vgdΓ => elements_Γ¹; 𝑎ᵅ(kᵅ, fᵅ)
prescribe!(elements_Γ¹, :α=>1.0, :g=>(x,y,z)->ψ(x))
𝑎ᵝ = ∫vgdΓ => elements_Γ¹; 𝑎ᵝ(kᵝ, fᵝ)

# 固定端 u=0、v=0
prescribe!(Γ⁴₀, :α=>α, :g=>0.0)
prescribe!(Γ²₀, :α=>α, :g=>0.0)
set𝝭!(Γ⁴₀); set𝝭!(Γ²₀)
𝑎ᵅ = ∫vgdΓ => Γ⁴₀ ∪ Γ²₀
𝑎ᵅ(kᵅ, zeros(nₚ))   # u=0
𝑎ᵅ(kᵝ, zeros(nₚ))   # v=0

# 解 slab0
dofs₀ = sort(collect(getDOFs(Ω₀)))
n₀ = length(dofs₀)
k₀ = [kᵘᵘ[dofs₀,dofs₀]+kᵅ[dofs₀,dofs₀]   kᵘᵛ[dofs₀,dofs₀];
      kᵛᵘ[dofs₀,dofs₀]                   kᵛᵛ[dofs₀,dofs₀]+kᵝ[dofs₀,dofs₀]]
dt₀ = k₀\[fᵅ[dofs₀]; fᵝ[dofs₀]]
d₀ = zeros(nₚ); δd₀ = zeros(nₚ)
d₀[dofs₀] = dt₀[1:n₀]; δd₀[dofs₀] = dt₀[n₀+1:end]

# ==================== slab1（跳变：上游为 slab0 顶边解） ====================
kᵘᵘ = zeros(nₚ,nₚ); kᵘᵛ = zeros(nₚ,nₚ); kᵛᵛ = zeros(nₚ,nₚ); kᵛᵘ = zeros(nₚ,nₚ)
prescribe!(Ω₁, :k=>EA, :c=>c)
set∇𝝭!(Ω₁)
𝑎₁ = ∫∫kNṄdxdt    => Ω₁
𝑎₂ = ∫∫kNNdxdt    => Ω₁
𝑎₃ = ∫∫NṄdxdt     => Ω₁
𝑎₄ = ∫∫c²B₁B₁dxdt => Ω₁
𝑎₁(kᵘᵘ)
𝑎₂(kᵘᵛ)
𝑎₃(kᵛᵛ)
𝑎₄(kᵛᵘ)

# 跳变罚矩阵（Γ³⁺ 底边）：K^α = k∫NN、K^β = ∫NN
kᵅ = zeros(nₚ,nₚ); kᵝ = zeros(nₚ,nₚ)
fᵅ = zeros(nₚ);     fᵝ = zeros(nₚ)
prescribe!(elements_Γ³⁺, :α=>EA,  :g=>0.0)
set𝝭!(elements_Γ³⁺)
𝑎ᵅ = ∫vgdΓ => elements_Γ³⁺; 𝑎ᵅ(kᵅ, zeros(nₚ))
prescribe!(elements_Γ³⁺, :α=>1.0, :g=>0.0)
𝑎ᵝ = ∫vgdΓ => elements_Γ³⁺; 𝑎ᵝ(kᵝ, zeros(nₚ))

# 跳变荷载：由上游解插值 u(t_n⁻), v(t_n⁻)，装配 fᵅ = k∫N u⁻、fᵝ = ∫N v⁻
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
            fᵅ[I] += EA*Nc[i]*u⁻*w
            fᵝ[I] += Nc[i]*v⁻*w
        end
    end
end

# 固定端 u=0、v=0
prescribe!(Γ⁴₁, :α=>α, :g=>0.0)
prescribe!(Γ²₁, :α=>α, :g=>0.0)
set𝝭!(Γ⁴₁); set𝝭!(Γ²₁)
𝑎ᵅ = ∫vgdΓ => Γ⁴₁ ∪ Γ²₁
𝑎ᵅ(kᵅ, zeros(nₚ))   # u=0
𝑎ᵅ(kᵝ, zeros(nₚ))   # v=0

# 解 slab1
dofs₁ = sort(collect(getDOFs(Ω₁)))
n₁ = length(dofs₁)
k₁ = [kᵘᵘ[dofs₁,dofs₁]+kᵅ[dofs₁,dofs₁]   kᵘᵛ[dofs₁,dofs₁];
      kᵛᵘ[dofs₁,dofs₁]                   kᵛᵛ[dofs₁,dofs₁]+kᵝ[dofs₁,dofs₁]]
dt₁ = k₁\[fᵅ[dofs₁]; fᵝ[dofs₁]]
d₁ = zeros(nₚ); δd₁ = zeros(nₚ)
d₁[dofs₁] = dt₁[1:n₁]; δd₁[dofs₁] = dt₁[n₁+1:end]

# ==================== 合并 ====================
d = d₀ + d₁
δd = δd₀ + δd₁
push!(nodes, :d=>d, :δd=>δd)

# ==================== 输出 ====================
# 界面跳变 [[u]] = u(t⁺) - u(t⁻)
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
println("节点数 nₚ = ", nₚ, "，时间层 = 2，界面最大跳变 [[u]] = ", jump)

points = zeros(3,nₚ)
for (i,node) in enumerate(nodes)
    points[1,i] = node.x
    points[2,i] = node.y
    points[3,i] = node.d
end
cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, [x.𝐼 for x in elm.𝓒]) for elm in vcat(Ω₀, Ω₁)]
mkpath("./vtk/DG_Continuous")
vtk_grid("./vtk/DG_Continuous/"*"DG_Continuous_"*string(ndiv)*".vtu", points, cells) do vtk
    vtk["u"]       = [node.d  for node in nodes]
    vtk["v"]       = [node.δd for node in nodes]
    vtk["u_exact"] = [u_exact(node.x, node.y) for node in nodes]
end
