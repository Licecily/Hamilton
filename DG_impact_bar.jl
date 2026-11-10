using ApproxOperator, LinearAlgebra
import ApproxOperator.WeightedResidual: ∫∫kNṄdxdt, ∫∫kNNdxdt, ∫∫NṄdxdt, ∫∫c²B₁B₁dxdt
import ApproxOperator.Heat: ∫vgdΓ
using ApproxOperator.GmshImport: getPhysicalGroups, getElements, get𝑿ᵢ
using WriteVTK
import Gmsh: gmsh

# ==================== 参数 ====================
ρA = 1.0; EA = 1.0; c = sqrt(EA/ρA)
α  = 1e6                       # 固定端 x=0 的罚系数（u=0 且 v=0）
α_T = 0.0                      # 末时刻边 Γ⁵ 的罚系数（u(x,T)=0）
                               #   0   → 不加条件。DG-in-time 中最后一层的顶边是出流边界，
                               #         本就不需要条件（u(x,T)=0 是精确解的推论，不是边界条件）。
                               #   1e6 → 强行施加。离散解在 T 处有 O(Δy) 误差，强罚会在最后一层
                               #         激起全局伪振荡：σ 的 L2 误差 0.095 → 0.191，σ 幅值被
                               #         抬到 1.96（精确解最大只有 1.0），云图上就是一片碎斑。
k_jump = 1e10                  # 时间层间跳变罚系数
                               #   1e10 → 强罚，界面 u、v 严格连续（[[u]]=0），应力更干净
                               #   EA   → 标准 DG 弱跳变，界面留 O(h) 跳变：u 更准但应力更脏
                               #   本算例 N=4, nelx=32（σ 为离波前 >0.05 单元的 L2 误差）：
                               #     1e10: max|u-u_exact|=0.0487  ‖u‖_L2=0.0129  ‖σ‖_L2=0.0946
                               #     EA  : max|u-u_exact|=0.0444  ‖u‖_L2=0.0096  ‖σ‖_L2=0.1089
u₀(x) = 0.0                    # 初位移 u(x,0)
v₀(x) = 1.0                    # 初速度 v(x,0)

integrationorder = 2
Nslab = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 4   # 时间层数（须与网格一致）
ndiv  = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 32  # 水平方向单元数
filename = "DG_impact_N"*string(Nslab)*"_"*string(ndiv)

# ==================== 网格 ====================
gmsh.initialize()
gmsh.open("./msh/DG/DG_impact_bar/"*filename*".msh")
entities = getPhysicalGroups()
nodes = get𝑿ᵢ()
nₚ = length(nodes)

elements_Ω   = getElements(nodes, entities["Ω"],   integrationorder)
elements_Γ¹  = getElements(nodes, entities["Γ¹"],  integrationorder)   # 底边 y=0（初值）
elements_Γ²  = getElements(nodes, entities["Γ²"],  integrationorder)   # 右边 x=1（自由端，不施加）
elements_Γ³⁻ = getElements(nodes, entities["Γ³⁻"], integrationorder)   # 时间界面下侧
elements_Γ³⁺ = getElements(nodes, entities["Γ³⁺"], integrationorder)   # 时间界面上侧
elements_Γ⁴  = getElements(nodes, entities["Γ⁴"],  integrationorder)   # 左边 x=0（固定端）
elements_Γ⁵  = getElements(nodes, entities["Γ⁵"],  integrationorder)   # 顶边 y=T（末时刻）

# 单元质心
cx(e) = sum(xᵢ.x for xᵢ in e.𝓒)/length(e.𝓒)
cy(e) = sum(xᵢ.y for xᵢ in e.𝓒)/length(e.𝓒)

# ==================== 按 Γ³⁻ 自动识别时间层 ====================
yₜ = maximum(xᵢ.y for xᵢ in nodes)                             # 时间方向总长 T
yᵢⁿᵗ = sort(unique(round.(cy.(elements_Γ³⁻), digits=8)))       # 内部界面高度
Nmesh = length(yᵢⁿᵗ) + 1
@assert Nmesh == Nslab "网格层数 $Nmesh 与 Nslab = $Nslab 不一致"
yedge = vcat(0.0, yᵢⁿᵗ, yₜ)                                    # N+1 个分界

Ωₖ  = [[e for e in elements_Ω  if yedge[k] < cy(e) < yedge[k+1]] for k in 1:Nslab]
Γ⁴ₖ = [[e for e in elements_Γ⁴ if yedge[k] < cy(e) < yedge[k+1]] for k in 1:Nslab]

# 每个界面：上侧（Γ³⁺，第 k 层底边）与下侧（Γ³⁻，第 k-1 层顶边）
iface_up = [[e for e in elements_Γ³⁺ if isapprox(cy(e), yᵢⁿᵗ[i]; atol=1e-8)] for i in 1:Nslab-1]
iface_dn = [[e for e in elements_Γ³⁻ if isapprox(cy(e), yᵢⁿᵗ[i]; atol=1e-8)] for i in 1:Nslab-1]
for i in 1:Nslab-1
    sort!(iface_up[i], by=cx); sort!(iface_dn[i], by=cx)       # 上下按 x 质心一一对应
    @assert length(iface_up[i]) == length(iface_dn[i])
    @assert all(abs.(cx.(iface_up[i]) .- cx.(iface_dn[i])) .< 1e-8) "界面 $i 上下单元不重合"
end

println("nₚ = ", nₚ, "，时间层 N = ", Nslab, "，界面高度 = ", yᵢⁿᵗ)
for k in 1:Nslab
    println("  slab", k-1, ": y ∈ [", yedge[k], ", ", yedge[k+1], "]，单元 ", length(Ωₖ[k]),
            "，左边 ", length(Γ⁴ₖ[k]))
end

# ==================== 形函数/材料（按整个物理组一次性 set，供各层子集复用） ====================
# 注意：set𝝭! / prescribe! 对 Vector 会按「最后一个单元」重新分配底层数组，
#       因此这里只在完整物理组上调用一次，之后各层子集只读不再 set。
prescribe!(elements_Ω, :k=>EA, :c=>c); set∇𝝭!(elements_Ω)
prescribe!(elements_Γ⁴, :α=>α, :g=>0.0); set𝝭!(elements_Γ⁴)      # 固定端 u=0、v=0
prescribe!(elements_Γ³⁺, :α=>k_jump, :g=>0.0); set𝝭!(elements_Γ³⁺)  # 跳变罚 K=k_jump∫NN
prescribe!(elements_Γ⁵, :α=>α_T, :g=>0.0); set𝝭!(elements_Γ⁵)     # 末时刻（默认不加条件）
set𝝭!(elements_Γ³⁻)                                              # 下侧只用来插值上游解
set𝝭!(elements_Γ¹)

# ==================== 跳变荷载 ====================
# 界面上、下单元分别属于两层各自的网格，节点不共享；
# 且 geo 中底边由左向右、顶边由右向左，两侧积分点顺序相反，
# 所以按积分点的物理 x 坐标配对，不能按下标 zip。
function add_jump_load!(fᵅ, fᵝ, up, dn, d⁻, δd⁻)
    for (ec, ep) in zip(up, dn)
        for ξc in ec.𝓖
            ξp = ep.𝓖[argmin([abs(ξp_.x - ξc.x) for ξp_ in ep.𝓖])]
            @assert abs(ξp.x - ξc.x) < 1e-8 "界面上下积分点未对齐"
            Nc = ξc[:𝝭]; Np = ξp[:𝝭]; 𝑤 = ξc.𝑤
            u⁻ = 0.0; v⁻ = 0.0
            for (l, xl) in enumerate(ep.𝓒)
                u⁻ += Np[l]*d⁻[xl.𝐼]
                v⁻ += Np[l]*δd⁻[xl.𝐼]
            end
            for (i, xi) in enumerate(ec.𝓒)
                fᵅ[xi.𝐼] += k_jump*Nc[i]*u⁻*𝑤
                fᵝ[xi.𝐼] += k_jump*Nc[i]*v⁻*𝑤
            end
        end
    end
end

# ==================== 逐层推进 ====================
d  = zeros(nₚ)                 # 节点位移 u
δd = zeros(nₚ)                 # 节点速度 v
dofsₖ = Vector{Vector{Int}}(undef, Nslab)

for k in 1:Nslab
    kᵘᵘ = zeros(nₚ,nₚ); kᵘᵛ = zeros(nₚ,nₚ); kᵛᵛ = zeros(nₚ,nₚ); kᵛᵘ = zeros(nₚ,nₚ)
    (∫∫kNṄdxdt    => Ωₖ[k])(kᵘᵘ)
    (∫∫kNNdxdt    => Ωₖ[k])(kᵘᵛ)
    (∫∫NṄdxdt     => Ωₖ[k])(kᵛᵛ)
    (∫∫c²B₁B₁dxdt => Ωₖ[k])(kᵛᵘ)

    kᵅ = zeros(nₚ,nₚ); kᵝ = zeros(nₚ,nₚ); fᵅ = zeros(nₚ); fᵝ = zeros(nₚ)

    if k == 1
        # ---- 初值（Γ¹）：K^α = EA∫NN、f^α = EA∫N u₀；K^β = ∫NN、f^β = ∫N v₀ ----
        prescribe!(elements_Γ¹, :α=>EA,  :g=>(x,y,z)->u₀(x))
        (∫vgdΓ => elements_Γ¹)(kᵅ, fᵅ)
        prescribe!(elements_Γ¹, :α=>1.0, :g=>(x,y,z)->v₀(x))
        (∫vgdΓ => elements_Γ¹)(kᵝ, fᵝ)
    else
        # ---- 时间层间跳变：罚矩阵（上侧底边）+ 上游荷载（下侧顶边） ----
        (∫vgdΓ => iface_up[k-1])(kᵅ, zeros(nₚ))
        (∫vgdΓ => iface_up[k-1])(kᵝ, zeros(nₚ))
        add_jump_load!(fᵅ, fᵝ, iface_up[k-1], iface_dn[k-1], d, δd)
    end

    # ---- 固定端 x=0：u=0 且 v=0 ----
    (∫vgdΓ => Γ⁴ₖ[k])(kᵅ, zeros(nₚ))
    (∫vgdΓ => Γ⁴ₖ[k])(kᵝ, zeros(nₚ))

    # ---- 末时刻边 y=T：出流边界，默认不加条件（α_T>0 时才强加 u=0） ----
    if k == Nslab && α_T > 0
        (∫vgdΓ => elements_Γ⁵)(kᵅ, zeros(nₚ))
    end

    dofsₖ[k] = sort(collect(getDOFs(Ωₖ[k])))
    if k > 1
        @assert isempty(intersect(dofsₖ[k], dofsₖ[k-1])) "层间节点被合并，DG 跳变格式失效"
    end
    n = length(dofsₖ[k])
    K = [kᵘᵘ[dofsₖ[k],dofsₖ[k]]+kᵅ[dofsₖ[k],dofsₖ[k]]   kᵘᵛ[dofsₖ[k],dofsₖ[k]];
         kᵛᵘ[dofsₖ[k],dofsₖ[k]]                         kᵛᵛ[dofsₖ[k],dofsₖ[k]]+kᵝ[dofsₖ[k],dofsₖ[k]]]
    dt = K\[fᵅ[dofsₖ[k]]; fᵝ[dofsₖ[k]]]
    d[dofsₖ[k]]  .= dt[1:n]
    δd[dofsₖ[k]] .= dt[n+1:end]

    println("slab", k-1, ": n = ", n, "  cond = ", round(cond(K), digits=2),
            "  u ∈ [", round(minimum(d[dofsₖ[k]]),digits=4), ", ",
                        round(maximum(d[dofsₖ[k]]),digits=4), "]")
end

# ==================== 界面跳变 ====================
for i in 1:Nslab-1
    jumpu = 0.0; jumpv = 0.0
    for (ec, ep) in zip(iface_up[i], iface_dn[i])
        for ξc in ec.𝓖
            ξp = ep.𝓖[argmin([abs(ξp_.x - ξc.x) for ξp_ in ep.𝓖])]
            u⁺ = 0.0; v⁺ = 0.0; u⁻ = 0.0; v⁻ = 0.0
            for (j,xj) in enumerate(ec.𝓒)
                u⁺ += ξc[:𝝭][j]*d[xj.𝐼]; v⁺ += ξc[:𝝭][j]*δd[xj.𝐼]
            end
            for (j,xj) in enumerate(ep.𝓒)
                u⁻ += ξp[:𝝭][j]*d[xj.𝐼]; v⁻ += ξp[:𝝭][j]*δd[xj.𝐼]
            end
            jumpu = max(jumpu, abs(u⁺-u⁻)); jumpv = max(jumpv, abs(v⁺-v⁻))
        end
    end
    println("界面 y=", yᵢⁿᵗ[i], ": max|[[u]]| = ", round(jumpu, digits=6),
            "，max|[[v]]| = ", round(jumpv, digits=6))
end

# ==================== 精确解校核 ====================
Ωall = reduce(vcat, Ωₖ)

# 固定-自由杆，c=1、L=1、u₀=0、v₀=1，用镜像法：V 为 x=0 奇延拓、x=1 偶延拓，
# 周期 4；F(s) = ∫₀ˢ V。u = (F(x+t)-F(x-t))/2，v = (V(x+t)+V(x-t))/2。
V_exact(s) = mod(s,4) < 2 ? 1.0 : -1.0
F_exact(s) = (r = mod(s,4); r <= 2 ? r : 4 - r)
u_exact(x,t) = (F_exact(x+t) - F_exact(x-t))/2
v_exact(x,t) = (V_exact(x+t) + V_exact(x-t))/2

eu = maximum(abs(d[i]  - u_exact(nodes[i].x, nodes[i].y)) for i in 1:nₚ)
ev = maximum(abs(δd[i] - v_exact(nodes[i].x, nodes[i].y)) for i in 1:nₚ)
println("\n与精确解对比: max|u-u_exact| = ", round(eu, digits=6),
        "，max|v-v_exact| = ", round(ev, digits=6))
println("（v 的精确解沿特征线 x±t=const 本身有间断，逐点最大误差没有意义）")

# u 的 L2 误差：每个三角形用 3 点求积（重心坐标 (2/3,1/6,1/6) 及其轮换）
L2 = let L2sq = 0.0
    for p in Ωall
        (x₁,y₁) = (p.𝓒[1].x, p.𝓒[1].y); (x₂,y₂) = (p.𝓒[2].x, p.𝓒[2].y); (x₃,y₃) = (p.𝓒[3].x, p.𝓒[3].y)
        area = abs((x₂-x₁)*(y₃-y₁) - (x₃-x₁)*(y₂-y₁))/2
        for λ in ((2/3,1/6,1/6), (1/6,2/3,1/6), (1/6,1/6,2/3))
            x = λ[1]*x₁ + λ[2]*x₂ + λ[3]*x₃
            y = λ[1]*y₁ + λ[2]*y₂ + λ[3]*y₃
            uₕ = λ[1]*d[p.𝓒[1].𝐼] + λ[2]*d[p.𝓒[2].𝐼] + λ[3]*d[p.𝓒[3].𝐼]
            L2sq += (uₕ - u_exact(x,y))^2 * area/3
        end
    end
    sqrt(L2sq)
end
println("‖u - u_exact‖_L2 = ", round(L2, digits=6))
println("关键位置 (精确解 -> 计算值):")
for (xval,yval) in [(0.0,0.0),(0.5,0.5),(1.0,1.0),(0.5,1.0),(0.5,1.5),(0.5,2.0),(1.0,2.0)]
    i = argmin([(nodes[i].x-xval)^2 + (nodes[i].y-yval)^2 for i in 1:nₚ])
    println("  (", xval, ", ", yval, "): u = ", round(u_exact(xval,yval),digits=6),
            " -> ", round(d[i],digits=6), "，v = ", round(v_exact(xval,yval),digits=6),
            " -> ", round(δd[i],digits=6))
end

# ==================== 应力 ====================
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
for k in 1:Nslab
    id = [j for (j,p) in enumerate(Ωall) if yedge[k] < cy(p) < yedge[k+1]]
    println("  slab", k-1, ": σ ∈ [", round(minimum(σ[id]),digits=3), ", ",
            round(maximum(σ[id]),digits=3), "]")
end

# σ = ∂u/∂x 的 L2 误差。σ 在特征线 x±t=const 上是真正间断的，骑在波前上的单元
# 无论网格多细都有 O(1) 的单元平均误差，所以只统计离波前 >0.05 的单元。
σ_exact(x,t) = (V_exact(x+t) - V_exact(x-t))/2
L2σ = let s = 0.0, a = 0.0
    for (j,p) in enumerate(Ωall)
        (x₁,y₁) = (p.𝓒[1].x, p.𝓒[1].y); (x₂,y₂) = (p.𝓒[2].x, p.𝓒[2].y); (x₃,y₃) = (p.𝓒[3].x, p.𝓒[3].y)
        area = abs((x₂-x₁)*(y₃-y₁) - (x₃-x₁)*(y₂-y₁))/2
        xc = (x₁+x₂+x₃)/3; yc = (y₁+y₂+y₃)/3
        if min(abs(xc-yc), abs(xc+yc-yₜ))/√2 > 0.05
            s += (σ[j] - σ_exact(xc,yc))^2 * area; a += area
        end
    end
    sqrt(s/a)
end
println("‖σ - σ_exact‖_L2（离波前 >0.05 的单元）= ", round(L2σ, digits=6))

# 节点平均应力：σ 是单元常值，直接画出来在 ParaView 里是一片按网格对角线排布的碎斑。
# 把单元应力在各节点上平均后作为点数据输出，ParaView 会自动线性插值 → 平滑云图。
# 而且它确实更接近精确解（逐点 RMS：单元常值 0.0871 → 节点平均 0.0629）。
σn = zeros(nₚ); cnt = zeros(Int, nₚ)
for (j, p) in enumerate(Ωall)
    for xᵢ in p.𝓒
        σn[xᵢ.𝐼] += σ[j]; cnt[xᵢ.𝐼] += 1
    end
end
σn ./= max.(cnt, 1)

# 逐点 RMS 对比（离特征线 >0.08 的采样点）
function elem_at(x, y)
    for (j, p) in enumerate(Ωall)
        (x₁,y₁) = (p.𝓒[1].x, p.𝓒[1].y); (x₂,y₂) = (p.𝓒[2].x, p.𝓒[2].y); (x₃,y₃) = (p.𝓒[3].x, p.𝓒[3].y)
        det = (y₂-y₃)*(x₁-x₃) + (x₃-x₂)*(y₁-y₃)
        λ₁ = ((y₂-y₃)*(x-x₃) + (x₃-x₂)*(y-y₃))/det
        λ₂ = ((y₃-y₁)*(x-x₃) + (x₁-x₃)*(y-y₃))/det
        λ₃ = 1 - λ₁ - λ₂
        if λ₁ ≥ -1e-9 && λ₂ ≥ -1e-9 && λ₃ ≥ -1e-9; return j, p, (λ₁,λ₂,λ₃); end
    end
    nothing, nothing, nothing
end
rms = let se = 0.0, sa = 0.0, m = 0
    for i in 1:120, j in 1:120
        x = (i-0.5)/120; y = (j-0.5)/120*yₜ
        min(abs(x-y), abs(x+y-yₜ))/√2 > 0.08 || continue
        jd, p, λ = elem_at(x,y)
        p === nothing && continue
        m += 1; ex = σ_exact(x,y)
        se += (σ[jd] - ex)^2
        sa += (sum(λ[k]*σn[p.𝓒[k].𝐼] for k in 1:3) - ex)^2
    end
    (sqrt(se/m), sqrt(sa/m), m)
end
println("应力逐点 RMS 误差（", rms[3], " 个采样点）：单元常值 = ", round(rms[1],digits=4),
        "，节点平均 = ", round(rms[2],digits=4))

# ==================== VTK 输出 ====================
points = zeros(3, nₚ)
for (i, node) in enumerate(nodes)
    points[1,i] = node.x; points[2,i] = node.y; points[3,i] = node.z
end
cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, [xᵢ.𝐼 for xᵢ in elm.𝓒]) for elm in Ωall]
mkpath("./VTK/DG_impact_bar")
vtk_grid("./VTK/DG_impact_bar/"*filename*".vtu", points, cells) do vtk
    vtk["u"] = d
    vtk["v"] = δd
    vtk["u_exact"] = [u_exact(node.x, node.y) for node in nodes]
    vtk["应力"] = σ            # 单元常值（cell data）
    vtk["应力_节点"] = σn      # 节点平均（point data，ParaView 里选这个画平滑云图）
end
println("完成。VTU: ./VTK/DG_impact_bar/"*filename*".vtu")
gmsh.finalize()
