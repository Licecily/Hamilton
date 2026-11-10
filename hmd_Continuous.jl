using  ApproxOperator, XLSX, LinearAlgebra, LinearSolve
import ApproxOperator.WeightedResidual: ∫∫kNṄdxdt, ∫∫kNNdxdt, ∫∫NṄdxdt, ∫∫c²B₁B₁dxdt
import ApproxOperator.Heat: ∫vtdΓ, ∫vgdΓ, ∫vbdΩ, L₂, ∫∫∇v∇udxdy, H₁
using ApproxOperator.GmshImport: getPhysicalGroups, getElements, get𝑿ᵢ
using WriteVTK
using GLMakie
import Gmsh: gmsh

α = 1e6
β = 1e6
# ρA = 1.0*1.0/289.0
ρA = 1.0
EA = 1.0
a = 1.0
l = 4.0
c = (EA/ρA)^0.5
φ(x) = sin(π*x/l)
𝑢(x,t) = cos.(π.*a.*c*t/l).*sin.(π.*x/l)
∂u∂t(x,t) = (-π.*a.*c/l)*sin.(π.*a.*c*t/l).*sin.(π.*x/l)
∂u∂x(x,t) = (π./l)*cos.(π.*a.*c*t/l).*cos.(π.*x/l)
# ∂²u∂t²(x,t) = -(π.*a./l)*(π.*a./l)*cos.(π.*a.*t/l).*sin.(π.*x/l)

integrationorder = 2
integrationorder_Ωᵍ = 10
ndiv= 32
filename = "square_"*string(ndiv)
gmsh.initialize()
gmsh.open("./msh/square/"*filename*".msh")
entities = getPhysicalGroups()
nodes = get𝑿ᵢ()
nₚ = length(nodes)
kᵘᵘ = zeros(nₚ,nₚ)
kᵘᵛ = zeros(nₚ,nₚ)
kᵛᵛ = zeros(nₚ,nₚ)
kᵛᵘ = zeros(nₚ,nₚ)

# elements,nodes = import_hmd_Tri3("./msh/Non-uniform/Tri3_"*string(ndiv)*".msh")
# elements,nodes = import_hmd_Tri3("./msh/square/square_"*string(ndiv)*".msh");uniform = "uniform"
# elements,nodes = import_hmd_Tri6("./msh/Non-uniform/Tri6/"*string(ndiv)*".msh");uniform = "uniform"
# elements,nodes = import_hmd_Tri6("./msh/square/Tri6_"*string(ndiv)*".msh")
# elements,nodes = import_hmd_Tri6("./msh/Non-uniform/拉伸压缩/2.5_"*string(ndiv)*".msh");uniform = "nonuniform"
# elements,nodes = import_hmd_Tri6("./msh/Non-uniform/RefineMesh_1.0/Tri6_"*string(ndiv)*".msh");uniform = "uniform"

# elements,nodes = import_hmd_Tri3("./msh/BiRefine/Continuous/square_4_r3_refined.msh");uniform = "uniform"

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
elements_Γ² = getElements(nodes, entities["Γ²"], integrationorder)
#elements_Γ³ = getElements(nodes, entities["Γ³"], integrationorder)
elements_Γ⁴ = getElements(nodes, entities["Γ⁴"], integrationorder)
prescribe!(elements_Γ¹,:α=>α,:g=>(x,y,z)->φ(x))
prescribe!(elements_Γ²,:α=>α,:g=>0.0)
#prescribe!(elements_Γ³,:α=>α,:g=>(x,y,z)->cos(π*a*c*y/l)*sin(π*x/l))
prescribe!(elements_Γ⁴,:α=>α,:g=>0.0)
set𝝭!(elements_Γ¹)
set𝝭!(elements_Γ²)
#set𝝭!(elements_Γ³)
set𝝭!(elements_Γ⁴)
#𝑎ᵅ = ∫vgdΓ=>elements_Γ¹∪elements_Γ²∪elements_Γ³∪elements_Γ⁴
𝑎ᵅ = ∫vgdΓ=>elements_Γ¹∪elements_Γ²∪elements_Γ⁴
𝑎ᵅ(kᵅ,fᵅ)

kᵝ = zeros(nₚ,nₚ)
fᵝ = zeros(nₚ)
elements_Γ¹ = getElements(nodes, entities["Γ¹"], integrationorder)
#elements_Γ² = getElements(nodes, entities["Γ²"], integrationorder)
#elements_Γ³ = getElements(nodes, entities["Γ³"], integrationorder)
#elements_Γ⁴ = getElements(nodes, entities["Γ⁴"], integrationorder)
prescribe!(elements_Γ¹,:α=>α,:g=>0.0)
#prescribe!(elements_Γ²,:α=>α,:g=>0.0)
#prescribe!(elements_Γ³,:α=>α,:g=>(x,y,z)->(-π*a*c/l)*sin(π*a*c*y/l)*sin(π*x/l))
#prescribe!(elements_Γ⁴,:α=>α,:g=>0.0)
set𝝭!(elements_Γ¹)
#set𝝭!(elements_Γ²)
#set𝝭!(elements_Γ³)
#set𝝭!(elements_Γ⁴)
#𝑎ᵝ = ∫vgdΓ=>elements_Γ¹∪elements_Γ²∪elements_Γ³∪elements_Γ⁴
𝑎ᵝ = ∫vgdΓ=>elements_Γ¹
𝑎ᵝ(kᵝ,fᵝ)

dt = [kᵘᵘ+kᵅ kᵘᵛ;kᵛᵘ kᵛᵛ+kᵝ]\[fᵅ;fᵝ]
d = dt[1:nₚ]
δd = dt[nₚ+1:end]
push!(nodes,:d=>d)
push!(nodes,:δd=>δd)

# ed = test_domain_error(elements["Ω"])
# e3 = test_boundary_error(elements["Γ₃ₜ"])
# println(ed)
# println(e3)

# 𝐿₂ = log10.(L₂(elements["Ωᵍ"]))
# 𝐻₁,𝐿₂ = log10.(H₁(elements["Ωᵍ"]))
# println(𝐻₁,𝐿₂)
nₑ = length(elements)
fig = Figure()
ax1 = Axis3(fig[1,1])
# ax2 = Axis3(fig[1,2])

xs = zeros(nₚ)
ys = zeros(nₚ)
zs = zeros(nₚ)
ds = zeros(nₚ)
# δds = zeros(nₚ)
# es = zeros(nₚ)
us = zeros(nₚ)
# qs = zeros(nₚ)
# as = zeros(nₚ)
for (i, node) in enumerate(nodes)
    x = node.x
    y = node.y
    us[i] = 𝑢(x,y)
    # qs[i] = ∂u∂t(x,y)
    # as[i] = ∂²u∂t²(x,y)
end
for (i,node) in enumerate(nodes)
    xs[i] = node.x
    ys[i] = node.y
    ds[i] = node.d
    # δds[i] = node.δd
    # es[i] = ds[i] - us[i]
end
face = zeros(nₑ,3)
for (i,elm) in enumerate(elements)
    face[i,:] .= [x.𝐼 for x in elm.𝓒]
end

# mesh!(ax,xs,ys,face,color=zs)
# meshscatter!(ax1,xs,ys,es,color=es,markersize = 0.1)
meshscatter!(ax1,xs,ys,ds,color=ds,markersize = 0.06)
# meshscatter!(ax1,xs,ys,us,color=us,markersize = 0.1)
# meshscatter!(ax2,xs,ys,δds,color=δds,markersize = 0.06)
fig

# save("./fig/测试/74测试/Tri6_非均布_LSG_c^2_4.png",fig)

# save("./fig/连续解/锁时间末端Tri_6非均布/t=19.png",fig)
# save("./fig/连续解/锁时间末端Tri_6均布/t=25.png",fig)
# save("./fig/连续解/mix_Tri_6均布/t=25.png",fig)
# save("./fig/连续解/mix_Tri_6非均布/n=41.png",fig)

# index = [4,8,16,32]
# # index = [5,10,20,40]
# XLSX.openxlsx("./excel/hmd_Continuous_BiRefine.xlsx", mode="rw") do xf
#     Sheet = xf[1]
#     ind = findfirst(n->n==ndiv,index)+1
#     Sheet["A"*string(ind)] = log10(4/ndiv)
#     # Sheet["A"*string(ind)] = log10(nₚ)
#     Sheet["B"*string(ind)] = 𝐻₁
#     Sheet["C"*string(ind)] = 𝐿₂
# end

 points = zeros(3,nₚ)
 for (i,node) in enumerate(nodes)
     points[1,i] = node.x
     points[2,i] = node.y
     points[3,i] = node.d
 end
 cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE_STRIP,[x.𝐼 for x in elm.𝓒]) for elm in elements]
 vtk_grid("./vtk/hmd_Continuous/lock3_uniform_"*string(ndiv)*".vtu",points,cells) do vtk
# vtk_grid("./vtk/hmd_Continuous/lock4_BiRefine_4.vtu",points,cells) do vtk
     vtk["d"] = [node.d for node in nodes]
 end
# xs = [node.x for node in nodes]'
# ys = [node.y for node in nodes]'
# zs = [node.z for node in nodes]'
# points = [xs; ys; zs]
# cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE_STRIP, [xᵢ.𝐼 for xᵢ in elm.𝓒]) for elm in elements["Ω"]]
# vtk_grid("./vtk/hmd_Continuous/error_uniform_"*string(ndiv), points, cells) do vtk
#     vtk["误差"] = es
#     # vtk["二阶导"] = as
# end
