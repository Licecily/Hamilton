using  ApproxOperator
using WriteVTK
import ApproxOperator.WeightedResidual: ∫kNṄdxdt, ∫kNNdxdt, ∫NṄdxdt, ∫c²B₁B₁dxdt
import ApproxOperator.Heat: ∫vtdΓ, ∫vgdΓ, ∫vbdΩ, L₂, ∫∫∇v∇udxdy, H₁
using ApproxOperator.GmshImport: getPhysicalGroups, getElements, get𝑿ᵢ
using GLMakie, XLSX, LinearAlgebra, LinearSolve
import Gmsh: gmsh

# ps = MKLPardisoSolver()
# set_matrixtype!(ps,2)

# ρA = 1.0*25.0/100.0
ρA = 1.0
EA = 1.0
α = 1e6
# β = 1e12
c = (EA/ρA)^0.5
𝑇(t) = t > 1.0 ? 0.0 : - sin(π*t)
function 𝑢(x,t)
    if x < t - 1
        return 2/π
    elseif x > t
        return 0.0
    else
        return (1-cos(π*(c*t - x)))/π
    end
end
function ∂u∂t(x, t)
    if x < t - 1 || x > t
        return 0.0
    else
        return sin(π * (c*t - x))
    end
end
function ∂u∂x(x, t)
    if x < t - 1
        return 0.0
    elseif x > t
        return 0.0
    else
        return -sin(π*(c*t - x))
    end
end
# function ∂²u∂t²(x, t)
#     if x < t - 1 || x > t
#         return 0.0
#     else
#         return π * cos(π * (t - x))
#     end
# end

integrationorder = 2
integrationorder_Ωᵍ = 10
ndiv= 0.04
filename = "tri3_"*string(ndiv)
gmsh.initialize()
gmsh.open("./msh/Non-uniform/局部加密/C=0.2/"*filename*".msh")
entities = getPhysicalGroups()
nodes = get𝑿ᵢ()
nₚ = length(nodes)
kᵘᵘ = zeros(nₚ,nₚ)
kᵘᵛ = zeros(nₚ,nₚ)
kᵛᵛ = zeros(nₚ,nₚ)
kᵛᵘ = zeros(nₚ,nₚ)


# elements,nodes = import_hmd_Tri6("./msh/Non-uniform/Tri6_"*string(ndiv)*".msh")
# elements,nodes = import_hmd_Tri6("./msh/Non-uniform/618/Tri6_0.5_"*string(ndiv)*".msh")
# elements,nodes = import_hmd_Tri6("./msh/square/square_"*string(ndiv)*".msh");uniform = "uniform"
# elements,nodes = import_hmd_Tri3("./msh/Non-uniform/Tri3_"*string(ndiv)*".msh");uniform = "uniform"
# elements,nodes = import_hmd_Tri3("./msh/Non-uniform/局部加密/C=0.2/Tri3_"*string(ndiv)*".msh");uniform = "uniform"
# elements,nodes = import_hmd_Tri6("./msh/Non-uniform/Tri6/"*string(ndiv)*".msh");uniform = "uniform"
# elements,nodes = import_hmd_Tri3("./msh/Non-uniform/拉伸压缩/2.1_"*string(ndiv)*".msh");uniform = "nonuniform"
# elements,nodes = import_hmd_Tri3("./msh/square/Tri3反向"*string(ndiv)*".msh");uniform = "uniform"
# elements,nodes = import_hmd_bar("./msh/bar/bar_"*string(ndiv)*".msh")

# elements,nodes = import_hmd_Tri3("./msh/BiRefine/2d/impact_4_refined_r13.msh");uniform = "uniform"


elements = getElements(nodes, entities["Ω"], integrationorder)
prescribe!(elements,:k=>EA,:c=>c)
set∇𝝭!(elements)
𝑎₁ = ∫kNṄdxdt => elements
𝑎₂ = ∫kNNdxdt => elements
𝑎₃ = ∫NṄdxdt => elements
𝑎₄ = ∫c²B₁B₁dxdt => elements
𝑎₁(kᵘᵘ)
𝑎₂(kᵘᵛ)
𝑎₃(kᵛᵛ)
𝑎₄(kᵛᵘ)

kᵅ = zeros(nₚ,nₚ)
fᵅ = zeros(nₚ)
elements_Γ¹ = getElements(nodes, entities["Γ¹"], integrationorder)
elements_Γ² = getElements(nodes, entities["Γ²"], integrationorder)
prescribe!(elements_Γ¹,:α=>α,:g=>0.0)
prescribe!(elements_Γ²,:α=>α,:g=>0.0)
set𝝭!(elements_Γ¹)
set𝝭!(elements_Γ²)
𝑎ᵅ = ∫vgdΓ=>elements_Γ¹∪elements_Γ²
𝑎ᵅ(kᵅ,fᵅ)

kᵝ = zeros(nₚ,nₚ)
fᵝ = zeros(nₚ)
elements_Γ¹ = getElements(nodes, entities["Γ¹"], integrationorder)
prescribe!(elements_Γ¹,:α=>α,:g=>0.0)
set𝝭!(elements_Γ¹)
𝑎ᵝ = ∫vgdΓ=>elements_Γ¹
𝑎ᵝ(kᵝ,fᵝ)

f = zeros(nₚ)
elements_Γ⁴ = getElements(nodes, entities["Γ⁴"], integrationorder)
prescribe!(elements_Γ⁴,:t=>(x,y,z)-> -c^2 * ∂u∂x(x, y))
set𝝭!(elements_Γ⁴)
𝑓 = ∫vtdΓ => elements_Γ⁴
𝑓(f)
# # kˢ = -k*kᵗ*k' + kᵝ
# kˢ = [k+kᵅ -k;-k kᵝ]
# C = condskeel(kˢ)
# println(C)

dt = [kᵘᵘ+kᵅ kᵘᵛ;kᵛᵘ kᵛᵛ+kᵝ]\[fᵅ;fᵝ+f]
d = dt[1:nₚ]
δd = dt[nₚ+1:end]
push!(nodes,:d=>d)
push!(nodes,:δd=>δd)
# dt = [k+kᵅ+kᵞ -k-kᵞ;-k-kᵞ kᵝ+kᵞ]\[fᵅ;-f+fᵝ]
# dt =(k+kᵅ)\(f+fᵅ)
# prob = LinearProblem([k+kᵅ+kᵞ -k-kᵞ;-k-kᵞ kᵝ+kᵞ], [fᵅ;-f+fᵝ])
# sol = solve(prob)
# dt = sol.u

# ed = test_domain_error(elements["Ω"])
# e3 = test_boundary_error(elements["Γ₃ₜ"])
# e4 = test_boundary_error(elements["Γ₄ₜ"])
# println(ed)
# println(e3)
# println(e4)

# 𝐿₂ = log10.(L₂(elements["Ωᵍ"]))
# 𝐻₁,𝐿₂ = log10.(H₁(elements["Ωᵍ"]))
# println(𝐻₁,𝐿₂)

# for i in 1:nₚ
#     x = nodes.x[i]
#     y = nodes.y[i]
#     d₁ = d[i]
#     # Δ = d[i] - 𝑢(x,y)
#         index = [10,20,40,80]
#         XLSX.openxlsx("./excel/square.xlsx", mode="rw") do xf
#         Sheet = xf[3]
#         ind = findfirst(n->n==ndiv,index)+1
#         # Sheet["A"*string(ind)] = x
#         # Sheet["B"*string(ind)] = y
#         # Sheet["C"*string(ind)] = d₁
#         # Sheet["D"*string(ind)] = Δ
#         Sheet["E"*string(ind)] = 𝐿₂
#         Sheet["F"*string(ind)] = log10(4/ndiv)
#     end
# end

# index = [4,8,16,32]
# # index = [0.4,0.3,0.2,0.1]
# # index = [0,1,2,3]
# XLSX.openxlsx("./excel/hmd_BiRefine.xlsx", mode="rw") do xf
#     Sheet = xf[3]
#     ind = findfirst(n->n==ndiv,index)+1
#     Sheet["A"*string(ind)] = log10(4/ndiv)
#     # Sheet["A"*string(ind)] = log10(nₚ)
#     Sheet["B"*string(ind)] = 𝐻₁
#     Sheet["C"*string(ind)] = 𝐿₂
# end
nₑ = length(elements)
fig = Figure()
ax1 = Axis3(fig[1,1])
# ax2 = Axis3(fig[1,2])

xs = zeros(nₚ)
ys = zeros(nₚ)
ds = zeros(nₚ)
δds = zeros(nₚ)
us = zeros(nₚ)
# qs = zeros(nₚ)
# as = zeros(nₚ)
es = zeros(nₚ)

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
    δds[i] = node.δd
    es[i] = ds[i] - us[i]
end
face = zeros(nₑ,3)
for (i,elm) in enumerate(elements)
    face[i,:] .= [x.𝐼 for x in elm.𝓒]
end

# # mesh!(ax,xs,ys,zs,face,color=ds)
# # meshscatter!(ax1,xs,ys,us,color=us,markersize = 0.1)
meshscatter!(ax1,xs,ys,ds,color=ds,markersize = 0.06)
# # meshscatter!(ax1,xs,ys,es,color=es,markersize = 0.06)
# meshscatter!(ax2,xs,ys,δds,color=δds,markersize = 0.06)
fig

# save("./fig/hmd_2d/test_x=20/t=98.png",fig)
# save("./fig/72测试/Tri6_非均布_LSG_32.png",fig)
# save("./fig/hmd_2d/锁三边x=20/Tri3/三维图/t=25.png",fig)
# save("./fig/hmd_2d/锁三边x=20/Tri6/均布/t=25.png",fig)
# save("./fig/hmd_2d/局部加密C=0.2/T6_c=0.05.png",fig)
# save("./fig/hmd_2d/Tri3/非均布/n=80.png",fig)


 points = zeros(3,nₚ)
 for (i,node) in enumerate(nodes)
     points[1,i] = node.x
     points[2,i] = node.y
     points[3,i] = node.d*4
     points[3,i] = us[i]*4
 end
 cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE_STRIP,[x.𝐼 for x in elm.𝓒]) for elm in elements]
 vtk_grid("./vtk/hmd_2d/Tri3_d_"*string(ndiv)*".vtu",points,cells) do vtk
     vtk["d"] = [node.d for node in nodes]
      # vtk["精确解"] = us
 end
# fₓ,fₜ,fₓₓ,fₜₜ = truncation_error(elements["Ω"],nₚ)
# println(fₓ)
# println(fₜ)
# println(fₛ)

# xs = [node.x for node in nodes]'
# ys = [node.y for node in nodes]'
# zs = [node.z for node in nodes]'
# points = [xs; ys; zs]
# cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE_STRIP, [xᵢ.𝐼 for xᵢ in elm.𝓒]) for elm in elements["Ω"]]
# vtk_grid("./vtk/hmd_2d/error/uniform_Tri3_"*string(ndiv), points, cells) do vtk
#     # vtk["fₓ"] = fₓ
#     # vtk["fₜ"] = fₜ
#     # vtk["fₓₓ"] = fₓₓ
#     # vtk["fₜₜ"] = fₜₜ
#     # vtk["fₓₓ/fₜₜ"] = fₓₓ./fₜₜ
#     vtk["误差"] = es
# end

