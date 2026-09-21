(* Local Y-state SOC expansion at phi0=0. No uniform helical-twist substitution.
   k is the existing code's momentum; stored bond vectors enter Exp[-I k.d]. *)
ClearAll["Global`*"];
root=DirectoryName[DirectoryName[$InputFileName]];
out=FileNameJoin[{root,"data-space","verification","260917-y-soc-conditions"}];
If[!DirectoryQ[out],CreateDirectory[out,CreateIntermediateDirectories->True]];
assume=sp>0 && jz>j>0 && a>0 && jz/(j+jz)<c<1 && Element[{p,ga},Reals];
s=Sqrt[1-c^2]; h0=3 sp ((j+jz)c-jz);
dirs={{1,0},{-1/2,Sqrt[3]/2},{-1/2,-Sqrt[3]/2}};
phases={0,2 Pi/3,4 Pi/3};
jmat[t_]:={{j+2 p Cos[t],-2 p Sin[t],-ga Sin[t]},
 {-2 p Sin[t],j-2 p Cos[t],ga Cos[t]}, {-ga Sin[t],ga Cos[t],jz}};
ex=FullSimplify[jmat /@ phases];
moment=Table[FullSimplify[Sum[a dirs[[d,mu]] ex[[d]],{d,3}]],{mu,2}];
n={{s,0,c},{-s,0,c},{0,0,-1}};
et={{c,0,-s},{c,0,s},{-1,0,0}};
ey=ConstantArray[{0,1,0},3];
basis=Table[Transpose[{et[[i]],ey[[i]]}],{i,3}];
pairs={{1,2},{2,3},{3,1}};
kh=ConstantArray[0,{6,6}]; kd=ConstantArray[0,{2,6,6}]; kdd=ConstantArray[0,{2,2,6,6}];
Do[kh[[i,i]]=kh[[i+3,i+3]]=h0 sp n[[i,3]],{i,3}];
Do[
 i=pairs[[b,1]];jj=pairs[[b,2]];ii={i,i+3};ij={jj,jj+3};
 block=sp^2 Transpose[basis[[i]]].ex[[d]].basis[[jj]];
 longitudinal=sp^2 n[[i]].ex[[d]].n[[jj]];
 Do[kh[[ii[[u]],ii[[u]]]]-=longitudinal;kh[[ij[[u]],ij[[u]]]]-=longitudinal,{u,2}];
 Do[
 kh[[ii[[u]],ij[[v]]]]+=block[[u,v]];kh[[ij[[v]],ii[[u]]]]+=block[[u,v]];
 Do[
 kd[[mu,ii[[u]],ij[[v]]]]+=-I a dirs[[d,mu]] block[[u,v]];
 kd[[mu,ij[[v]],ii[[u]]]]+=I a dirs[[d,mu]] block[[u,v]];
 Do[kdd[[mu,nu,ii[[u]],ij[[v]]]]+=-a^2 dirs[[d,mu]] dirs[[d,nu]] block[[u,v]];
 kdd[[mu,nu,ij[[v]],ii[[u]]]]+=-a^2 dirs[[d,mu]] dirs[[d,nu]] block[[u,v]],{nu,2}],{mu,2}],{u,2},{v,2}],{b,3},{d,3}];
simp[z_]:=FullSimplify[z,Assumptions->assume];
kh=simp[kh];g={0,0,0,s,-s,0};
proj=Transpose[{UnitVector[6,1],UnitVector[6,2],UnitVector[6,3],(UnitVector[6,4]+UnitVector[6,5])/Sqrt[2],UnitVector[6,6]}];
zero3=ConstantArray[0,{3,3}];
omega=sp ArrayFlatten[{{zero3,-IdentityMatrix[3]},{IdentityMatrix[3],zero3}}];
H=simp[Transpose[proj].kh.proj];
ell=simp[Table[-I kd[[mu]].g,{mu,2}]];
L=simp[Transpose[proj].Transpose[ell]];
bvec=simp[Transpose[proj].omega.g];
chi=simp[bvec.LinearSolve[H,bvec]];
Dmat=simp[Table[g.kdd[[mu,nu]].g/2,{mu,2},{nu,2}]];
relax=simp[Transpose[L].LinearSolve[H,L]];
Cmat=simp[Dmat-relax];
drift=simp[bvec.LinearSolve[H,L]/chi];
expectedEll=a sp^2 {{-3 ga s^2/2,3 ga s^2/2,0,-3 p s,-3 p s,6 p s},
 {-3 p s c,-3 p s c,-6 p s,0,0,0}};
(* Expand the zero-SOC restricted Y canting explicitly as an additional check. *)
eY[cc_,gamma_]:=sp^2(-j(1-cc^2)gamma+jz(cc^2-2 cc))-hh sp(2 cc-1)/3;
cQ=(jz+hh/(3 sp))/(jz+j gamma);
cantingSeries=Normal[Series[cQ/.{gamma->1-a^2 qq^2/4,hh->h0},{qq,0,2}]];
checks=<|
 "bond_sum_SOC_cancels"->(simp[Total[ex]-3 DiagonalMatrix[{j,j,jz}]]===ConstantArray[0,{3,3}]),
 "uniform_Hessian_SOC_independent"->(simp[kh-(kh/.{p->0,ga->0})]===ConstantArray[0,{6,6}]),
 "Goldstone_null"->(simp[kh.g]===ConstantArray[0,6]),
 "moment_x"->(simp[moment[[1]]-a {{3 p,0,0},{0,-3 p,3 ga/2},{0,3 ga/2,0}}]===ConstantArray[0,{3,3}]),
 "moment_y"->(simp[moment[[2]]-a {{0,-3 p,-3 ga/2},{-3 p,0,0},{-3 ga/2,0,0}}]===ConstantArray[0,{3,3}]),
 "phase_gradient_hard_sources"->(simp[ell-expectedEll]===ConstantArray[0,{2,6}]),
 "direct_gradient_tensor"->(simp[Dmat-3 a^2 sp^2 s^2 DiagonalMatrix[{j-p,j+p}]/2]===ConstantArray[0,{2,2}]),
 "uniform_susceptibility"->(simp[chi-2/(3(j+jz))]===0),
 "Gamma_drift"->(simp[drift-{3 a sp ga s/2,0}] === {0,0}),
 "static_tensor_diagonal_phi0"->(simp[Cmat[[1,2]]]===0 && simp[Cmat[[2,1]]]===0),
 "zero_SOC_limit"->(simp[(Cmat/.{p->0,ga->0})-3 a^2 j sp^2 s^2 IdentityMatrix[2]/2]===ConstantArray[0,{2,2}]),
 "Y_canting_stationarity"->(FullSimplify[D[eY[cc,gamma],cc]/.cc->cQ]===0),
 "Y_canting_Q_squared_shift"->(simp[cantingSeries-c-c j a^2 qq^2/(4(j+jz))]===0)
|>;
result=<|"scope"->"Classical harmonic Y at phi0=0; hard subspace stable; not quantum pinning or a full SOC phase scan",
 "wolfram_version"->$Version,"checks"->checks,
 "first_moments_InputForm"->ToString[moment,InputForm],
 "uniform_Hessian_InputForm"->ToString[kh,InputForm],
 "hard_sources_InputForm"->ToString[ell,InputForm],
 "direct_D_InputForm"->ToString[Dmat,InputForm],
 "relaxation_subtraction_InputForm"->ToString[relax,InputForm],
 "static_C_InputForm"->ToString[Cmat,InputForm],
 "drift_InputForm"->ToString[drift,InputForm],
 "chi_cell_InputForm"->ToString[chi,InputForm],
 "source_sha256"->FileHash[$InputFileName,"SHA256","HexString"]|>;
Export[FileNameJoin[{out,"symbolic-soc-check.json"}],result,"RawJSON"];
Print[ExportString[result,"RawJSON"]];If[!And@@Values[checks],Exit[1]];
