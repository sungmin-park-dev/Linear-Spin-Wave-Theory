(* Classical Y-state stiffness at zero bond-dependent SOC.
   The wavevector uses physical site positions, not magnetic-cell coordinates.
   Run from any working directory with wolframscript -file <this file>. *)
ClearAll["Global`*"];
root = DirectoryName[DirectoryName[$InputFileName]];
out = FileNameJoin[{root, "data-space", "verification", "260917-y-stiffness"}];
If[!DirectoryQ[out], CreateDirectory[out, CreateIntermediateDirectories -> True]];
assumptions = sp > 0 && jz > j > 0 && a > 0 && 0 < c < 1;
ss = Sqrt[1-c^2];
deltas = a {{1, 0}, {-1/2, Sqrt[3]/2}, {-1/2, -Sqrt[3]/2}};
q = {qx, qy};
rot[z_] := {{Cos[z], -Sin[z], 0}, {Sin[z], Cos[z], 0}, {0, 0, 1}};
n0 = {{ss, 0, c}, {-ss, 0, c}, {0, 0, -1}};
et = {{c, 0, -ss}, {c, 0, ss}, {-1, 0, 0}};
x = {x1, x2, x3}; y = {y1, y2, y3}; vars = Join[x, y];
(* Exact through second order in regular tangent coordinates, including C. *)
n = Table[n0[[i]] (1-(x[[i]]^2+y[[i]]^2)/2) + et[[i]] x[[i]] + {0,1,0} y[[i]], {i,3}];
pairs = {{1,2}, {2,3}, {3,1}};
ex = DiagonalMatrix[{j,j,jz}];
energy = sp^2 Sum[n[[pairs[[b,1]]]].ex.rot[q.deltas[[d]]].n[[pairs[[b,2]]]], {b,3},{d,3}] - h sp Total[n[[All,3]]];
zero = Thread[Join[vars,q] -> 0];
hRule = h -> 3 sp ((j+jz)c-jz);
simp[z_] := FullSimplify[z /. hRule, Assumptions -> assumptions];
stationarity = simp[Table[D[energy,u] /. zero, {u,vars}]];
mixed = simp[Table[D[energy,u,v] /. zero, {u,vars},{v,q}]];
hard = simp[Table[D[energy,u,v] /. zero, {u,vars},{v,vars}]];
qHessian = simp[Table[D[energy,u,v] /. zero, {u,q},{v,q}]];
area = Sqrt[3] a^2/2;
rho = simp[qHessian/(3 area)];
w = -sp {ss,-ss,0};
chi = simp[w.LinearSolve[hard[[1;;3,1;;3]],w]/3];
goldstone = Join[{0,0,0},{ss,-ss,0}];
velocitySquared = simp[area rho[[1,1]]/chi];
checks = <|
 "bond_first_moment_zero" -> (Total[deltas] === {0,0}),
 "bond_second_moment" -> simp[Total[Outer[Times,#,#]& /@ deltas] - 3 a^2 IdentityMatrix[2]/2] === ConstantArray[0,{2,2}],
 "stationary_Y" -> (stationarity === ConstantArray[0,6]),
 "mixed_twist_internal_Hessian_zero" -> (mixed === ConstantArray[0,{6,2}]),
 "uniform_Goldstone_null" -> (simp[hard.goldstone] === ConstantArray[0,6]),
 "rho_formula" -> (simp[rho - j sp^2 (1-c^2) IdentityMatrix[2]/Sqrt[3]] === ConstantArray[0,{2,2}]),
 "chi_formula" -> (simp[chi - 2/(9 (j+jz))] === 0),
 "dispersion_slope_formula" -> (simp[velocitySquared - 9 a^2 j sp^2 (j+jz)(1-c^2)/4] === 0)
|>;
result = <|"scope" -> "Classical zero-SOC Y branch; positive hard Hessian required; no thermal or quantum-renormalized stiffness", "wolfram_version" -> $Version,
 "checks" -> checks, "rho_tensor_InputForm" -> ToString[rho,InputForm],
 "chi_InputForm" -> ToString[chi,InputForm], "energy_slope_squared_InputForm" -> ToString[velocitySquared,InputForm],
 "hard_Hessian_per_cell_InputForm" -> ToString[hard,InputForm],
 "source_sha256" -> FileHash[$InputFileName,"SHA256","HexString"]|>;
Export[FileNameJoin[{out,"symbolic-check.json"}],result,"RawJSON"];
Print[ExportString[result,"RawJSON"]];
If[!And@@Values[checks], Exit[1]];
