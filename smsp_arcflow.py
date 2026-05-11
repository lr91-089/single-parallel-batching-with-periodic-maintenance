# -*- coding: utf-8 -*-
"""
Arc-Flow Reflect model for P|pm|Cmax
(Single Machine Scheduling with Periodic Maintenance, minimize makespan).

Model
-----
  Non-last bins : FRE reflect arc-flow  (Delorme & Iori 2020)
  Last bin      : continuous y[t] variables, one per distinct processing time

  Demand split     : flow_t + y[t] = d[t]   for all t
  Last bin cap     : sum_t  t * y[t] <= T
  Objective        : min  z*(T + t_charge) + sum_t t*y[t]   =  Cmax

Instance format (LOW / MOD sets)
---------------------------------
  Line 1       : n
  Lines 2..n+1 : p_j  (one per line)
  Last line    : T

Usage
-----
  # one instance
  python smsp_arcflow.py --instance path/to/file --out_csv r.csv --out_sol sol/

  # whole folder
  python smsp_arcflow.py --folder path/to/LOW --out_csv r.csv --out_sol sol/
"""

from __future__ import annotations
import argparse, csv, os, time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import re
import os

import gurobipy as gp
from gurobipy import GRB, quicksum


# ─────────────────────────────────────────────
# Instance
# ─────────────────────────────────────────────

@dataclass
class SMInstance:
    n:          int
    jobs:       List[int]        # p_j in original order (0-indexed)
    T:          int
    t_charge:   int = 0
    item_types: Dict[int,int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        c: Dict[int,int] = {}
        for p in self.jobs:
            c[p] = c.get(p, 0) + 1
        
        # Separate full-bin items — not supported by reflect graph
        self.full_bin_items = {t: d for t, d in c.items() if t == self.T}
        self.full_bin_count = sum(self.full_bin_items.values())
        
        # Only regular items go into the arc-flow
        self.item_types = {t: d for t, d in c.items() if t < self.T}

    @classmethod
    def from_file(cls, path: str, t_charge: int = 0) -> "SMInstance":
        with open(path) as f:
            vals = [int(v) for v in f.read().split()]
        n    = vals[0]
        jobs = vals[1:n+1]
        T    = vals[n+1]
        if len(jobs) != n:
            raise ValueError(f"Expected {n} jobs, got {len(jobs)} in {path}")
        return cls(n=n, jobs=jobs, T=T, t_charge=t_charge)

    def summary(self) -> str:
        return (f"n={self.n}, T={self.T}, t_charge={self.t_charge}, "
                f"#types={len(self.item_types)}, total_load={sum(self.jobs)}")


# ─────────────────────────────────────────────
# Reflect graph  (Algorithm 7, Delorme & Iori 2020)
# ─────────────────────────────────────────────

class ReflectGraph:
    def __init__(self, inst: SMInstance):
        self.C      = inst.T
        self.H      = self.C // 2
        self.S_arcs: List[Tuple] = []
        self.R_arcs: List[Tuple] = []
        self.nodes:  set         = set()
        self.Aj:     Dict        = defaultdict(list)
        self._build(inst)

    def _build(self, inst: SMInstance):
        C, H = self.C, self.H
        items = sorted(inst.item_types.items(), key=lambda x: -x[0])
        M = [0]*(H+1); M[0] = 1; self.nodes.add(0)
 
        for wi, di in items:
            if wi > C:
                raise ValueError(f"p={wi} > T={C}: infeasible")
 
            Hp = [0]*(H+1)
            for _ in range(di):
                for l in range(H-1, -1, -1):
                    if Hp[l]==0 and M[l]==1:
                        Hp[l] = 1
                        if l+wi <= H:
                            arc = (l, l+wi, wi)
                            self.S_arcs.append(arc); self.Aj[wi].append(arc)
                            self.nodes.add(l+wi); M[l+wi] = 1
                        elif l <= C-(l+wi):
                            arc = (l, C-(l+wi), wi)
                            self.R_arcs.append(arc); self.Aj[wi].append(arc)
                            self.nodes.add(C-(l+wi))
 
        self.nodes.add(H)
        sv = sorted(self.nodes)
        for u, v in zip(sv, sv[1:]):
            if v <= H:
                self.S_arcs.append((u, v, None))
        self.R_arcs.append((H, H, None))


    def summary(self) -> str:
        si = sum(1 for *_,t in self.S_arcs if t is not None)
        sl = sum(1 for *_,t in self.S_arcs if t is None)
        ri = sum(1 for *_,t in self.R_arcs if t is not None)
        rl = sum(1 for *_,t in self.R_arcs if t is None)
        return (f"C={self.C}, H={self.H}, nodes={len(self.nodes)}, "
                f"S={si}+{sl}loss, R={ri}+{rl}special")


# ─────────────────────────────────────────────
# Bin reconstruction from arc-flow solution
# ─────────────────────────────────────────────

def _reconstruct_bins(xi_s_val, xi_r_val, graph):
    """Return list-of-lists of processing times per non-last bin."""
    xs = defaultdict(int, {k: round(v) for k,v in xi_s_val.items()})
    xr = defaultdict(int, {k: round(v) for k,v in xi_r_val.items()})
    R_paths: Dict[int, List] = defaultdict(list)
    S_paths: Dict[int, List] = defaultdict(list)

    for _ in range(sum(xs.values()) + sum(xr.values()) + 10):
        if not any(v>0 for v in xs.values()) and not any(v>0 for v in xr.values()):
            break
        path=[]; node=0; is_R=False; collision=None
        while True:
            ro = [(e,t) for (d,e,t) in graph.R_arcs if d==node and xr[d,e,t]>0]
            so = [(e,t) for (d,e,t) in graph.S_arcs if d==node and xs[d,e,t]>0]
            if ro:
                e,t = ro[0]; path.append(t); xr[node,e,t]-=1
                collision=e; is_R=True; break
            elif so:
                e,t = so[0]; path.append(t); xs[node,e,t]-=1; node=e
            else:
                collision=node; is_R=False; break
        if not path: break
        items = [t for t in path if t is not None]
        (R_paths if is_R else S_paths)[collision].append(items)

    bins = []
    for v in list(R_paths):
        while R_paths[v] and S_paths[v]:
            bins.append(R_paths[v].pop(0) + S_paths[v].pop(0))
    return bins


def _assign_indices(inst, bins_ptimes, last_ptimes):
    """Map processing-time lists back to original 0-based job indices."""
    pool: Dict[int,List[int]] = defaultdict(list)
    for idx, p in enumerate(inst.jobs):
        pool[p].append(idx)

    def pick(p): return pool[p].pop(0)

    bins_idx = [sorted(pick(p) for p in pt) for pt in bins_ptimes]
    last_idx = sorted(pick(p) for p in last_ptimes)
    return bins_idx, last_idx


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────

@dataclass
class SolverResult:
    status:       str
    cmax:         Optional[float]
    z_nonlast:    Optional[int]
    last_load:    Optional[float]
    lb:           Optional[float]
    gap:          Optional[float]
    runtime:      float
    root_gap : float
    instance:     SMInstance
    bins_indices: Optional[List[List[int]]] = None
    last_indices: Optional[List[int]]       = None
    bins_ptimes:  Optional[List[List[int]]] = None
    last_ptimes:  Optional[List[int]]       = None
    machine_assignments: Optional[Dict[int, Dict]] = None


# ─────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────

def get_root_gap(logfile, obj_val=None):
    root_bound     = None
    root_incumbent = None
    root_cutoff    = False

    with open(logfile, "r") as f:
        for line in f:
            # 1. Heuristic solution before root
            if "Found heuristic solution: objective" in line:
                match = re.search(r"objective\s+([\-0-9.eE+]+)", line)
                if match and root_incumbent is None:
                    root_incumbent = float(match.group(1))

            # 2. Root relaxation bound or cutoff
            if "Root relaxation:" in line:
                if "cutoff" in line or "infeasible" in line:
                    root_cutoff = True
                else:
                    match = re.search(r"objective\s+([\-0-9.eE+]+)", line)
                    if match and root_bound is None:
                        root_bound = float(match.group(1))

            # 3. H or * lines at node 0
            if re.match(r"\s*[H\*]\s+0\s+", line):
                match = re.search(r"\s+(\d+\.\d+)\s+\d+\.\d+\s+\d+\.\d+%", line)
                if match and root_incumbent is None:
                    root_incumbent = float(match.group(1))

    # Fallback
    if root_incumbent is None:
        root_incumbent = 2*root_bound

    # Cutoff: root bound = incumbent → gap = 0
    if root_cutoff and root_incumbent is not None:
        return 0.0, root_incumbent, root_incumbent

    if root_bound is not None and root_incumbent is not None and root_incumbent != 0:
        gap = abs(root_incumbent - root_bound) / abs(root_incumbent)
        return gap, root_bound, root_incumbent

    return None, root_bound, root_incumbent

class ArcFlowReflectSMSP:
    def __init__(self, inst, time_limit=3600.0, verbose=False, threads=1, machines=1):
        self.inst       = inst
        self.time_limit = time_limit
        self.verbose    = verbose
        self.threads    = threads
        self.graph      = ReflectGraph(inst)
        self._model = None
        self._xi_s = {}; self._xi_r = {}; self._y = {}
        self.machines = machines
        self.model_name = "arcflow_reflect"

    def build_model(self):
        inst=self.inst; graph=self.graph; T=inst.T; tc=inst.t_charge
        m = gp.Model("SMSP_ArcFlowReflect")
        m.Params.TimeLimit = self.time_limit
        m.Params.Threads   = self.threads
        m.Params.MIPGap = 1e-6
        m.setParam("OutputFlag", 1 if self.verbose else 0)
        M = range(self.machines)
 
        xi_s = {(d,e,t): m.addVar(vtype=GRB.INTEGER, lb=0, name=f"xs_{d}_{e}_{t}")
                for (d,e,t) in graph.S_arcs}
        xi_r = {(d,e,t): m.addVar(vtype=GRB.INTEGER, lb=0, name=f"xr_{d}_{e}_{t}")
                for (d,e,t) in graph.R_arcs}
        if self.machines>1:
            y    = {(t,k): m.addVar(vtype=GRB.INTEGER, lb=0, ub=d, name=f"y_{t}_{k}")
                    for t,d in inst.item_types.items() for k in M}
            u = {(t,k): m.addVar(vtype=GRB.INTEGER, lb=0, ub=d, name=f"u_{t}_{k}")
                    for t,d in inst.item_types.items() for k in M}
            for k in M:
                u[-1,k] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"u_{-1}_{k}")
            ub = {k: m.addVar(vtype=GRB.INTEGER, lb=0, ub=inst.full_bin_count, name=f"ub_{k}")
                    for k in M}
            Cmax = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Cmax")
        else:
            y    = {t: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=d, name=f"y_{t}")
                    for t,d in inst.item_types.items()}
        m.update()
 
        # (17) flow conservation
        for e in graph.nodes:
            if e==0: continue
            in_s  = quicksum(xi_s[d,i,t] for (d,i,t) in graph.S_arcs if i==e)
            in_r  = quicksum(xi_r[d,i,t] for (d,i,t) in graph.R_arcs if i==e)
            out_s = quicksum(xi_s[i,f,t] for (i,f,t) in graph.S_arcs if i==e)
            out_r = quicksum(xi_r[i,f,t] for (i,f,t) in graph.R_arcs if i==e)
            lhs=in_s; rhs=in_r+out_s+out_r
            if lhs.size()>0 or rhs.size()>0:
                m.addConstr(lhs==rhs, name=f"flow_{e}")
 
        # (18) source outflow = 2*z
        z_expr = quicksum(xi_r[d,e,t] for (d,e,t) in graph.R_arcs)
        out_0  = (quicksum(xi_s[0,e,t] for (d,e,t) in graph.S_arcs if d==0) +
                  quicksum(xi_r[0,e,t] for (d,e,t) in graph.R_arcs if d==0))
        m.addConstr(out_0 == 2*z_expr, name="source_outflow")
 
        # (D) demand split — one constraint per item type
        s_set = set(graph.S_arcs)
        for t, arcs in graph.Aj.items():
            flow_t = quicksum(
                xi_s[d,e,tt] if (d,e,tt) in s_set else xi_r[d,e,tt]
                for (d,e,tt) in arcs)
            if self.machines > 1:
                m.addConstr(flow_t + quicksum(y[t,k] for k in M) == inst.item_types[t],
                            name=f"dem_{t}")
            else:
                m.addConstr(flow_t + y[t] == inst.item_types[t], name=f"dem_{t}")
        
        if self.machines > 1:
            # --- u linkage: sum over machines = reflect arc flow per type ---
            # Item-type arcs (t is not None)
            for t in inst.item_types:
                z_exprK = quicksum(xi_r[d, e, t1]
                                   for (d, e, t1) in graph.R_arcs if t1 == t)
                m.addConstr(quicksum(u[t, k] for k in M) == z_exprK,
                            name=f"u_link_{t}")
        
            # Loss arc (t1 is None) — represented by u[-1, k]
            z_loss = quicksum(xi_r[d, e, t1]
                              for (d, e, t1) in graph.R_arcs if t1 is None)
            m.addConstr(quicksum(u[-1, k] for k in M) == z_loss,
                        name="u_link_loss")
        
            # --- per-machine last-bin and Cmax constraints ---
            m.addConstr(quicksum(ub[k] for k in M) == self.inst.full_bin_count,
                        name="ub_split")
            for k in M:
                m.addConstr(quicksum(t * y[t, k] for t in inst.item_types) <= T,
                            name=f"last_cap_{k}")
                # total bins on machine k = u_k (non-last, item arcs)
                #                         + u[-1,k] (non-last, loss arcs)
                #                         + ub[k]   (full bins)
                total_bins_k = (quicksum(u[t, k] for t in inst.item_types)
                                + u[-1, k] + ub[k])
                m.addConstr(
                    total_bins_k * (T + tc)
                    + quicksum(t * y[t, k] for t in inst.item_types) <= Cmax,
                    name=f"cmax_{k}")
                if k>0:
                    for t in inst.item_types:
                        m.addConstr(
                            y[t, k] <= quicksum(y[t1, k-1] for t1 in inst.item_types if t1<t),
                            name=f"cmax_{k}")
                
        else:
            m.addConstr(quicksum(t * y[t] for t in inst.item_types) <= T,
                        name="last_cap")
 

        
        import math
        total_load = sum(inst.jobs)
        full_bin_load = inst.full_bin_count * inst.T
        partial_load  = total_load - full_bin_load
        
        if self.machines>1:
            z_lb = max(0, math.ceil(partial_load / (T*len(M))) - 1)
            m.addConstr(z_expr >= z_lb, name="z_lb")
            total_load = sum(inst.jobs)
            cmax_lb = math.ceil(total_load / self.machines)
            m.addConstr(Cmax >= cmax_lb, name="cmax_lb_load")
        else:
            z_lb = max(0, math.ceil(partial_load / T) - 1)
            m.addConstr(z_expr >= z_lb, name="z_lb")
 
        # objective: exact Cmax = z*(T+tc) + sum_t t*y[t]
        # Objective — add full_bin_count as constant offset to z
        if self.machines>1:
            m.setObjective(
                Cmax,
                GRB.MINIMIZE
            )
        else:
            m.setObjective(
                (z_expr + inst.full_bin_count) * (T + tc) + quicksum(t*y[t] for t in inst.item_types),
                GRB.MINIMIZE
            )
        m._root_obj = None
        m.Params.LogFile = ""
        m.Params.LogFile = "gurobi_run2.log"
        m.Params.LogToConsole = 1  # keep console output too
        #m.Params.NodeLimit = 1
        self._model=m; self._xi_s=xi_s; self._xi_r=xi_r; self._y=y
        if self.machines>1:
            self._u = u
            self._ub = ub  # add alongside self._u = u
        else:
            self._u = None
            self._ub = None
        return m


    def solve(self) -> SolverResult:
        if self._model is None: self.build_model()
        m=self._model; t0=time.time(); 
        m.optimize()
        m.Params.LogFile = ""   # release Gurobi's handle
        root_gap, root_bound, root_incumbent = get_root_gap("gurobi_run2.log")
        rt=time.time()-t0; st=m.Status
        os.remove("gurobi_run2.log")  # delete after reading
        if m.SolCount==0:
            if m.status==3:
                m.computeIIS()
                m.write("infeasible_model.ilp")
            return SolverResult(
                status="infeasible" if st==GRB.INFEASIBLE else "timeout",
                cmax=None, z_nonlast=None, last_load=None,
                lb=None, gap=None, runtime=rt, instance=self.inst, root_gap=None)
        
        if self.machines>1:
            z_val = round(sum(v.X for v in self._xi_r.values())) + self.inst.full_bin_count
            last_load = sum(t * self._y[t, k].X
                    for t in self.inst.item_types
                    for k in range(self.machines)
                    if t < self.inst.T)
        else:
            z_val = round(sum(v.X for v in self._xi_r.values())) + self.inst.full_bin_count
            last_load = sum(t*self._y[t].X for t in self.inst.item_types if t < self.inst.T)
        cmax      = m.ObjVal
        status    = "optimal" if st==GRB.OPTIMAL else "feasible"
        

        # reconstruct
        xs_val      = {k:v.X for k,v in self._xi_s.items() if v.X>0.5}
        xr_val      = {k:v.X for k,v in self._xi_r.items() if v.X>0.5}
        bins_ptimes = _reconstruct_bins(xs_val, xr_val, self.graph)
        for t, count in self.inst.full_bin_items.items():
            for _ in range(count):
                bins_ptimes.append([t])

        last_ptimes: List[int] = []
        if self.machines > 1:
            for (t, k), var in self._y.items():
                last_ptimes.extend([t] * round(var.X))
        else:
            for t, var in self._y.items():
                last_ptimes.extend([t] * round(var.X))
        last_ptimes.sort()
        try:
            for v in m.getVars():
                if v.X > 1e-5:
                    print(f"{v.varName} = {v.X:.4g}")
        except Exception:
            print("Could not retrieve variable values.")

        # Replace the _assign_indices call and the entire machine_assignments block with this:

        # Single unified index pool
        full_pool: Dict[int, List[int]] = defaultdict(list)
        for idx, p in enumerate(self.inst.jobs):
            full_pool[p].append(idx)
        
        # Assign non-last bin indices first (order matches bins_ptimes)
        # Assign non-last bin indices first (order matches bins_ptimes)
        bins_idx = [sorted(full_pool[p].pop(0) for p in pt) for pt in bins_ptimes]
        
        # Sort both together by the first (smallest) job index in each bin
        if bins_idx:
            bins_idx, bins_ptimes = zip(*sorted(zip(bins_idx, bins_ptimes), key=lambda x: x[0][0]))
            bins_idx = list(bins_idx)
            bins_ptimes = list(bins_ptimes)
        if self.machines > 1:
            # Per-machine last bin: read directly from y[t,k]
            machine_assignments = {}
            for k in range(self.machines):
                n_bins_k = (sum(round(self._u[t, k].X) for t in self.inst.item_types)
                    + round(self._u[-1, k].X)
                    + round(self._ub[k].X))
                last_k_ptimes = []
                for t in sorted(self.inst.item_types):
                    last_k_ptimes.extend([t] * round(self._y[t, k].X))
                last_k_ptimes.sort()
                last_k_indices = sorted(full_pool[p].pop(0) for p in last_k_ptimes)
                machine_assignments[k] = {
                    "bins_ptimes":  [None] * n_bins_k,  # placeholder, count only
                    "bins_indices": [],
                    "last_ptimes":  last_k_ptimes,
                    "last_indices": last_k_indices,
                }
            # Aggregate last for the top-level result
            last_ptimes = sorted(p for k in range(self.machines)
                                 for p in machine_assignments[k]["last_ptimes"])
            last_idx = sorted(i for k in range(self.machines)
                              for i in machine_assignments[k]["last_indices"])
        else:
            machine_assignments = None
            last_ptimes = []
            for t, var in self._y.items():
                last_ptimes.extend([t] * round(var.X))
            last_ptimes.sort()
            last_idx = sorted(full_pool[p].pop(0) for p in last_ptimes)
                
        

        return SolverResult(
            status=status, cmax=cmax, z_nonlast=int(z_val),
            last_load=last_load, lb=m.ObjBound, gap=m.MIPGap,
            runtime=rt, root_gap=root_gap, instance=self.inst,
            bins_indices=bins_idx, last_indices=last_idx,
            bins_ptimes=bins_ptimes, last_ptimes=last_ptimes,
            machine_assignments=machine_assignments)  


# ─────────────────────────────────────────────
# Solution file
# ─────────────────────────────────────────────

def write_solution_file(res: SolverResult, path: str):
    with open(path, "w") as f:
        if res.cmax is None:
            f.write(f"Status: {res.status}\nNo feasible solution found.\n")
            return

        f.write(f"The num of Batches: {res.z_nonlast}\n")
        f.write(f"makespan: {res.cmax}\n")

        # Always write all batches first (aggregated)
        f.write("\nThe jobs in each Batch\n")
        for b, (idx, pt) in enumerate(zip(res.bins_indices, res.bins_ptimes)):
            f.write(f"Batch{b}:index:{idx} Processing Time: {sum(pt)}\n")
        if res.machine_assignments is None:
            f.write("The last Batch\n")
            f.write(f"index:{res.last_indices} "
                    f"Processing Time: {sum(res.last_ptimes or [])}\n")

        # Per-machine breakdown (only if multi-machine)
        if res.machine_assignments is not None:
            f.write("\n--- Per-Machine Assignment ---\n")
            for k, asgn in res.machine_assignments.items():
                n_bins = len(asgn["bins_ptimes"])
                f.write(f"\nMachine {k}  ({n_bins} non-last bin(s)):\n")
                f.write(f"  Last bin: index:{asgn['last_indices']}  "
                        f"Processing Time: {sum(asgn['last_ptimes'])}\n")


# ─────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────

CSV_FIELDS = ["instance","n","T","status","cmax","lb","gap_pct",
              "z_nonlast","last_load","runtime_s","runtime_phase1",
              "root_gap","root_gap_phase2","model","numOfThreads",
              "set","comment","machines"]

def append_csv(path: str, res: SolverResult, name: str,
               model: str = "", num_threads: int = 1,
               set_name: str = "", machines: int = 1):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, delimiter=";")
        if new: w.writeheader()
        w.writerow({
            "instance":        name,
            "n":               res.instance.n,
            "T":               res.instance.T,
            "status":          res.status,
            "cmax":            f"{res.cmax:.1f}"      if res.cmax      is not None else "",
            "lb":              f"{res.lb:.2f}"        if res.lb        is not None else "",
            "gap_pct":         f"{res.gap*100:.4f}"   if res.gap       is not None else "",
            "z_nonlast":       res.z_nonlast                            if res.z_nonlast is not None else "",
            "last_load":       f"{res.last_load:.1f}" if res.last_load is not None else "",
            "runtime_s":       f"{res.runtime:.3f}",
            "runtime_phase1":  "",
            "root_gap":        f"{res.root_gap:.8f}"  if res.root_gap  is not None else "",
            "root_gap_phase2": "",
            "model":           model,
            "numOfThreads":    num_threads,
            "set":             set_name,
            "comment":         "",
            "machines":        machines,
        })


# ─────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────

def run_folder(folder, csv_path, sol_dir=None, t_charge=0,
               time_limit=3600.0, verbose=False, threads=1, machines=1):
    if sol_dir: os.makedirs(sol_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(folder)
                   if os.path.isfile(os.path.join(folder, f)))
    if not files:
        print(f"No files found in {folder}"); return

    print(f"Instances : {len(files)}  |  CSV: {csv_path}"
          + (f"  |  Sol: {sol_dir}" if sol_dir else ""))
    print("-"*72)
    set_name = os.path.basename(os.path.normpath(folder))

    for i, fname in enumerate(files, 1):
        fpath = os.path.join(folder, fname)
        print(f"[{i:4d}/{len(files)}] {fname:<32s}", end=" ", flush=True)
        try:
            inst   = SMInstance.from_file(fpath, t_charge=t_charge)
            solver = ArcFlowReflectSMSP(inst, time_limit=time_limit,
                                         verbose=verbose, threads=threads, machines=machines)
            solver.build_model()
            res = solver.solve()
            if res.cmax is not None:
                gap_s = f"{res.gap*100:.2f}%" if res.gap is not None else "?"
                print(f"{res.status:8s}  Cmax={res.cmax:>9.1f}  "
                      f"z={res.z_nonlast:>3d}  gap={gap_s:>8s}  {res.runtime:.2f}s")
            else:
                print(f"{res.status}  {res.runtime:.2f}s")
        except Exception as exc:
            print(f"ERROR: {exc}")
            dummy = SMInstance(n=0, jobs=[], T=0, t_charge=t_charge)
            res = SolverResult(status="error", cmax=None, z_nonlast=None,
                               last_load=None, lb=None, gap=None,
                               runtime=0.0,root_gap=None, instance=dummy)

        append_csv(csv_path, res, fname,
               model=solver.model_name, num_threads=threads,
               set_name=set_name, machines=solver.machines)
        if sol_dir and res.status != "error":
            write_solution_file(res, os.path.join(sol_dir, fname+".sol"))

    print("-"*72)
    print(f"Done. Results -> {csv_path}")


def run_single(fpath, csv_path=None, sol_dir=None, t_charge=0,
               time_limit=3600.0, verbose=True, threads=1, machines=1):
    fname = os.path.basename(fpath)
    inst  = SMInstance.from_file(fpath, t_charge=t_charge)
    print(f"Instance : {fname}\n  {inst.summary()}")
    print(f"Graph    : {ReflectGraph(inst).summary()}")

    solver = ArcFlowReflectSMSP(inst, time_limit=time_limit,
                                  verbose=verbose, threads=threads,machines=machines)
    solver.build_model()
    res = solver.solve()
    
    
    set_name = os.path.basename(os.path.dirname(os.path.abspath(fpath)))
    print(f"\nStatus   : {res.status}")
    if res.cmax is not None:
        print(f"Cmax     : {res.cmax}")
        if res.lb  is not None: print(f"LB       : {res.lb:.2f}")
        if res.gap is not None: print(f"Gap      : {res.gap*100:.4f}%")
        print(f"z (non-last bins) : {res.z_nonlast}")
        print(f"Last bin load     : {res.last_load:.1f}")
    print(f"Runtime  : {res.runtime:.2f}s")

    if res.bins_indices is not None:
        print(f"\nThe num of Batches: {res.z_nonlast}")
        print(f"makespan: {res.cmax}")
        print("The jobs in each Batch ")
        for b,(idx,pt) in enumerate(zip(res.bins_indices, res.bins_ptimes)):
            print(f"Batch{b}:index:{idx} Processing Time: {sum(pt)}")
        print("The last Batch")
        print(f"index:{res.last_indices} "
              f"Processing Time: {sum(res.last_ptimes or [])}")

    if csv_path:
        append_csv(csv_path, res, fname,
                   model=solver.model_name, num_threads=threads,
                   set_name=set_name, machines=solver.machines)
        print(f"\nCSV -> {csv_path}")
    if sol_dir:
        os.makedirs(sol_dir, exist_ok=True)
        sp = os.path.join(sol_dir, fname+".sol")
        write_solution_file(res, sp)
        print(f"Sol -> {sp}")
    return res



"""
folder = "LOW"
run_folder(
    folder     = f"Benchmark Instances/Instances/{folder}/",
    csv_path   = f"results/{folder}_results_2M_symm.csv",
    sol_dir    = f"results/{folder}_solutions_2M_symm",
    t_charge   = 0,
    time_limit = 3600.0,
    threads    = 1,
    verbose    = True,
    machines = 2
)#"""

#"""
setstr = "LOW"
file = "L_00000069"
folder     = f"Benchmark Instances/Instances/{setstr}/"
csv_path   = f"results/{setstr}_results_parallel.csv"
sol_dir    = f"results/{setstr}_solutions_parallel"
result = run_single(f"Benchmark Instances/Instances/{setstr}/"+file, t_charge=0, time_limit=300,machines=1)
print(result.cmax, result.z_nonlast, result.last_load, result.root_gap)#"""

"""

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Arc-Flow Reflect solver for 1|pm|Cmax")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--instance", help="Single instance file")
    grp.add_argument("--folder",   help="Folder of instances (batch)")
    p.add_argument("--out_csv",    default="results.csv")
    p.add_argument("--out_sol",    default=None, help="Dir for .sol files")
    p.add_argument("--t_charge",   type=int,   default=0)
    p.add_argument("--time_limit", type=float, default=3600.0)
    p.add_argument("--threads",    type=int,   default=1)
    p.add_argument("--verbose",    action="store_true")
    args = p.parse_args()

    kw = dict(t_charge=args.t_charge, time_limit=args.time_limit,
              verbose=args.verbose, threads=args.threads)
    if args.instance:
        run_single(args.instance, csv_path=args.out_csv,
                   sol_dir=args.out_sol, **kw)
    else:
        run_folder(args.folder, csv_path=args.out_csv,
                   sol_dir=args.out_sol, **kw)"""