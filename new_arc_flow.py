# -*- coding: utf-8 -*-
"""
Arc-flow model for P_m|pm|C_max, inspired and extending Mrad & Souayah (2018).

Integer-flow formulation:
  - x[i,j,p] : INTEGER, ub = d_p  (flow per item TYPE, for conservation + demand)
  - y[i,j,p] : BINARY  indicator = 1 iff x[i,j,p] >= 1  (for Cmax constraint only)

Why the split is necessary:
  Mrad's constraint z >= j * x_ij is correct only when x ∈ {0,1}.
  With integer x[i,j,p]=2 (two machines using same arc), z >= 2j is wrong.
  Binary y[i,j,p] = 1{x >= 1} fixes this: z >= j * y[i,j,p].
  Flow conservation and demand use integer x (efficient).
  Cmax uses binary y (correct).

Variable count vs pure-binary:
  Pure binary  : one var per (arc, job)  →  sum_p d_p * |arcs_p|
  Integer+bin  : two vars per (arc, type) →  2 * |unique arcs|
  Saving       : large when many jobs share the same processing time.
"""
from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB, quicksum

from smsp_arcflow import ReflectGraph, SMInstance


# ─────────────────────────────────────────────
# Instance
# ─────────────────────────────────────────────

@dataclass
class BPPMInstance:
    n:    int
    jobs: List[int]
    T:    int
    m:    int = 1
    item_types:       Dict[int, int] = field(default_factory=dict)
    full_batch_count: int            = 0

    def __post_init__(self):
        c: Dict[int, int] = {}
        for p in self.jobs:
            c[p] = c.get(p, 0) + 1
        self.full_batch_count = sum(d for p, d in c.items() if p == self.T)
        self.item_types = {p: d for p, d in c.items()}

    @classmethod
    def from_file(cls, path: str, machines: int = 1) -> "BPPMInstance":
        with open(path) as f:
            vals = [int(v) for v in f.read().split()]
        n = vals[0]; jobs = vals[1:n+1]; T = vals[n+1]
        return cls(n=n, jobs=jobs, T=T, m=machines)

    def summary(self) -> str:
        return (f"n={self.n}, T={self.T}, m={self.m}, "
                f"#types={len(self.item_types)}, "
                f"load={sum(self.jobs)}, "
                f"cmax_lb={math.ceil(sum(self.jobs)/self.m)}")


# ─────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────

class BatchGraph:
    """
    Extension of Mrad & Souayah (2018) to multiple batch slots,
    with arcs indexed by item TYPE instead of individual jobs.

    A'  (item arcs):  (i, j, p)  where j-i=p, both in [b*T, (b+1)*T]
    A'' (trans arcs): (i, (b+1)*T) — flow conservation only, no Cmax contrib

    Bug fix vs Mrad original: BFS inner loop per type p so that chains like
    0→3→6 with p=3 are correctly generated (node 3, created by the first
    p=3 arc, can host a second p=3 arc within the same pass).

    Note on batch-crossing arcs: merging the transition hop and the first
    item arc into a single cross arc (i → (b+1)T+p) adds one arc per
    (node, type) pair per batch boundary, multiplying trans_arcs by |types|.
    In practice this exceeds the variable savings from the symmetry filter
    p ≤ p_creating[i], so we keep the simpler two-hop structure.
    """

    def __init__(self, inst: BPPMInstance, z_max: int):
        self.inst  = inst
        self.T     = inst.T
        self.z_max = z_max
        self.UB    = z_max * inst.T

        self.item_arcs:  List[Tuple[int, int, int]] = []   # A'  (i, j, p)
        self.cross_arcs: List[Tuple[int, int, int]] = []   # empty — kept for API compat
        self.trans_arcs: List[Tuple[int, int]]       = []   # A'' (i, (b+1)*T)
        self.nodes:      Set[int]                    = set()
        self.At:         Dict[int, List]             = defaultdict(list)

        self._build()

    def _build(self):
        inst = self.inst; T = self.T; z_max = self.z_max

        types_sorted = sorted(inst.item_types.keys(), reverse=True)
        seen:      Set[Tuple] = set()
        reachable: Set[int]   = set()

        # ── Within-batch item arcs ────────────────────────────────────────
        for b in range(z_max):
            start = b * T
            reachable.add(start)
            batch_heads: Set[int] = {start}

            for p in types_sorted:
                # BFS so that a node created by arc (i→j, p) can immediately
                # host another arc of the same type p within the same pass.
                frontier: Set[int] = set(batch_heads)
                while frontier:
                    next_frontier: Set[int] = set()
                    for i in sorted(frontier):
                        j = i + p
                        if j <= start + T:
                            arc = (i, j, p)
                            if arc not in seen:
                                seen.add(arc)
                                self.item_arcs.append(arc)
                                self.At[p].append(arc)
                                reachable.add(j)
                                if j not in batch_heads:
                                    batch_heads.add(j)
                                    next_frontier.add(j)
                    frontier = next_frontier
                    

        # ── Transition arcs A'': each reachable node → next batch-start ──
        # One arc per reachable node (not multiplied by |types|).
        for i in sorted(reachable):
            b = i // T
            j = (b + 1) * T
            if j <= self.UB:
                self.trans_arcs.append((i, j))

        self.nodes = reachable | {b * T for b in range(z_max + 1)}

    def summary(self) -> str:
        n_binary_equiv = sum(self.inst.item_types[p]
                             for (i,j,p) in self.item_arcs)
        return (f"UB={self.UB}, T={self.T}, z_max={self.z_max}, "
                f"nodes={len(self.nodes)}, "
                f"item_arcs={len(self.item_arcs)}, "
                f"trans_arcs={len(self.trans_arcs)} "
                f"(binary-equiv={n_binary_equiv})")


# ─────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────

@dataclass
class SolveResult:
    status:  str
    cmax:    Optional[float]
    lb:      Optional[float]
    gap:     Optional[float]
    runtime: float
    graph:   object   # BatchGraph, kept for solution writing
    xi:      dict
    yi:      dict
    xt:      dict


def solve(inst: BPPMInstance, z_max: int,
          time_limit: float = 3600.0,
          verbose:    bool  = True,
          threads:    int   = 1) -> SolveResult:

    graph = BatchGraph(inst, z_max=z_max)
    T, m, UB = inst.T, inst.m, graph.UB
    print(f"Graph: {graph.summary()}")

    model = gp.Model("PBPM_BatchGraph_Int")
    model.Params.TimeLimit  = time_limit
    model.Params.Threads    = threads
    model.Params.MIPGap     = 1e-6
    model.setParam("OutputFlag", 1 if verbose else 0)

    # ── Variables ─────────────────────────────────────────────────────
    # One INTEGER var per (arc, type) for flow; one BINARY var for Cmax
    xi = {(i,j,p): model.addVar(vtype=GRB.INTEGER, lb=0,
                                 ub=inst.item_types[p], name=f"x_{i}_{j}_{p}")
          for (i,j,p) in graph.item_arcs}
    yi = {(i,j,p): model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{p}")
          for (i,j,p) in graph.item_arcs}
    xt = {(i,j): model.addVar(vtype=GRB.INTEGER, lb=0, ub=m, name=f"xt_{i}_{j}")
          for (i,j) in graph.trans_arcs}
    z  = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="z")
    model.update()

    # ── Objective ─────────────────────────────────────────────────────
    model.setObjective(z, GRB.MINIMIZE)

    # ── Link x and y ──────────────────────────────────────────────────
    for (i,j,p) in graph.item_arcs:
        model.addConstr(xi[i,j,p] >= yi[i,j,p])
        model.addConstr(xi[i,j,p] <= inst.item_types[p] * yi[i,j,p])

    # ── Makespan: z >= j * y  (binary y keeps this correct for any x) ─
    lb_val = math.ceil(sum(inst.jobs) / m)
    model.addConstr(z >= lb_val)
    model.addConstr(z <= UB)
    for (i,j,p) in graph.item_arcs:
        model.addConstr(z >= j * yi[i,j,p])

    # ── Source outflow = m ─────────────────────────────────────────────
    out_0 = (quicksum(xi[0,j,p] for (i,j,p) in graph.item_arcs  if i == 0) +
             quicksum(xt[0,j]   for (i,j)   in graph.trans_arcs if i == 0))
    model.addConstr(out_0 == m)

    # ── Flow conservation ──────────────────────────────────────────────
    for v in graph.nodes:
        if v == 0 or v == UB:
            continue
        in_f  = (quicksum(xi[i,v,p] for (i,j,p) in graph.item_arcs  if j == v) +
                 quicksum(xt[i,v]   for (i,j)   in graph.trans_arcs if j == v))
        out_f = (quicksum(xi[v,j,p] for (i,j,p) in graph.item_arcs  if i == v) +
                 quicksum(xt[v,j]   for (i,j)   in graph.trans_arcs if i == v))
        if in_f.size() + out_f.size() > 0:
            model.addConstr(in_f == out_f)

    # ── Demand: all units of each type scheduled exactly once ──────────
    for p, d in inst.item_types.items():
        arcs = graph.At.get(p, [])
        if arcs:
            model.addConstr(quicksum(xi[i,j,pp] for (i,j,pp) in arcs) == d)
    if inst.full_batch_count > 0:
        arcs_T = graph.At.get(inst.T, [])
        model.addConstr(
            quicksum(xi[i,j,inst.T] for (i,j,p) in arcs_T) == inst.full_batch_count)
    t0 = time.time()
    model.optimize()
    rt = time.time() - t0

    if model.SolCount == 0:
        status = "infeasible" if model.Status == GRB.INFEASIBLE else "timeout"
        return SolveResult(status=status, cmax=None, lb=None, gap=None,
                           runtime=rt, graph=graph, xi=xi, yi=yi, xt=xt)

    cmax   = model.ObjVal
    lb_obj = model.ObjBound
    gap    = model.MIPGap
    status = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
    print(f"\nCmax = {cmax:.1f}  (gap={gap*100:.4f}%)")

    if verbose:
        print("\nItem arcs used (integer flow):")
        for (i,j,p), v in sorted(xi.items()):
            if v.X > 0.5:
                print(f"  type p={p:3d}  x={round(v.X)}  y={round(yi[i,j,p].X)}  "
                      f"batch {i//T}  ({i:3d}→{j:3d})  Cmax_contrib={j}")
        print("Transition arcs used:")
        for (i,j), v in sorted(xt.items()):
            if v.X > 0.5:
                print(f"  ({i:3d}→{j:3d}) x{round(v.X)}  batch {i//T}→{j//T}")
    xi_vals = {k: v.X for k, v in xi.items()}
    yi_vals = {k: v.X for k, v in yi.items()}
    xt_vals = {k: v.X for k, v in xt.items()}

    return SolveResult(status=status, cmax=cmax, lb=lb_obj, gap=gap,
                       runtime=rt, graph=graph, xi=xi_vals, yi=yi_vals, xt=xt_vals)


# ─────────────────────────────────────────────
# Single-machine LB (reflect arc-flow, phase 1)
# ─────────────────────────────────────────────

def lb_single_machine(inst: SMInstance, time_limit: int = 30,
                      verbose: bool = True) -> int:
    graph = ReflectGraph(inst); T = inst.T

    m = gp.Model("LB_reflect")
    m.Params.TimeLimit = time_limit; m.Params.Threads = 1
    m.Params.MIPGap = 1e-6; m.setParam("OutputFlag", 1 if verbose else 0)

    xi_s = {k: m.addVar(vtype=GRB.INTEGER, lb=0) for k in graph.S_arcs}
    xi_r = {k: m.addVar(vtype=GRB.INTEGER, lb=0) for k in graph.R_arcs}

    for e in graph.nodes:
        if e == 0: continue
        in_s  = quicksum(xi_s[d,i,t] for (d,i,t) in graph.S_arcs if i == e)
        in_r  = quicksum(xi_r[d,i,t] for (d,i,t) in graph.R_arcs if i == e)
        out_s = quicksum(xi_s[i,f,t] for (i,f,t) in graph.S_arcs if i == e)
        out_r = quicksum(xi_r[i,f,t] for (i,f,t) in graph.R_arcs if i == e)
        if in_s.size()+in_r.size()+out_s.size()+out_r.size() > 0:
            m.addConstr(in_s == in_r + out_s + out_r)

    z_expr = quicksum(xi_r[k] for k in graph.R_arcs)
    out_0  = (quicksum(xi_s[d,e,t] for (d,e,t) in graph.S_arcs if d == 0) +
              quicksum(xi_r[d,e,t] for (d,e,t) in graph.R_arcs if d == 0))
    m.addConstr(out_0 == 2 * z_expr)

    s_set = set(graph.S_arcs)
    for t, arcs in graph.Aj.items():
        flow_t = quicksum(xi_s[a] if a in s_set else xi_r[a] for a in arcs)
        m.addConstr(flow_t == inst.item_types[t])

    partial = sum(inst.jobs) - inst.full_bin_count * T
    large   = [p for p in inst.jobs if T/2 < p < T]
    small   = [p for p in inst.jobs if 0   < p <= T/2]
    slack   = len(large)*T - sum(large)
    z_lb    = max(0,
                  math.ceil(partial / T) - 1,
                  len(large) + math.ceil(max(0, sum(small)-slack) / T) - 1)
    m.addConstr(z_expr >= z_lb)
    m.setObjective(z_expr + inst.full_bin_count, GRB.MINIMIZE)
    m.optimize()
    return round(m.ObjVal)


# ─────────────────────────────────────────────
# CSV + solution file helpers
# ─────────────────────────────────────────────

import csv, os, time

CSV_FIELDS = ["instance", "n", "T", "m", "status", "cmax", "lb",
              "gap_pct", "z_max", "runtime_s", "runtime_phase1",
              "model", "numOfThreads", "set"]

def append_csv(csv_path: str, row: dict):
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, delimiter=";")
        if new:
            w.writeheader()
        w.writerow(row)

def write_solution_file(sol_path: str, inst: BPPMInstance,
                        cmax: Optional[float], xi: dict, xt: dict,
                        yi: dict, graph: "BatchGraph"):
    with open(sol_path, "w") as f:
        if cmax is None:
            f.write("Status: no solution\n"); return
        f.write(f"Cmax: {cmax:.1f}\n")
        f.write(f"n={inst.n}  T={inst.T}  m={inst.m}\n\n")
        T = inst.T
        for b in range(graph.z_max):
            arcs_in_batch = [(i,j,p) for (i,j,p) in graph.item_arcs if i//T == b]
            used = [(i,j,p,round(xi[i,j,p])) for (i,j,p) in arcs_in_batch
                    if xi[i,j,p] > 0.5]
            if used:
                f.write(f"Batch {b} (t=[{b*T},{(b+1)*T}]):\n")
                for i,j,p,cnt in sorted(used):
                    f.write(f"  arc ({i:4d}->{j:4d})  type p={p:3d}  x{cnt}\n")
        f.write("\nTransition arcs:\n")
        for (i,j), v in sorted(xt.items()):
            if v > 0.5:
                f.write(f"  ({i:4d}->{j:4d})  x{round(v)}\n")


# ─────────────────────────────────────────────
# Folder runner
# ─────────────────────────────────────────────

def run_folder(folder: str, csv_path: str, sol_dir: str = None,
               t_charge: int = 0, time_limit: float = 3600.0,
               verbose: bool = False, threads: int = 1,
               machines: int = 1, two_phase: bool = False):
    if sol_dir:
        os.makedirs(sol_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(folder)
                   if os.path.isfile(os.path.join(folder, f)))
    if not files:
        print(f"No files found in {folder}"); return

    set_name = os.path.basename(os.path.normpath(folder))
    print(f"Instances : {len(files)}  |  CSV: {csv_path}"
          + (f"  |  Sol: {sol_dir}" if sol_dir else ""))
    print(f"Machines  : {machines}")
    print("-" * 72)

    for idx, fname in enumerate(files, 1):
        if idx-1 in [262,
309,
346,
370,
499,
506]:
            fpath = os.path.join(folder, fname)
            print(f"[{idx:4d}/{len(files)}] {fname:<32s}", end=" ", flush=True)
    
            res = None; inst = None; z_max_used = 0; lb_runtime = 0.0
    
            try:
                inst         = BPPMInstance.from_file(fpath, machines=machines)
                inst_reflect = SMInstance.from_file(fpath, t_charge=t_charge)
    
                t0         = time.time()
                z_single   = lb_single_machine(inst_reflect)
                lb_runtime = time.time() - t0
                z_max      = math.ceil(z_single / machines)
                z_max_used = z_max
    
                res = solve(inst, z_max=z_max, time_limit=time_limit,
                            verbose=verbose, threads=threads)
                res.runtime += lb_runtime   # include LB solve in total runtime
    
            except Exception as exc:
                import traceback
                print(f"ERROR: {exc}")
                traceback.print_exc()
    
            # ── console output ────────────────────────────────────────────
            if res is not None and res.cmax is not None:
                gap_s = f"{res.gap*100:.2f}%" if res.gap is not None else "?"
                print(f"{res.status:8s}  Cmax={res.cmax:>9.1f}  "
                      f"z_max={z_max_used:>3d}  gap={gap_s:>8s}  {res.runtime:.2f}s")
            else:
                status = res.status if res is not None else "error"
                rt     = res.runtime if res is not None else 0.0
                print(f"{status}  {rt:.2f}s")
    
            # ── CSV ───────────────────────────────────────────────────────
            append_csv(csv_path, {
                "instance":     fname,
                "n":            inst.n if inst else "",
                "T":            inst.T if inst else "",
                "m":            machines,
                "status":       res.status  if res  else "error",
                "cmax":         f"{res.cmax:.1f}"        if res and res.cmax is not None else "",
                "lb":           f"{res.lb:.2f}"          if res and res.lb   is not None else "",
                "gap_pct":      f"{res.gap*100:.4f}"     if res and res.gap  is not None else "",
                "z_max":        z_max_used,
                "runtime_s":    f"{res.runtime:.3f}"     if res else "0.000",
                "runtime_phase1": f"{lb_runtime:.3f}",
                "model":        "batch_graph_int",
                "numOfThreads": threads,
                "set":          set_name,
            })
    
            # ── solution file ─────────────────────────────────────────────
            if sol_dir and res is not None and res.cmax is not None:
                write_solution_file(
                    os.path.join(sol_dir, fname + ".sol"),
                    inst, res.cmax, res.xi, res.xt, res.yi, res.graph)
    
        print("-" * 72)
        print(f"Done. Results -> {csv_path}")


# ─────────────────────────────────────────────
# Schedule reconstruction
# ─────────────────────────────────────────────

def reconstruct_schedule(inst: BPPMInstance, res: SolveResult) -> List[Dict]:
    """
    Decompose the integer arc-flow solution into m machine schedules.

    Algorithm: standard flow-path decomposition on a DAG.
      - One pass per machine = one unit of flow from node 0 to UB.
      - At each node, greedily follow item arcs (collecting items),
        then take one transition arc to the next batch boundary.

    Returns list of m dicts  {batch_index -> [processing_times]}.
    """
    from collections import defaultdict

    graph = res.graph
    T, m  = inst.T, inst.m
    UB    = graph.UB

    # Mutable integer flow copies
    xi = defaultdict(int, {k: round(v) for k, v in res.xi.items() if v > 0.5})
    xt = defaultdict(int, {k: round(v) for k, v in res.xt.items() if v > 0.5})

    # Outgoing arc lookups (sorted for determinism)
    item_out: Dict[int, List] = defaultdict(list)
    for arc in sorted(graph.item_arcs):
        item_out[arc[0]].append(arc)

    trans_out: Dict[int, List] = defaultdict(list)
    for arc in sorted(graph.trans_arcs):
        trans_out[arc[0]].append(arc)

    machines = []

    for machine_num in range(m):
        node     = 0
        schedule: Dict[int, List[int]] = {}   # batch_b -> [p, ...]

        while node != UB:
            b           = node // T
            batch_items: List[int] = []

            # Follow item arcs, recording batch crossings on the fly.
            # An arc (i, j, p) can land on j = (b+1)*T (batch boundary),
            # so j // T may be b+1.  When that happens save the current
            # batch and start accumulating for the new one immediately.
            while True:
                moved = False
                for arc in item_out[node]:
                    i, j, p = arc
                    if xi[arc] > 0:
                        batch_items.append(p)
                        xi[arc] -= 1
                        node = j
                        moved = True
                        break    # restart from new node
                if not moved:
                    break
                # Detect batch boundary crossing mid-path
                new_b = node // T
                if new_b != b:
                    if batch_items:
                        schedule[b] = batch_items
                    batch_items = []
                    b = new_b

            if batch_items:
                schedule[b] = batch_items

            if node == UB:
                break

            # Take one transition arc to next batch boundary
            taken = False
            for arc in trans_out[node]:
                i, j = arc
                if xt[arc] > 0:
                    xt[arc] -= 1
                    node = j
                    taken = True
                    break

            if not taken:
                print(f"  WARNING machine {machine_num}: stuck at node {node}")
                break

        machines.append(schedule)

    return machines


def format_schedule(inst: BPPMInstance, machines: List[Dict],
                    cmax: Optional[float] = None) -> str:
    """
    Format machine schedules as a human-readable string,
    mapping processing times back to original job indices.
    """
    from collections import defaultdict

    lines = []
    if cmax is not None:
        lines.append(f"Cmax = {cmax:.1f}")
    lines.append(f"n={inst.n}  T={inst.T}  m={inst.m}")
    lines.append("")

    # Build job pool: p -> [job_indices] (consumed as we assign)
    pool: Dict[int, List[int]] = defaultdict(list)
    for idx, p in enumerate(inst.jobs):
        pool[p].append(idx)

    for k, schedule in enumerate(machines):
        active_batches = sorted(schedule.keys())
        lines.append(f"Machine {k}  ({len(active_batches)} bin(s)):")
        for b in active_batches:
            ptimes  = schedule[b]
            indices = sorted(pool[p].pop(0) for p in ptimes)
            # compute actual completion: cumulative sum within batch
            pos = b * inst.T
            for p in ptimes:
                pos += p
            lines.append(f"  Batch {b}: jobs {indices}  "
                         f"load={sum(ptimes)}  completion={pos}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

def run_single(fpath: str,
               machines:   int   = 1,
               time_limit: float = 3600.0,
               verbose:    bool  = True,
               threads:    int   = 1,
               t_charge:   int   = 0,
               two_phase:  bool  = False) -> SolveResult:
    import os
    fname        = os.path.basename(fpath)
    inst         = BPPMInstance.from_file(fpath, machines=machines)
    inst_reflect = SMInstance.from_file(fpath, t_charge=t_charge)
    print(f"Instance : {fname}\n  {inst.summary()}")

    z_single = lb_single_machine(inst_reflect)
    z_max    = math.ceil(z_single / machines)
    print(f"  z_single={z_single}  →  z_max={z_max}  (UB={z_max*inst.T})")

    res = solve(inst, z_max=z_max,
                time_limit=time_limit,
                verbose=verbose,
                threads=threads)

    if res.cmax is not None:
        sched = reconstruct_schedule(inst, res)
        print()
        print(format_schedule(inst, sched, cmax=res.cmax))

    return res


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # --- single instance ---
    #"""
    setstr = "MOD"
    file   = "L_00000262"
    result = run_single(
        f"Benchmark Instances/Instances/{setstr}/{file}",
        t_charge   = 0,
        time_limit = 100,
        machines   = 5,
    )
    print(f"Final Cmax: {result.cmax}")
    #"""
    """
    # --- full folder ---
    folder = "MOD"
    run_folder(
        folder     = f"Benchmark Instances/Instances/{folder}/",
        csv_path   = f"results/{folder}_new_arcflow_5m_fixed2.csv",
        sol_dir    = f"results/{folder}_new_arcflow_5m_fixed2_sol/",
        t_charge   = 0,
        time_limit = 720.0,
        threads    = 1,
        verbose    = False,
        machines   = 5,
    )#"""