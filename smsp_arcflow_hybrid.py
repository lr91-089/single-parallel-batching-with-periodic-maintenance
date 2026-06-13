# -*- coding: utf-8 -*-
"""
Arc-Flow Reflect model for P_m|pm|Cmax — two-phase only.

Phase 1 : minimise z* = total non-last bins (machine-agnostic reflect graph).
Phase 2 : fix z*, add last-batch graph, introduce scalar z_k (INTEGER in
          [floor(z*/m), ceil(z*/m)]) for the Cmax-machine bin count.
          Cmax = z_k*(T+tc) + last_bin_load.

By the one-bin-apart lemma every machine has either floor(z*/m) or
ceil(z*/m) non-last bins, so z_k fully captures the Cmax machine without
any per-machine u[t,k]/ub[k] assignment variables.
"""

from __future__ import annotations
import argparse, csv, math, os, re, time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
from gurobipy import GRB, quicksum


# ─────────────────────────────────────────────
# Instance
# ─────────────────────────────────────────────

@dataclass
class SMInstance:
    n:          int
    jobs:       List[int]
    T:          int
    t_charge:   int = 0
    item_types: Dict[int, int] = field(default_factory=dict, init=False)
    machines:   int = 1

    def __post_init__(self):
        c: Dict[int, int] = {}
        for p in self.jobs:
            c[p] = c.get(p, 0) + 1
        self.full_bin_items = {t: d for t, d in c.items() if t == self.T}
        self.full_bin_count = sum(self.full_bin_items.values())
        self.item_types     = {t: d for t, d in c.items() if t < self.T}

    @classmethod
    def from_file(cls, path: str, t_charge: int = 0, machines=1) -> "SMInstance":
        with open(path) as f:
            vals = [int(v) for v in f.read().split()]
        n    = vals[0]
        jobs = vals[1:n + 1]
        T    = vals[n + 1]
        if len(jobs) != n:
            raise ValueError(f"Expected {n} jobs, got {len(jobs)} in {path}")
        return cls(n=n, jobs=jobs, T=T, t_charge=t_charge, machines=machines)

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
        M = [0] * (H + 1); M[0] = 1; self.nodes.add(0)
        for wi, di in items:
            if wi > C:
                raise ValueError(f"p={wi} > T={C}: infeasible")
            Hp = [0] * (H + 1)
            for _ in range(di):
                for l in range(H - 1, -1, -1):
                    if Hp[l] == 0 and M[l] == 1:
                        Hp[l] = 1
                        if l + wi <= H:
                            arc = (l, l + wi, wi)
                            self.S_arcs.append(arc); self.Aj[wi].append(arc)
                            self.nodes.add(l + wi); M[l + wi] = 1
                        elif l <= C - (l + wi):
                            arc = (l, C - (l + wi), wi)
                            self.R_arcs.append(arc); self.Aj[wi].append(arc)
                            self.nodes.add(C - (l + wi))
        self.nodes.add(H)
        sv = sorted(self.nodes)
        for u, v in zip(sv, sv[1:]):
            if v <= H:
                self.S_arcs.append((u, v, None))
        self.R_arcs.append((H, H, None))

    def summary(self) -> str:
        si = sum(1 for *_, t in self.S_arcs if t is not None)
        sl = sum(1 for *_, t in self.S_arcs if t is None)
        ri = sum(1 for *_, t in self.R_arcs if t is not None)
        rl = sum(1 for *_, t in self.R_arcs if t is None)
        return (f"C={self.C}, H={self.H}, nodes={len(self.nodes)}, "
                f"S={si}+{sl}loss, R={ri}+{rl}special")


# ─────────────────────────────────────────────
# Last-batch graph  (single Mrad batch, no offset)
# ─────────────────────────────────────────────

class LastBatchGraph:
    def __init__(self, inst: SMInstance):
        self.T            = inst.T
        self.types_sorted = sorted(inst.item_types.keys(), reverse=True)
        self.item_arcs:  List[Tuple[int, int, int]] = []
        self.loss_arcs:  List[Tuple[int, int]]      = []   # (v, T)
        self.nodes:      Set[int]                   = set()
        self._arc_set:   Set[Tuple[int, int, int]]  = set()
        self._build(inst)

    def _build(self, inst: SMInstance):
        T = self.T
        types_sorted = self.types_sorted  # descending
    
        # reachable[lev] = set of nodes reachable using only item types at indices < lev
        reachable: List[Set[int]] = [set() for _ in range(len(types_sorted) + 1)]
        reachable[0].add(0)
        self.nodes.add(0)
    
        for lev, p in enumerate(types_sorted):
            # inherit all nodes from previous level
            reachable[lev + 1] = set(reachable[lev])
            # only extend from nodes that existed BEFORE this item type
            for pos in reachable[lev]:
                j = pos + p
                if j <= T:
                    arc = (pos, j, p)
                    if arc not in self._arc_set:
                        self._arc_set.add(arc)
                        self.item_arcs.append(arc)
                    self.nodes.add(j)
                    reachable[lev + 1].add(j)
    
        self.nodes.add(T)
        # loss arcs: every non-sink node -> T
        for v in self.nodes:
            if v != T:
                self.loss_arcs.append((v, T))

    @property
    def At(self) -> Dict[int, List[Tuple]]:
        result: Dict[int, List] = defaultdict(list)
        for arc in self.item_arcs:
            result[arc[2]].append(arc)
        return result

    def summary(self) -> str:
        return (f"T={self.T}, nodes={len(self.nodes)}, "
                f"item_arcs={len(self.item_arcs)}, loss_arcs={len(self.loss_arcs)}")


# ─────────────────────────────────────────────
# Bin reconstruction
# ─────────────────────────────────────────────

def _reconstruct_bins(xi_s_val, xi_r_val, graph):
    xs = defaultdict(int, {k: round(v) for k, v in xi_s_val.items()})
    xr = defaultdict(int, {k: round(v) for k, v in xi_r_val.items()})
    R_paths: Dict[int, List] = defaultdict(list)
    S_paths: Dict[int, List] = defaultdict(list)
    for _ in range(sum(xs.values()) + sum(xr.values()) + 10):
        if not any(v > 0 for v in xs.values()) and not any(v > 0 for v in xr.values()):
            break
        path = []; node = 0; is_R = False; collision = None
        while True:
            ro = [(e, t) for (d, e, t) in graph.R_arcs if d == node and xr[d, e, t] > 0]
            so = [(e, t) for (d, e, t) in graph.S_arcs if d == node and xs[d, e, t] > 0]
            if ro:
                e, t = ro[0]; path.append(t); xr[node, e, t] -= 1
                collision = e; is_R = True; break
            elif so:
                e, t = so[0]; path.append(t); xs[node, e, t] -= 1; node = e
            else:
                collision = node; is_R = False; break
        if not path:
            break
        items = [t for t in path if t is not None]
        (R_paths if is_R else S_paths)[collision].append(items)
    bins = []
    for v in list(R_paths):
        while R_paths[v] and S_paths[v]:
            bins.append(R_paths[v].pop(0) + S_paths[v].pop(0))
    return bins


# ─────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────

@dataclass
class SolverResult:
    status:          str
    cmax:            Optional[float]
    z_nonlast:       Optional[int]
    last_load:       Optional[float]
    lb:              Optional[float]
    gap:             Optional[float]
    runtime:         float
    root_gap:        Optional[float]
    runtime_phase1:  Optional[float]
    phase1_root_gap: Optional[float]
    phase2_root_gap: Optional[float]
    instance:        SMInstance
    bins_indices:    Optional[List[List[int]]] = None
    last_indices:    Optional[List[int]]       = None
    bins_ptimes:     Optional[List[List[int]]] = None
    last_ptimes:     Optional[List[int]]       = None
    machine_bins:    Optional[Dict[int, int]]  = None  # z_k[k] per machine
    last_bins_ptimes:  Optional[List[List[int]]] = None
    last_bins_indices: Optional[List[List[int]]] = None

# ─────────────────────────────────────────────
# Root-gap helpers
# ─────────────────────────────────────────────

def _get_root_gap_from_log(logfile):
    root_bound = None; root_incumbent = None; root_cutoff = False
    with open(logfile, "r") as f:
        for line in f:
            if "Found heuristic solution: objective" in line:
                m = re.search(r"objective\s+([\-0-9.eE+]+)", line)
                if m and root_incumbent is None:
                    root_incumbent = float(m.group(1))
            if "Root relaxation:" in line:
                if "cutoff" in line or "infeasible" in line:
                    root_cutoff = True
                else:
                    m = re.search(r"objective\s+([\-0-9.eE+]+)", line)
                    if m and root_bound is None:
                        root_bound = float(m.group(1))
            if re.match(r"\s*[H\*]\s+0\s+", line):
                m = re.search(r"\s+(\d+\.\d+)\s+\d+\.\d+\s+\d+\.\d+%", line)
                if m and root_incumbent is None:
                    root_incumbent = float(m.group(1))
    if root_incumbent is None and root_bound is not None:
        root_incumbent = 2 * root_bound
    if root_cutoff and root_incumbent is not None:
        return 0.0, root_incumbent, root_incumbent
    if root_bound is not None and root_incumbent is not None and root_incumbent != 0:
        return abs(root_incumbent - root_bound) / abs(root_incumbent), root_bound, root_incumbent
    return None, root_bound, root_incumbent


def _phase_callback(model, where):
    if where == GRB.Callback.MIPNODE:
        if model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
            if model._root_bound is None:
                model._root_bound = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                model._root_obj   = model.cbGet(GRB.Callback.MIPNODE_OBJBST)


# ─────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────

class ArcFlowReflectSMSP:
    """
    Two-phase arc-flow reflect solver for P_m|pm|Cmax.

    Phase 1 — minimise z* (total non-last bins, machine-agnostic).
    Phase 2 — fix z*, introduce per-machine z_k[k] in [floor(z*/m), ceil(z*/m)]
              with sum_k z_k[k] == z*, add last-batch graph, minimise Cmax.

    By the one-bin-apart lemma every machine has either floor(z*/m) or
    ceil(z*/m) non-last bins. The sum constraint links z_k to the reflect
    solution without needing per-machine u[t,k]/ub[k] arc assignment.
    """

    def __init__(self, inst: SMInstance, time_limit: float = 3600.0,
                 verbose: bool = False, threads: int = 1,
                 machines: int = 1):
        self.inst       = inst
        self.time_limit = time_limit
        self.verbose    = verbose
        self.threads    = threads
        self.machines   = machines
        self.graph      = ReflectGraph(inst)
        self.last_graph = LastBatchGraph(inst)
        self.model_name = "arcflow_reflect_zk_two_phase"
        self._xi_s = {}; self._xi_r = {}
        self._xi   = {}; self._yi   = {}; self._xl   = {}
        self._z_k  = None
        self._model = None

    # ── reflect graph helpers ─────────────────────────────────────────────────

    def _add_arc_flow_vars(self, m: gp.Model):
        graph = self.graph
        xi_s = {(d, e, t): m.addVar(vtype=GRB.INTEGER, lb=0, name=f"xs_{d}_{e}_{t}")
                for (d, e, t) in graph.S_arcs}
        xi_r = {(d, e, t): m.addVar(vtype=GRB.INTEGER, lb=0, name=f"xr_{d}_{e}_{t}")
                for (d, e, t) in graph.R_arcs}
        return xi_s, xi_r

    def _add_flow_conservation(self, m: gp.Model, xi_s, xi_r):
        graph = self.graph
        for e in graph.nodes:
            if e == 0:
                continue
            in_s  = quicksum(xi_s[d, i, t] for (d, i, t) in graph.S_arcs if i == e)
            in_r  = quicksum(xi_r[d, i, t] for (d, i, t) in graph.R_arcs if i == e)
            out_s = quicksum(xi_s[i, f, t] for (i, f, t) in graph.S_arcs if i == e)
            out_r = quicksum(xi_r[i, f, t] for (i, f, t) in graph.R_arcs if i == e)
            lhs = in_s; rhs = in_r + out_s + out_r
            if lhs.size() > 0 or rhs.size() > 0:
                m.addConstr(lhs == rhs, name=f"flow_{e}")

    def _add_source_outflow(self, m: gp.Model, xi_s, xi_r):
        graph  = self.graph
        z_expr = quicksum(xi_r[d, e, t] for (d, e, t) in graph.R_arcs)
        out_0  = (quicksum(xi_s[0, e, t] for (d, e, t) in graph.S_arcs if d == 0) +
                  quicksum(xi_r[0, e, t] for (d, e, t) in graph.R_arcs if d == 0))
        m.addConstr(out_0 == 2 * z_expr, name="source_outflow")
        return z_expr

    def _add_z_lower_bound(self, m: gp.Model, z_expr):
        inst      = self.inst; T = inst.T
        total     = sum(inst.jobs)
        full_load = inst.full_bin_count * T
        partial   = total - full_load
        z_lb_load = math.ceil(partial / T) - self.machines
        large     = [p for p in inst.jobs if T / 2 < p < T]
        small     = [p for p in inst.jobs if 0 < p <= T / 2]
        slack     = len(large) * T - sum(large)
        ovf       = max(0, sum(small) - slack)
        z_lb_mt   = len(large) + math.ceil(ovf / T) - self.machines
        z_lb      = max(0, z_lb_load, z_lb_mt)
        print(f"  z_lb={z_lb}")
        m.addConstr(z_expr >= z_lb, name="z_lb")

    # ── demand constraints ────────────────────────────────────────────────────

    def _add_demand_constraints(self, m: gp.Model, xi_s, xi_r, xi):
        """
        xi=None  → phase-1 upper-bound form: reflect_flow[t] <= d[t]
        xi given → phase-2 equality:  reflect_flow[t] + graph_flow[t] == d[t]
        """
        inst  = self.inst; graph = self.graph
        s_set = set(graph.S_arcs); lg = self.last_graph
        for t, d in inst.item_types.items():
            reflect_flow = quicksum(
                xi_s[arc] if arc in s_set else xi_r[arc]
                for arc in graph.Aj.get(t, []))
            if xi is None:
                m.addConstr(reflect_flow == d, name=f"dem_{t}")
            else:
                lb_flow = quicksum(xi[i, j, p] for (i, j, p) in lg.At.get(t, []))
                m.addConstr(reflect_flow + lb_flow == d, name=f"dem_{t}")

    # ── last-batch graph vars ─────────────────────────────────────────────────

    def _add_last_batch_vars(self, m: gp.Model):
        lg = self.last_graph; inst = self.inst
        xi = {(i, j, p): m.addVar(vtype=GRB.INTEGER, lb=0,
                                   ub=inst.item_types[p], name=f"xi_{i}_{j}_{p}")
              for (i, j, p) in lg.item_arcs}
        yi = {(i, j, p): m.addVar(vtype=GRB.BINARY, name=f"yi_{i}_{j}_{p}")
              for (i, j, p) in lg.item_arcs}
        xl = {(u, v): m.addVar(vtype=GRB.INTEGER, lb=0, ub=self.machines,
                                name=f"xl_{u}_{v}")
              for (u, v) in lg.loss_arcs}
        return xi, yi, xl

    # ── last-batch constraints + Cmax via z_k ─────────────────────────────────

    def _add_last_batch_constraints(self, m: gp.Model, xi, yi, xl, Cmax, z_k):
        lg = self.last_graph; inst = self.inst
        T  = inst.T; tc = inst.t_charge; mm = self.machines
        M  = range(mm)

        # xi / yi linking
        for (i, j, p) in lg.item_arcs:
            m.addConstr(xi[i, j, p] >= yi[i, j, p],                          name=f"lb_{i}_{j}_{p}")
            m.addConstr(xi[i, j, p] <= inst.item_types[p] * yi[i, j, p],     name=f"ub_{i}_{j}_{p}")

        # Per-machine Cmax
        for k in M:
            for (i, j, p) in lg.item_arcs:
                m.addConstr(z_k[k] * (T + tc) + j * yi[i, j, p] <= Cmax,
                            name=f"cmax_{k}_{i}_{j}_{p}")

        # Flow conservation at intermediate nodes
        for v in lg.nodes:
            if v == 0 or v == T:
                continue
            in_f  = quicksum(xi[i, v, p] for (i, j, p) in lg.item_arcs if j == v)
            out_f = (quicksum(xi[v, j, p] for (i, j, p) in lg.item_arcs if i == v)
                     + xl[v, T])
            m.addConstr(in_f == out_f, name=f"lb_flow_{v}")

        # Source: exactly mm units leave (item arcs + loss arc from 0)
        source_out = (quicksum(xi[0, j, p] for (i, j, p) in lg.item_arcs if i == 0)
                      + xl[0, T])
        m.addConstr(source_out == mm, name="lb_source")

        # Sink: flow conservation enforces in_T == mm automatically,

    # ── public API ────────────────────────────────────────────────────────────

    def build_model(self) -> gp.Model:
        return self._build_model_two_phase()

    def _build_model_two_phase(self) -> gp.Model:
        inst = self.inst; T = inst.T; tc = inst.t_charge

        m = gp.Model("SMSP_ArcFlowReflect_zk")
        m.Params.TimeLimit = self.time_limit
        m.Params.Threads   = self.threads
        m.Params.MIPGap    = 1e-6
        m.setParam("OutputFlag", 1 if self.verbose else 0)

        xi_s, xi_r = self._add_arc_flow_vars(m)
        self._add_flow_conservation(m, xi_s, xi_r)
        z_expr = self._add_source_outflow(m, xi_s, xi_r)

        # ── Phase 1: minimise z* ──────────────────────────────────────────────
        self._add_demand_constraints(m, xi_s, xi_r, None)
        self._add_z_lower_bound(m, z_expr)
        m.update()

        m.setObjective(z_expr + inst.full_bin_count, GRB.MINIMIZE)
        m._root_bound = None; m._root_obj = None
        m.optimize(_phase_callback)

        phase1_time = m.Runtime
        if m.SolCount == 0:
            m._phase1_time     = phase1_time
            m._phase1_root_gap = None
            m._phase2_ready    = False
            self._model = m; self._xi_s = xi_s; self._xi_r = xi_r
            return m

        z_opt = round(m.ObjVal)
        phase1_root_gap = (
            abs(m._root_bound - z_opt) / abs(z_opt)
            if m._root_bound is not None and z_opt != 0 else 0.0)

        print(f"  Phase 1: z*={z_opt}  (runtime {phase1_time:.2f}s)")


        # ── Phase 2 setup ─────────────────────────────────────────────────────
        z_arc = z_opt - inst.full_bin_count

        # Remove phase-1 demand upper bounds
        for t in inst.item_types:
            c = m.getConstrByName(f"dem_{t}")
            if c is not None:
                m.remove(c)

        # Fix total reflect bin count
        # (allow up to m slack: last bins not yet counted)
        m.addConstr(z_expr >= max(0, z_arc - self.machines), name="fix_z_lb")
        m.addConstr(z_expr <= z_arc,                          name="fix_z_ub")
        # z_k[k]: non-last bin count per machine, in [z_min, z_max]
        # sum_k z_k[k] == z_opt links the machine assignment to the reflect solution
        z_min = max(0, (z_opt - self.machines) // self.machines)
        z_max = math.ceil(z_opt / self.machines)
        M     = range(self.machines)
        z_k   = {k: m.addVar(vtype=GRB.INTEGER, lb=z_min, ub=z_max, name=f"z_k_{k}")
                 for k in M}
        m.addConstr(quicksum(z_k[k] for k in M) == z_expr+inst.full_bin_count, name="z_k_sum")
        for k in M[1:]:
            m.addConstr(z_k[k]<=z_k[k-1])

        # Last-batch graph
        xi, yi, xl = self._add_last_batch_vars(m)
        Cmax = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Cmax")
        m.update()

        self._add_demand_constraints(m, xi_s, xi_r, xi)
        self._add_last_batch_constraints(m, xi, yi, xl, Cmax, z_k)

        # Global load lower bound on Cmax
        cmax_lb = math.ceil(sum(inst.jobs) / self.machines)
        m.addConstr(Cmax >= cmax_lb, name="cmax_lb_load")

        m.setObjective(Cmax, GRB.MINIMIZE)
        m._root_bound      = None; m._root_obj = None
        m._phase1_time     = phase1_time
        m._phase1_root_gap = phase1_root_gap
        m._phase2_ready    = True

        self._model = m; self._xi_s = xi_s; self._xi_r = xi_r
        self._xi = xi; self._yi = yi; self._xl = xl; self._z_k = z_k; self._z_opt = z_opt
        return m

    # ── solve ─────────────────────────────────────────────────────────────────

    def solve(self) -> SolverResult:
        if self._model is None:
            self.build_model()
        return self._solve_two_phase()

    def _solve_two_phase(self) -> SolverResult:
        m = self._model; t0 = time.time()

        if not getattr(m, "_phase2_ready", False):
            rt = time.time() - t0 + m._phase1_time
            return SolverResult(
                status="infeasible" if m.Status == GRB.INFEASIBLE else "timeout",
                cmax=None, z_nonlast=None, last_load=None,
                lb=None, gap=None, runtime=rt, root_gap=None,
                runtime_phase1=m._phase1_time,
                phase1_root_gap=m._phase1_root_gap,
                phase2_root_gap=None, instance=self.inst)

        m.optimize(_phase_callback)
        st = m.Status

        phase2_root_gap = (
            abs(m._root_bound - m.ObjVal) / abs(m.ObjVal)
            if m._root_bound is not None and m.SolCount > 0 and m.ObjVal != 0
            else 0.0)

        rt = time.time() - t0 + m._phase1_time

        if m.SolCount == 0:
            if m.Status == GRB.INFEASIBLE:
                print("[IIS] Model infeasible — computing IIS...")
                m.computeIIS()
                m.write("hybrid_iis.ilp")
            return SolverResult(
                status="infeasible" if st == GRB.INFEASIBLE else "timeout",
                cmax=None, z_nonlast=None, last_load=None,
                lb=None, gap=None, runtime=rt, root_gap=None,
                runtime_phase1=m._phase1_time,
                phase1_root_gap=m._phase1_root_gap,
                phase2_root_gap=phase2_root_gap, instance=self.inst)

        z_val     = (round(sum(v.X for v in self._xi_r.values()))
                     + self.inst.full_bin_count)
        vals      = [j * v.X for (i, j, p), v in self._yi.items() if v.X > 0.5]
        last_load = max(vals) if vals else 0.0
        cmax      = m.ObjVal
        status    = "optimal" if st == GRB.OPTIMAL else "feasible"

        return self._build_result(
            status=status, cmax=cmax, z_val=z_val, last_load=last_load,
            lb=m.ObjBound, gap=m.MIPGap, runtime=rt,
            runtime_phase1=m._phase1_time,
            phase1_root_gap=m._phase1_root_gap,
            phase2_root_gap=phase2_root_gap)

    # ── result builder ────────────────────────────────────────────────────────

    def _build_result(self, *, status, cmax, z_val, last_load,
                      lb, gap, runtime, runtime_phase1,
                      phase1_root_gap, phase2_root_gap) -> SolverResult:
        inst = self.inst

        try:
            for v in self._model.getVars():
                if v.X > 1e-5:
                    print(f"{v.varName} = {v.X:.4g}")
        except Exception:
            pass

        xs_val      = {k: v.X for k, v in self._xi_s.items() if v.X > 0.5}
        xr_val      = {k: v.X for k, v in self._xi_r.items() if v.X > 0.5}
        bins_ptimes = _reconstruct_bins(xs_val, xr_val, self.graph)
        for t, count in inst.full_bin_items.items():
            for _ in range(count):
                bins_ptimes.append([t])

        last_ptimes: List[int] = []
        for (i, j, p), v in self._yi.items():
            if v.X > 0.5:
                last_ptimes.extend([p] * round(self._xi[i, j, p].X))
        last_ptimes.sort()

        full_pool: Dict[int, List[int]] = defaultdict(list)
        for idx, p in enumerate(inst.jobs):
            full_pool[p].append(idx)

        bins_idx = [sorted(full_pool[p].pop(0) for p in pt) for pt in bins_ptimes]
        if bins_idx:
            bins_idx, bins_ptimes = zip(
                *sorted(zip(bins_idx, bins_ptimes), key=lambda x: x[0][0]))
            bins_idx    = list(bins_idx)
            bins_ptimes = list(bins_ptimes)
            
        xi_val = {k: v.X for k, v in self._xi.items() if v.X > 0.5}
        xl_val = {k: v.X for k, v in self._xl.items() if v.X > 0.5}

        last_bins_ptimes = _reconstruct_last_bins(xi_val, xl_val, self.last_graph)
        while len(last_bins_ptimes) < self.machines:
            last_bins_ptimes.append([])

        last_bins_indices = [sorted(full_pool[p].pop(0) for p in pt)
                             for pt in last_bins_ptimes]
        last_ptimes = [p for pt in last_bins_ptimes for p in pt]
        last_idx    = [i for bi in last_bins_indices for i in bi]

        machine_bins = {k: round(self._z_k[k].X) for k in range(self.machines)} \
                       if self._z_k is not None else None

        return SolverResult(
            status=status, cmax=cmax, z_nonlast=int(z_val),
            last_load=last_load, lb=lb, gap=gap, runtime=runtime,
            root_gap=None,
            runtime_phase1=runtime_phase1,
            phase1_root_gap=phase1_root_gap,
            phase2_root_gap=phase2_root_gap,
            instance=inst,
            bins_indices=bins_idx, last_indices=last_idx,
            bins_ptimes=bins_ptimes, last_ptimes=last_ptimes,
            machine_bins=machine_bins, last_bins_ptimes=last_bins_ptimes,
            last_bins_indices=last_bins_indices)


# ─────────────────────────────────────────────
# Solution file
# ─────────────────────────────────────────────

def _reconstruct_last_bins(xi_val, xl_val, graph: LastBatchGraph) -> List[List[int]]:
    T  = graph.T
    xi = defaultdict(int, {k: round(v) for k, v in xi_val.items()})
    xl = defaultdict(int, {k: round(v) for k, v in xl_val.items()})

    total_flow = (sum(xi[0, j, p] for (i, j, p) in graph.item_arcs if i == 0)
                  + xl[0, T])
    bins: List[List[int]] = []

    for _ in range(total_flow):
        path: List[int] = []
        node = 0
        while node != T:
            moved = False
            for (i, j, p) in graph.item_arcs:
                if i == node and xi[i, j, p] > 0:
                    xi[i, j, p] -= 1
                    path.append(p)
                    node = j
                    moved = True
                    break
            if not moved:
                # take loss arc to T
                xl[node, T] -= 1
                node = T
        bins.append(path)

    bins.sort(key=sum, reverse=True)
    return bins


def _format_solution(res: SolverResult) -> str:
    if res.cmax is None:
        return f"Status: {res.status}\nNo feasible solution found.\n"
    n_last = sum(1 for pt in res.last_bins_ptimes if pt) if res.last_bins_ptimes else res.instance.machines
    lines = [f"Makespan : {res.cmax}",
             f"Batches  : {res.z_nonlast} non-last + {n_last} last",
             ""]

    lines.append("Non-last batches (shared pool):")
    for b, (idx, pt) in enumerate(zip(res.bins_indices, res.bins_ptimes)):
        lines.append(f"  Batch {b:2d}: jobs={idx}  load={sum(pt)}")

    lines.append("")
    lines.append("Last batch per machine (largest load to machine 0):")
    if res.last_bins_ptimes is not None:
        lbp = res.last_bins_ptimes
        lbi = res.last_bins_indices or [[]] * res.instance.machines
        for k in range(res.instance.machines):
            pt    = lbp[k] if k < len(lbp) else []
            idx   = lbi[k] if k < len(lbi) else []
            z_k   = res.machine_bins[k] if res.machine_bins else "?"
            total = (z_k + 1) if isinstance(z_k, int) and pt else z_k
            if pt:
                lines.append(f"  Machine {k}: {z_k} non-last + last(load={sum(pt)})  "
                             f"jobs={idx}  total_bins={total}")
            else:
                lines.append(f"  Machine {k}: {z_k} non-last only  total_bins={total}")

    return "\n".join(lines) + "\n"


def write_solution_file(res: SolverResult, path: str):
    with open(path, "w") as f:
        f.write(_format_solution(res))


# ─────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────

CSV_FIELDS = [
    "instance", "n", "T", "status", "cmax", "lb", "gap_pct",
    "z_nonlast", "last_load", "runtime_s", "runtime_phase1",
    "root_gap", "root_gap_phase2",
    "model", "numOfThreads", "set", "comment", "machines",
]


def append_csv(path: str, res: SolverResult, name: str,
               model: str = "", num_threads: int = 1,
               set_name: str = "", machines: int = 1):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, delimiter=";")
        if new:
            w.writeheader()
        root_gap_col = res.root_gap if res.root_gap is not None else res.phase1_root_gap
        w.writerow({
            "instance":        name,
            "n":               res.instance.n,
            "T":               res.instance.T,
            "status":          res.status,
            "cmax":            f"{res.cmax:.1f}"      if res.cmax      is not None else "",
            "lb":              f"{res.lb:.2f}"        if res.lb        is not None else "",
            "gap_pct":         f"{res.gap*100:.4f}"   if res.gap       is not None else "",
            "z_nonlast":       res.z_nonlast          if res.z_nonlast is not None else "",
            "last_load":       f"{res.last_load:.1f}" if res.last_load is not None else "",
            "runtime_s":       f"{res.runtime:.3f}",
            "runtime_phase1":  (f"{res.runtime_phase1:.3f}"
                                if res.runtime_phase1 is not None else ""),
            "root_gap":        (f"{root_gap_col:.8f}" if root_gap_col is not None else ""),
            "root_gap_phase2": (f"{res.phase2_root_gap:.8f}"
                                if res.phase2_root_gap is not None else ""),
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
    if sol_dir:
        os.makedirs(sol_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(folder)
                   if os.path.isfile(os.path.join(folder, f)))
    if not files:
        print(f"No files found in {folder}"); return
    print(f"Instances : {len(files)}  |  CSV: {csv_path}")
    print("-" * 72)
    set_name = os.path.basename(os.path.normpath(folder))
    for i, fname in enumerate(files, 1):
        fpath = os.path.join(folder, fname)
        print(f"[{i:4d}/{len(files)}] {fname:<32s}", end=" ", flush=True)
        try:
            inst   = SMInstance.from_file(fpath, t_charge=t_charge, machines=machines)
            solver = ArcFlowReflectSMSP(inst, time_limit=time_limit,
                                         verbose=verbose, threads=threads,
                                         machines=machines)
            solver.build_model()
            res = solver.solve()
            if res.cmax is not None:
                gap_s = f"{res.gap*100:.2f}%" if res.gap is not None else "?"
                print(f"{res.status:8s}  Cmax={res.cmax:>9.1f}  "
                      f"z={res.z_nonlast:>3d}  gap={gap_s:>8s}  {res.runtime:.2f}s")
            else:
                print(f"{res.status}  {res.runtime:.2f}s")
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"ERROR: {exc}")
            dummy = SMInstance(n=0, jobs=[], T=0, t_charge=t_charge)
            res   = SolverResult(
                status="error", cmax=None, z_nonlast=None, last_load=None,
                lb=None, gap=None, runtime=0.0, root_gap=None,
                runtime_phase1=None, phase1_root_gap=None,
                phase2_root_gap=None, instance=dummy)
        append_csv(csv_path, res, fname,
                   model=solver.model_name, num_threads=threads,
                   set_name=set_name, machines=machines)
        if sol_dir and res.status != "error":
            write_solution_file(res, os.path.join(sol_dir, fname + ".sol"))
    print("-" * 72)
    print(f"Done. Results -> {csv_path}")


def run_single(fpath, csv_path=None, sol_dir=None, t_charge=0,
               time_limit=3600.0, verbose=True, threads=1, machines=1):
    fname = os.path.basename(fpath)
    inst  = SMInstance.from_file(fpath, t_charge=t_charge,machines=machines)
    print(f"Instance : {fname}\n  {inst.summary()}")
    print(f"Reflect  : {ReflectGraph(inst).summary()}")
    print(f"LastBatch: {LastBatchGraph(inst).summary()}")

    solver = ArcFlowReflectSMSP(inst, time_limit=time_limit,
                                  verbose=verbose, threads=threads,
                                  machines=machines)
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
        print(f"z_k per machine      : { {k: round(solver._z_k[k].X) for k in range(solver.machines)} }")
        print(f"Phase-1 runtime   : {res.runtime_phase1:.2f}s")
        print(f"Phase-1 root gap  : {res.phase1_root_gap}")
        print(f"Phase-2 root gap  : {res.phase2_root_gap}")
    print(f"Runtime  : {res.runtime:.2f}s")
    if res.bins_indices is not None:
        print(_format_solution(res), end="")
    if csv_path:
        append_csv(csv_path, res, fname,
                   model=solver.model_name, num_threads=threads,
                   set_name=set_name, machines=machines)
        print(f"\nCSV -> {csv_path}")
    if sol_dir:
        os.makedirs(sol_dir, exist_ok=True)
        sp = os.path.join(sol_dir, fname + ".sol")
        write_solution_file(res, sp)
        print(f"Sol -> {sp}")
    return res


# ─────────────────────────────────────────────
# Example invocations
# ─────────────────────────────────────────────

#"""
# --- Single instance run ---
setstr = "MOD"
file   = "L_example"
result = run_single(
    f"Benchmark Instances/Instances/{setstr}/{file}",
    t_charge   = 0,
    time_limit = 720,
    machines   = 2
)
print(result.cmax, result.z_nonlast, result.last_load)
#"""

"""
# --- Folder run ---
folder = "MOD"
run_folder(
    folder     = f"Benchmark Instances/Instances/{folder}/",
    csv_path   = f"results/{folder}_results_hybrid_zk_m10_symmCheck.csv",
    #sol_dir = f"results/{folder}_solutions_hybrid_zk_m10",
    t_charge   = 0,
    time_limit = 720.0,
    threads    = 1,
    verbose    = True,
    machines   = 10,
)
#"""