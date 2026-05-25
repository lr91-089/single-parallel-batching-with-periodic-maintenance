# -*- coding: utf-8 -*-
"""
Arc-Flow Reflect model for P_m|pm|Cmax  (single-machine: m=1)
(Parallel-Machine Scheduling with Periodic Maintenance, minimize makespan).

Model
-----
  Non-last bins : FRE reflect arc-flow  (Delorme & Iori 2020)
  Last bin      : single-batch Mrad arc-flow graph (no offset, positions [0,T])
                  replaces the scalar y[t] / y[t,k] variables.

  Last-batch graph
  ----------------
    Nodes        : subset of [0..T] reachable by item-type arcs
    Item arcs    : (i, j, p)  where j = i+p, INTEGER flow xi[i,j,p]
    Indicator    : yi[i,j,p]  BINARY,  xi >= yi,  xi <= d_p * yi
    Loss arcs    : xl[v]  INTEGER in [0, m], node v -> T  (machines with
                   no last-bin items drain here)
    Source inflow: <= m   (machines that finish exactly on a bin boundary
                   contribute 0 to the last-batch graph)
    Cmax link    : z_nonlast*(T+tc) + j*yi[i,j,p] <= Cmax  for every arc

  Demand split   : reflect_flow[p] + lastbatch_flow[p] == d[p]   for all p

  Objective (1-ph) : min Cmax
  Objective (2-ph) : phase 1 -> min z_nonlast (machine-agnostic bin count)
                     phase 2 -> fix z_nonlast, min Cmax

  Note: for machines=1 the two-phase approach is still valid.
        For machines>1 phase 1 is machine-agnostic (arc-flow only, no last
        batch structure needed) and phase 2 adds the full parallel model.

Instance format (LOW / MOD sets)
---------------------------------
  Line 1       : n
  Lines 2..n+1 : p_j  (one per line)
  Last line    : T

Usage
-----
  # one instance
  python smsp_arcflow_lastbatch.py --instance path/to/file --out_csv r.csv

  # whole folder
  python smsp_arcflow_lastbatch.py --folder path/to/LOW --out_csv r.csv
"""

from __future__ import annotations
import warnings
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

    def __post_init__(self):
        c: Dict[int, int] = {}
        for p in self.jobs:
            c[p] = c.get(p, 0) + 1
        self.full_bin_items = {t: d for t, d in c.items() if t == self.T}
        self.full_bin_count = sum(self.full_bin_items.values())
        self.item_types     = {t: d for t, d in c.items() if t < self.T}

    @classmethod
    def from_file(cls, path: str, t_charge: int = 0) -> "SMInstance":
        with open(path) as f:
            vals = [int(v) for v in f.read().split()]
        n    = vals[0]
        jobs = vals[1:n + 1]
        T    = vals[n + 1]
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
    """
    Integer arc-flow graph for exactly one bin of capacity T,
    with nodes in [0, T].

    Item arcs  : (i, j, p)   j = i + p,  reachable from 0
    Loss arc   : every node v -> T  (so machines with slack can exit)

    Items are processed largest-first (same ordering as Mrad) to reduce
    the number of reachable nodes and break assignment symmetry.
    """

    def __init__(self, inst: SMInstance):
        self.T            = inst.T
        self.types_sorted = sorted(inst.item_types.keys(), reverse=True)
        self.item_arcs:  List[Tuple[int, int, int]] = []
        self.nodes:      Set[int]                   = set()
        self._arc_set:   Set[Tuple[int, int, int]]  = set()
        self._build(inst)

    def _build(self, inst: SMInstance):
        T       = self.T
        n_types = len(self.types_sorted)

        # Level-by-level reachability (same as MradGraph._build)
        level_nodes: Dict[int, Set[int]] = defaultdict(set)
        level_nodes[0].add(0)
        self.nodes.add(0)

        for lev, p in enumerate(self.types_sorted):
            frontier = list(level_nodes[lev])
            while frontier:
                next_frontier = []
                for pos in frontier:
                    j = pos + p
                    if j <= T:
                        arc = (pos, j, p)
                        if arc not in self._arc_set:
                            self._arc_set.add(arc)
                            self.item_arcs.append(arc)
                        if j not in level_nodes[lev]:
                            level_nodes[lev].add(j)
                            self.nodes.add(j)
                            next_frontier.append(j)
                frontier = next_frontier

            # Carry reachable nodes forward to next level
            if lev < n_types - 1:
                for pos in level_nodes[lev]:
                    if pos not in level_nodes[lev + 1]:
                        level_nodes[lev + 1].add(pos)
                        self.nodes.add(pos)

        self.nodes.add(T)

    @property
    def At(self) -> Dict[int, List[Tuple]]:
        result: Dict[int, List] = defaultdict(list)
        for arc in self.item_arcs:
            result[arc[2]].append(arc)
        return result

    @property
    def loss_nodes(self) -> List[int]:
        """Nodes eligible for a loss arc to T (all nodes except T itself)."""
        return [v for v in self.nodes if v != self.T]

    def summary(self) -> str:
        return (f"T={self.T}, nodes={len(self.nodes)}, "
                f"item_arcs={len(self.item_arcs)}")


# ─────────────────────────────────────────────
# Bin reconstruction from reflect arc-flow solution
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
# Result dataclass
# ─────────────────────────────────────────────

@dataclass
class SolverResult:
    status:              str
    cmax:                Optional[float]
    z_nonlast:           Optional[int]
    last_load:           Optional[float]
    lb:                  Optional[float]
    gap:                 Optional[float]
    runtime:             float
    root_gap:            Optional[float]
    runtime_phase1:      Optional[float]
    phase1_root_gap:     Optional[float]
    phase2_root_gap:     Optional[float]
    instance:            SMInstance
    bins_indices:        Optional[List[List[int]]] = None
    last_indices:        Optional[List[int]]       = None
    bins_ptimes:         Optional[List[List[int]]] = None
    last_ptimes:         Optional[List[int]]       = None
    machine_assignments: Optional[Dict[int, Dict]] = None


# ─────────────────────────────────────────────
# Root-gap helpers
# ─────────────────────────────────────────────

def _get_root_gap_from_log(logfile, obj_val=None):
    root_bound     = None
    root_incumbent = None
    root_cutoff    = False
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
        gap = abs(root_incumbent - root_bound) / abs(root_incumbent)
        return gap, root_bound, root_incumbent
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
    Arc-flow reflect solver for P_m|pm|Cmax with a last-batch Mrad graph
    replacing the scalar y[t] / y[t,k] last-bin variables.

    Last-batch graph
    ----------------
    Nodes in [0, T] (no offset).  Item arcs xi[i,j,p] (INTEGER), indicator
    arcs yi[i,j,p] (BINARY), loss arcs xl[v] (INTEGER, ub=m).
    Source inflow <= m.
    Cmax: z_nonlast*(T+tc) + j*yi[i,j,p] <= Cmax  for every item arc.

    Parameters
    ----------
    two_phase : bool
        Phase 1 minimises z_nonlast (machine-agnostic, no last-batch vars).
        Phase 2 fixes z_nonlast and minimises Cmax using the full model.
    """

    def __init__(self, inst: SMInstance, time_limit: float = 3600.0,
                 verbose: bool = False, threads: int = 1,
                 machines: int = 1, two_phase: bool = False):
        self.inst        = inst
        self.time_limit  = time_limit
        self.verbose     = verbose
        self.threads     = threads
        self.machines    = machines
        self.two_phase   = two_phase
        self.graph       = ReflectGraph(inst)
        self.last_graph  = LastBatchGraph(inst)
        self.model_name  = "arcflow_reflect_lastbatch"
        if two_phase:
            self.model_name += "_two_phase"
        self._xi_s = {}; self._xi_r = {}
        self._xi   = {}; self._yi   = {}; self._xl   = {}
        self._u    = None; self._ub  = None
        self._model = None

    # ── internal helpers ──────────────────────────────────────────────────────

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

    def _add_last_batch_vars(self, m: gp.Model):
        """Add xi, yi, xl variables for the last-batch Mrad graph."""
        lg   = self.last_graph
        inst = self.inst
        xi = {(i, j, p): m.addVar(vtype=GRB.INTEGER, lb=0,
                                   ub=inst.item_types[p],
                                   name=f"xi_{i}_{j}_{p}")
              for (i, j, p) in lg.item_arcs}
        yi = {(i, j, p): m.addVar(vtype=GRB.BINARY, name=f"yi_{i}_{j}_{p}")
              for (i, j, p) in lg.item_arcs}
        xl = {v: m.addVar(vtype=GRB.INTEGER, lb=0, ub=self.machines,
                          name=f"xl_{v}")
              for v in lg.loss_nodes}
        return xi, yi, xl

    def _add_last_batch_constraints(self, m: gp.Model, xi, yi, xl,
                                    Cmax, z_expr):
        """
        Flow conservation + source inflow + xi/yi linking + Cmax constraints
        for the last-batch graph.

        z_expr  : Gurobi linear expression for the number of non-last bins
                  (reflect z, NOT including full-bin count yet).
        Cmax    : Gurobi variable (or expression for single-machine case).
        """
        lg   = self.last_graph
        inst = self.inst
        T    = inst.T; tc = inst.t_charge
        mm   = self.machines

        # xi / yi linking
        for (i, j, p) in lg.item_arcs:
            m.addConstr(xi[i, j, p] >= yi[i, j, p],          name=f"lb_{i}_{j}_{p}")
            m.addConstr(xi[i, j, p] <= inst.item_types[p] * yi[i, j, p],
                        name=f"ub_{i}_{j}_{p}")

        # Single-machine Cmax: the one machine runs all non-last bins then the last bin.
        # z_nonlast = z_expr (reflect arcs) + full_bin_count.
        # Cmax >= z_nonlast*(T+tc) + j  for every used last-batch arc ending at j.
        for (i, j, p) in lg.item_arcs:
            m.addConstr(
                (z_expr + inst.full_bin_count) * (T + tc) + j * yi[i, j, p] <= Cmax,
                name=f"cmax_arc_{i}_{j}_{p}")

        # Flow conservation in last-batch graph
        for v in lg.nodes:
            if v == 0 or v == T:
                continue
            in_f  = quicksum(xi[i, v, p] for (i, j, p) in lg.item_arcs if j == v)
            out_f = (quicksum(xi[v, j, p] for (i, j, p) in lg.item_arcs if i == v) +
                     xl[v])
            if in_f.size() + out_f.size() > 0:
                m.addConstr(in_f == out_f, name=f"lb_flow_{v}")

        # Source inflow <= m  (machines that end exactly on a bin boundary skip last batch)
        source_out = (quicksum(xi[0, j, p] for (i, j, p) in lg.item_arcs if i == 0) +
                      xl[0])
        m.addConstr(source_out <= mm, name="lb_source_inflow")

        # Sink: all m units must arrive at T
        in_T = (quicksum(xi[i, T, p] for (i, j, p) in lg.item_arcs if j == T) +
                quicksum(xl[v] for v in lg.loss_nodes))
        m.addConstr(in_T == mm, name="lb_sink")

    def _add_demand_constraints(self, m: gp.Model, xi_s, xi_r, xi, z_expr):
        """
        Demand: reflect_flow[p] + lastbatch_flow[p] == d[p]  for all p.

        xi is None during phase-1 (last-batch graph not yet built).
        In that case demand is enforced purely via reflect flow (temporarily
        relaxed — demand is re-added in phase 2).
        """
        inst  = self.inst; graph = self.graph
        s_set = set(graph.S_arcs)
        lg    = self.last_graph

        for t, d in inst.item_types.items():
            reflect_flow = quicksum(
                xi_s[arc] if arc in s_set else xi_r[arc]
                for arc in graph.Aj.get(t, []))
            if xi is None:
                # Phase 1: no last-batch graph yet; demand is unconstrained
                # from below — we only enforce the reflect capacity / flow
                # structure. Full demand rebalance happens in phase 2.
                m.addConstr(reflect_flow <= d, name=f"dem_{t}")
            else:
                lb_flow = quicksum(xi[i, j, p]
                                   for (i, j, p) in lg.At.get(t, []))
                m.addConstr(reflect_flow + lb_flow == d, name=f"dem_{t}")

    def _add_z_lower_bound(self, m: gp.Model, z_expr):
        inst       = self.inst
        total_load = sum(inst.jobs)
        full_load  = inst.full_bin_count * inst.T
        partial    = total_load - full_load
        T          = inst.T
        z_lb_load  = math.ceil(partial / T) - self.machines
        large  = [p for p in inst.jobs if T / 2 < p < T]
        small  = [p for p in inst.jobs if 0 < p <= T / 2]
        slack  = len(large) * T - sum(large)
        ovf    = max(0, sum(small) - slack)
        z_lb_mt = len(large) + math.ceil(ovf / T) - self.machines
        z_lb    = max(0, z_lb_load, z_lb_mt)
        print(f"  z_lb={z_lb}")
        m.addConstr(z_expr >= z_lb, name="z_lb")

    # ── parallel-machine non-last bin splitting ───────────────────────────────

    def _add_machine_split_vars(self, m: gp.Model):
        """u[t,k] + ub[k]: split non-last bins across machines."""
        inst = self.inst; M = range(self.machines)
        u = {(t, k): m.addVar(vtype=GRB.INTEGER, lb=0,
                               ub=inst.item_types[t], name=f"u_{t}_{k}")
             for t in inst.item_types for k in M}
        for k in M:
            u[-1, k] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"u_{-1}_{k}")
        ub = {k: m.addVar(vtype=GRB.INTEGER, lb=0,
                          ub=inst.full_bin_count, name=f"ub_{k}")
              for k in M}
        return u, ub

    def _add_machine_split_constraints(self, m: gp.Model, xi_s, xi_r,
                                       u, ub, xi, yi, xl, Cmax, z_expr):
        """
        Per-machine bin counts linked to reflect flow + Cmax via u/ub,
        plus the last-batch graph constraints.

        For m>1 the Cmax constraint tightens to per-machine:
          (bins_k) * (T+tc) + j*yi[i,j,p] <= Cmax
        where bins_k = sum_t u[t,k] + u[-1,k] + ub[k].
        """
        inst  = self.inst; graph = self.graph; M = range(self.machines)
        T     = inst.T; tc = inst.t_charge

        # u linkage to reflect arcs
        for t in inst.item_types:
            z_k = quicksum(xi_r[d, e, t1]
                           for (d, e, t1) in graph.R_arcs if t1 == t)
            m.addConstr(quicksum(u[t, k] for k in M) == z_k,
                        name=f"u_link_{t}")
        z_loss = quicksum(xi_r[d, e, t1]
                          for (d, e, t1) in graph.R_arcs if t1 is None)
        m.addConstr(quicksum(u[-1, k] for k in M) == z_loss,
                    name="u_link_loss")
        m.addConstr(quicksum(ub[k] for k in M) == inst.full_bin_count,
                    name="ub_split")

        # Last-batch graph constraints (shared yi, no k index)
        lg = self.last_graph
        for (i, j, p) in lg.item_arcs:
            m.addConstr(xi[i, j, p] >= yi[i, j, p])
            m.addConstr(xi[i, j, p] <= inst.item_types[p] * yi[i, j, p])

        # Per-machine Cmax
        for k in M:
            bins_k = (quicksum(u[t, k] for t in inst.item_types) +
                      u[-1, k] + ub[k])
            for (i, j, p) in lg.item_arcs:
                m.addConstr(
                    bins_k * (T + tc) + j * yi[i, j, p] <= Cmax,
                    name=f"cmax_arc_{k}_{i}_{j}_{p}")

        # Flow conservation in last-batch graph
        for v in lg.nodes:
            if v == 0 or v == T:
                continue
            in_f  = quicksum(xi[i, v, p] for (i, j, p) in lg.item_arcs if j == v)
            out_f = (quicksum(xi[v, j, p] for (i, j, p) in lg.item_arcs if i == v) +
                     xl[v])
            if in_f.size() + out_f.size() > 0:
                m.addConstr(in_f == out_f, name=f"lb_flow_{v}")

        source_out = (quicksum(xi[0, j, p] for (i, j, p) in lg.item_arcs if i == 0) +
                      xl[0])
        m.addConstr(source_out <= self.machines, name="lb_source_inflow")

        in_T = (quicksum(xi[i, T, p] for (i, j, p) in lg.item_arcs if j == T) +
                quicksum(xl[v] for v in lg.loss_nodes))
        m.addConstr(in_T == self.machines, name="lb_sink")

        # Global Cmax lower bound from load
        cmax_lb = math.ceil(sum(inst.jobs) / self.machines)
        m.addConstr(Cmax >= cmax_lb, name="cmax_lb_load")

    # ── public API ────────────────────────────────────────────────────────────

    def build_model(self) -> gp.Model:
        if self.two_phase:
            return self._build_model_two_phase()
        return self._build_model_single_phase()

    # ── single-phase model ────────────────────────────────────────────────────

    def _build_model_single_phase(self) -> gp.Model:
        inst = self.inst; T = inst.T; tc = inst.t_charge
        M    = range(self.machines)

        m = gp.Model("SMSP_ArcFlowReflect_LB")
        m.Params.TimeLimit = self.time_limit
        m.Params.Threads   = self.threads
        m.Params.MIPGap    = 1e-6
        m.setParam("OutputFlag", 1 if self.verbose else 0)

        xi_s, xi_r = self._add_arc_flow_vars(m)
        xi, yi, xl = self._add_last_batch_vars(m)

        if self.machines > 1:
            u, ub  = self._add_machine_split_vars(m)
            Cmax   = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Cmax")
        else:
            u = ub = None
            Cmax = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Cmax")
        m.update()

        self._add_flow_conservation(m, xi_s, xi_r)
        z_expr = self._add_source_outflow(m, xi_s, xi_r)
        self._add_demand_constraints(m, xi_s, xi_r, xi, z_expr)
        self._add_z_lower_bound(m, z_expr)

        if self.machines > 1:
            self._add_machine_split_constraints(
                m, xi_s, xi_r, u, ub, xi, yi, xl, Cmax, z_expr)
        else:
            self._add_last_batch_constraints(m, xi, yi, xl, Cmax, z_expr)

        m.setObjective(Cmax, GRB.MINIMIZE)
        m.Params.LogFile      = "gurobi_run3.log"
        m.Params.LogToConsole = 1
        m._root_obj           = None

        self._model = m; self._xi_s = xi_s; self._xi_r = xi_r
        self._xi = xi; self._yi = yi; self._xl = xl
        self._u = u; self._ub = ub
        return m

    # ── two-phase model ───────────────────────────────────────────────────────

    def _build_model_two_phase(self) -> gp.Model:
        """
        Phase 1: minimise z_nonlast (reflect graph only, no last-batch vars).
                 Demand uses reflect_flow <= d[t] (upper bound only).
        Phase 2: fix z_nonlast, add last-batch graph, minimise Cmax.
        """
        inst = self.inst; T = inst.T; tc = inst.t_charge
        M    = range(self.machines)

        m = gp.Model("SMSP_ArcFlowReflect_LB_2phase")
        m.Params.TimeLimit = self.time_limit
        m.Params.Threads   = self.threads
        m.Params.MIPGap    = 1e-6
        m.setParam("OutputFlag", 1 if self.verbose else 0)

        xi_s, xi_r = self._add_arc_flow_vars(m)
        self._add_flow_conservation(m, xi_s, xi_r)
        z_expr = self._add_source_outflow(m, xi_s, xi_r)

        # Phase 1: demand as upper bound (no last-batch graph yet)
        self._add_demand_constraints(m, xi_s, xi_r, None, z_expr)
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

        # Fix z at optimum (with slack for m>1: each machine needs at least 1 last bin)
        z_arc = z_opt - inst.full_bin_count
        # Remove phase-1 demand upper-bound constraints
        for t in inst.item_types:
            c = m.getConstrByName(f"dem_{t}")
            if c is not None:
                m.remove(c)

        if self.machines > 1:
            # Allow up to m fewer bins from reflect (machines may share last bins)
            m.addConstr(z_expr >= max(0, z_arc - self.machines), name="fix_z")
        else:
            m.addConstr(z_expr == z_arc, name="fix_z")

        # Add last-batch graph variables and full demand constraints
        xi, yi, xl = self._add_last_batch_vars(m)

        if self.machines > 1:
            u, ub  = self._add_machine_split_vars(m)
            Cmax   = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Cmax")
        else:
            u = ub = None
            Cmax = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Cmax")
        m.update()

        self._add_demand_constraints(m, xi_s, xi_r, xi, z_expr)

        if self.machines > 1:
            self._add_machine_split_constraints(
                m, xi_s, xi_r, u, ub, xi, yi, xl, Cmax, z_expr)
        else:
            self._add_last_batch_constraints(m, xi, yi, xl, Cmax, z_expr)

        m.setObjective(Cmax, GRB.MINIMIZE)
        m._root_bound      = None; m._root_obj = None
        m._phase1_time     = phase1_time
        m._phase1_root_gap = phase1_root_gap
        m._phase2_ready    = True

        self._model = m; self._xi_s = xi_s; self._xi_r = xi_r
        self._xi = xi; self._yi = yi; self._xl = xl
        self._u = u; self._ub = ub
        return m

    # ── solve ─────────────────────────────────────────────────────────────────

    def solve(self) -> SolverResult:
        if self._model is None:
            self.build_model()
        return self._solve_two_phase() if self.two_phase else self._solve_single_phase()

    def _solve_single_phase(self) -> SolverResult:
        m = self._model; t0 = time.time()
        m.optimize()
        m.Params.LogFile = ""
        root_gap, _, _ = _get_root_gap_from_log("gurobi_run3.log")
        try:
            os.remove("gurobi_run3.log")
        except OSError:
            pass
        rt = time.time() - t0; st = m.Status

        if m.SolCount == 0:
            if st == GRB.INFEASIBLE:
                m.computeIIS(); m.write("infeasible_model.ilp")
            return SolverResult(
                status="infeasible" if st == GRB.INFEASIBLE else "timeout",
                cmax=None, z_nonlast=None, last_load=None,
                lb=None, gap=None, runtime=rt, root_gap=None,
                runtime_phase1=None, phase1_root_gap=None, phase2_root_gap=None,
                instance=self.inst)

        z_val     = (round(sum(v.X for v in self._xi_r.values()))
                     + self.inst.full_bin_count)
        last_load = self._compute_last_load()
        cmax      = m.ObjVal
        status    = "optimal" if st == GRB.OPTIMAL else "feasible"

        return self._build_result(
            status=status, cmax=cmax, z_val=z_val, last_load=last_load,
            lb=m.ObjBound, gap=m.MIPGap, runtime=rt,
            root_gap=root_gap,
            runtime_phase1=None, phase1_root_gap=None, phase2_root_gap=None)

    def _solve_two_phase(self) -> SolverResult:
        m  = self._model; t0 = time.time()

        if not getattr(m, "_phase2_ready", False):
            rt = time.time() - t0 + m._phase1_time
            st = m.Status
            return SolverResult(
                status="infeasible" if st == GRB.INFEASIBLE else "timeout",
                cmax=None, z_nonlast=None, last_load=None,
                lb=None, gap=None, runtime=rt, root_gap=None,
                runtime_phase1=m._phase1_time,
                phase1_root_gap=m._phase1_root_gap,
                phase2_root_gap=None,
                instance=self.inst)

        m.optimize(_phase_callback)
        st = m.Status

        phase2_root_gap: Optional[float]
        if m._root_bound is not None and m.SolCount > 0 and m.ObjVal != 0:
            phase2_root_gap = abs(m._root_bound - m.ObjVal) / abs(m.ObjVal)
        else:
            phase2_root_gap = 0.0

        rt = time.time() - t0 + m._phase1_time

        if m.SolCount == 0:
            return SolverResult(
                status="infeasible" if st == GRB.INFEASIBLE else "timeout",
                cmax=None, z_nonlast=None, last_load=None,
                lb=None, gap=None, runtime=rt, root_gap=None,
                runtime_phase1=m._phase1_time,
                phase1_root_gap=m._phase1_root_gap,
                phase2_root_gap=phase2_root_gap,
                instance=self.inst)

        z_val     = (round(sum(v.X for v in self._xi_r.values()))
                     + self.inst.full_bin_count)
        last_load = self._compute_last_load()
        cmax      = m.ObjVal
        status    = "optimal" if st == GRB.OPTIMAL else "feasible"

        return self._build_result(
            status=status, cmax=cmax, z_val=z_val, last_load=last_load,
            lb=m.ObjBound, gap=m.MIPGap, runtime=rt,
            root_gap=None,
            runtime_phase1=m._phase1_time,
            phase1_root_gap=m._phase1_root_gap,
            phase2_root_gap=phase2_root_gap)

    def _compute_last_load(self) -> float:
        vals = [j * v.X for (i, j, p), v in self._yi.items() if v.X > 0.5]
        return max(vals) if vals else 0.0

    # ── result builder ────────────────────────────────────────────────────────

    def _build_result(self, *, status, cmax, z_val, last_load,
                      lb, gap, runtime,
                      root_gap, runtime_phase1, phase1_root_gap,
                      phase2_root_gap) -> SolverResult:
        inst = self.inst; lg = self.last_graph

        try:
            for v in self._model.getVars():
                if v.X > 1e-5:
                    print(f"{v.varName} = {v.X:.4g}")
        except Exception:
            pass

        # Reconstruct non-last bins from reflect graph
        xs_val      = {k: v.X for k, v in self._xi_s.items() if v.X > 0.5}
        xr_val      = {k: v.X for k, v in self._xi_r.items() if v.X > 0.5}
        bins_ptimes = _reconstruct_bins(xs_val, xr_val, self.graph)
        for t, count in inst.full_bin_items.items():
            for _ in range(count):
                bins_ptimes.append([t])

        # Reconstruct last-bin contents from yi arcs
        last_ptimes: List[int] = []
        for (i, j, p), v in self._yi.items():
            if v.X > 0.5:
                count = round(self._xi[i, j, p].X)
                last_ptimes.extend([p] * count)
        last_ptimes.sort()

        # Map p-time lists to job indices
        full_pool: Dict[int, List[int]] = defaultdict(list)
        for idx, p in enumerate(inst.jobs):
            full_pool[p].append(idx)

        bins_idx = [sorted(full_pool[p].pop(0) for p in pt) for pt in bins_ptimes]
        if bins_idx:
            bins_idx, bins_ptimes = zip(
                *sorted(zip(bins_idx, bins_ptimes), key=lambda x: x[0][0]))
            bins_idx    = list(bins_idx)
            bins_ptimes = list(bins_ptimes)

        last_idx = sorted(full_pool[p].pop(0) for p in last_ptimes)

        # Per-machine assignment (parallel machines)
        machine_assignments = None
        if self.machines > 1 and self._u is not None:
            bin_iter = iter(range(len(bins_ptimes)))
            machine_assignments = {}
            for k in range(self.machines):
                n_bins_k = (sum(round(self._u[t, k].X) for t in inst.item_types)
                            + round(self._u[-1, k].X)
                            + round(self._ub[k].X))
                k_bin_indices = [next(bin_iter) for _ in range(n_bins_k)]
                k_bins_ptimes = [bins_ptimes[i] for i in k_bin_indices]
                k_bins_idx    = [bins_idx[i]    for i in k_bin_indices]
                # Last bin items from yi (shared pool — take from remaining full_pool)
                last_k_ptimes = sorted(last_ptimes)   # shared last bin
                last_k_idx    = []
                machine_assignments[k] = {
                    "bins_ptimes":  k_bins_ptimes,
                    "bins_indices": k_bins_idx,
                    "last_ptimes":  last_k_ptimes,
                    "last_indices": last_k_idx,
                }

        return SolverResult(
            status=status, cmax=cmax, z_nonlast=int(z_val),
            last_load=last_load, lb=lb, gap=gap, runtime=runtime,
            root_gap=root_gap,
            runtime_phase1=runtime_phase1,
            phase1_root_gap=phase1_root_gap,
            phase2_root_gap=phase2_root_gap,
            instance=inst,
            bins_indices=bins_idx, last_indices=last_idx,
            bins_ptimes=bins_ptimes, last_ptimes=last_ptimes,
            machine_assignments=machine_assignments)


# ─────────────────────────────────────────────
# Solution file
# ─────────────────────────────────────────────

def _format_solution(res: SolverResult) -> str:
    if res.cmax is None:
        return f"Status: {res.status}\nNo feasible solution found.\n"
    lines = []
    lines.append(f"The num of Batches: {res.z_nonlast}")
    lines.append(f"makespan: {res.cmax}")
    if res.machine_assignments is None:
        lines.append("The jobs in each Batch")
        for b, (idx, pt) in enumerate(zip(res.bins_indices, res.bins_ptimes)):
            lines.append(f"Batch{b}:index:{idx} Processing Time: {sum(pt)}")
        lines.append("The last Batch")
        lines.append(f"index:{res.last_indices} "
                     f"Processing Time: {sum(res.last_ptimes or [])}")
    else:
        lines.append("The jobs in each Batch")
        for b, (idx, pt) in enumerate(zip(res.bins_indices, res.bins_ptimes)):
            lines.append(f"Batch{b}:index:{idx} Processing Time: {sum(pt)}")
        lines.append("--- Per-Machine Assignment ---")
        for k, asgn in res.machine_assignments.items():
            n_bins = len(asgn["bins_ptimes"])
            lines.append(f"\nMachine {k}  ({n_bins} non-last bin(s)):")
            for b, (idx, pt) in enumerate(zip(asgn["bins_indices"], asgn["bins_ptimes"])):
                lines.append(f"  Batch{b}:index:{idx} Processing Time: {sum(pt)}")
            lines.append(f"  Last bin: index:{asgn['last_indices']}  "
                         f"Processing Time: {sum(asgn['last_ptimes'])}")
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
            "cmax":            f"{res.cmax:.1f}"       if res.cmax      is not None else "",
            "lb":              f"{res.lb:.2f}"         if res.lb        is not None else "",
            "gap_pct":         f"{res.gap * 100:.4f}"  if res.gap       is not None else "",
            "z_nonlast":       res.z_nonlast           if res.z_nonlast is not None else "",
            "last_load":       f"{res.last_load:.1f}"  if res.last_load is not None else "",
            "runtime_s":       f"{res.runtime:.3f}",
            "runtime_phase1":  (f"{res.runtime_phase1:.3f}"
                                if res.runtime_phase1 is not None else ""),
            "root_gap":        (f"{root_gap_col:.8f}"
                                if root_gap_col is not None else ""),
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
               time_limit=3600.0, verbose=False, threads=1,
               machines=1, two_phase=False):
    if sol_dir:
        os.makedirs(sol_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(folder)
                   if os.path.isfile(os.path.join(folder, f)))
    if not files:
        print(f"No files found in {folder}"); return
    print(f"Instances : {len(files)}  |  CSV: {csv_path}"
          + (f"  |  Sol: {sol_dir}" if sol_dir else ""))
    print(f"Mode      : {'2-phase' if two_phase else 'single-phase'}")
    print("-" * 72)
    set_name = os.path.basename(os.path.normpath(folder))

    for i, fname in enumerate(files, 1):
        fpath = os.path.join(folder, fname)
        print(f"[{i:4d}/{len(files)}] {fname:<32s}", end=" ", flush=True)
        try:
            inst   = SMInstance.from_file(fpath, t_charge=t_charge)
            solver = ArcFlowReflectSMSP(inst, time_limit=time_limit,
                                         verbose=verbose, threads=threads,
                                         machines=machines, two_phase=two_phase)
            solver.build_model()
            res = solver.solve()
            if res.cmax is not None:
                gap_s = f"{res.gap * 100:.2f}%" if res.gap is not None else "?"
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
               time_limit=3600.0, verbose=True, threads=1,
               machines=1, two_phase=False):
    fname  = os.path.basename(fpath)
    inst   = SMInstance.from_file(fpath, t_charge=t_charge)
    print(f"Instance : {fname}\n  {inst.summary()}")
    lg = LastBatchGraph(inst)
    print(f"Reflect  : {ReflectGraph(inst).summary()}")
    print(f"LastBatch: {lg.summary()}")
    print(f"Mode     : {'2-phase' if two_phase else 'single-phase'}")

    solver = ArcFlowReflectSMSP(inst, time_limit=time_limit,
                                  verbose=verbose, threads=threads,
                                  machines=machines, two_phase=two_phase)
    solver.build_model()
    res = solver.solve()

    set_name = os.path.basename(os.path.dirname(os.path.abspath(fpath)))
    print(f"\nStatus   : {res.status}")
    if res.cmax is not None:
        print(f"Cmax     : {res.cmax}")
        if res.lb  is not None: print(f"LB       : {res.lb:.2f}")
        if res.gap is not None: print(f"Gap      : {res.gap * 100:.4f}%")
        print(f"z (non-last bins) : {res.z_nonlast}")
        print(f"Last bin load     : {res.last_load:.1f}")
        if res.runtime_phase1 is not None:
            print(f"Phase-1 runtime  : {res.runtime_phase1:.2f}s")
            print(f"Phase-1 root gap : {res.phase1_root_gap}")
            print(f"Phase-2 root gap : {res.phase2_root_gap}")
        else:
            print(f"Root gap         : {res.root_gap}")
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
# Example invocations (uncomment one block)
# ─────────────────────────────────────────────

#"""
# --- Single instance run ---
setstr = "MOD"
file   = "L_00000247"
result = run_single(
    f"Benchmark Instances/Instances/{setstr}/{file}",
    t_charge   = 0,
    time_limit = 720,
    machines   = 2,
    two_phase  = True,
)
print(result.cmax, result.z_nonlast, result.last_load)
#"""

"""
# --- Folder run, single-phase, m machines ---
folder = "MOD"
run_folder(
    folder     = f"Benchmark Instances/Instances/{folder}/",
    csv_path   = f"results/{folder}_results_lastbatch_hybrid_m5.csv",
    sol_dir    = f"results/{folder}_solutions_lastbatch_hybrid_m5",
    t_charge   = 0,
    time_limit = 720.0,
    threads    = 1,
    verbose    = True,
    machines   = 5,
    two_phase  = True,
)
#"""

"""
# --- Folder run, 2-phase, single machine ---
folder = "LOW"
run_folder(
    folder     = f"Benchmark Instances/Instances/{folder}/",
    csv_path   = f"results/{folder}_results_lastbatch_2phase.csv",
    t_charge   = 0,
    time_limit = 3600.0,
    threads    = 1,
    verbose    = True,
    machines   = 1,
    two_phase  = True,
)
"""

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
"""
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Arc-Flow Reflect + Last-Batch solver for P_m|pm|Cmax")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--instance", help="Single instance file")
    grp.add_argument("--folder",   help="Folder of instances (batch)")
    p.add_argument("--out_csv",    default="results.csv")
    p.add_argument("--out_sol",    default=None)
    p.add_argument("--t_charge",   type=int,   default=0)
    p.add_argument("--time_limit", type=float, default=3600.0)
    p.add_argument("--threads",    type=int,   default=1)
    p.add_argument("--machines",   type=int,   default=1)
    p.add_argument("--two_phase",  action="store_true")
    p.add_argument("--verbose",    action="store_true")
    args = p.parse_args()
    kw = dict(t_charge=args.t_charge, time_limit=args.time_limit,
              verbose=args.verbose, threads=args.threads,
              machines=args.machines, two_phase=args.two_phase)
    if args.instance:
        run_single(args.instance, csv_path=args.out_csv,
                   sol_dir=args.out_sol, **kw)
    else:
        run_folder(args.folder, csv_path=args.out_csv,
                   sol_dir=args.out_sol, **kw)
"""