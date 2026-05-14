# -*- coding: utf-8 -*-
"""
Arc-Flow Reflect model for P_m|pm|Cmax  (single-machine: m=1)
(Parallel-Machine Scheduling with Periodic Maintenance, minimize makespan).

Model
-----
  Non-last bins : FRE reflect arc-flow  (Delorme & Iori 2020)
  Last bin      : continuous y[t] / integer y[t,k] variables per distinct p-time

  Demand split     : flow_t + y[t] = d[t]                for all t
  Last bin cap     : sum_t  t * y[t] <= T
  Objective (1-ph) : min  z*(T + t_charge) + sum_t t*y[t]   =  Cmax
  Objective (2-ph) : phase 1 → min z;  phase 2 → fix z, min sum_t t*y[t]

  Note: two_phase=True is only supported for the single-machine case (machines=1).
        For machines > 1 the solver always uses the single-phase Cmax formulation.

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
import warnings
import argparse, csv, math, os, re, time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    item_types: Dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        c: Dict[int, int] = {}
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
# Bin reconstruction from arc-flow solution
# ─────────────────────────────────────────────

def _reconstruct_bins(xi_s_val, xi_r_val, graph):
    """Return list-of-lists of processing times per non-last bin."""
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


def _assign_indices(inst, bins_ptimes, last_ptimes):
    """Map processing-time lists back to original 0-based job indices."""
    pool: Dict[int, List[int]] = defaultdict(list)
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
    status:              str
    cmax:                Optional[float]
    z_nonlast:           Optional[int]
    last_load:           Optional[float]
    lb:                  Optional[float]
    gap:                 Optional[float]
    runtime:             float
    root_gap:            Optional[float]   # single-phase root gap (None for 2-phase)
    runtime_phase1:      Optional[float]   # 2-phase only (None for single-phase)
    phase1_root_gap:     Optional[float]   # 2-phase only
    phase2_root_gap:     Optional[float]   # 2-phase only
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
    """Parse a Gurobi log file and return (gap, root_bound, root_incumbent)."""
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
    """Gurobi callback that records root-node bounds for the 2-phase approach."""
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
    Arc-flow reflect solver for P_m|pm|Cmax.

    Parameters
    ----------
    two_phase : bool
        If True, use the 2-phase approach:
          phase 1 — minimise total z (machine-agnostic bin count);
          phase 2 — fix z, then minimise Cmax by optimally distributing
                    bins and last-bin load across machines.
        Valid for both single and parallel machines: phase 1 solves a
        pure bin-packing problem with no machine structure, so z* is the
        same regardless of m.  Phase 2 retains full freedom to assign
        bins to machines.
    """

    def __init__(self, inst: SMInstance, time_limit: float = 3600.0,
             verbose: bool = False, threads: int = 1,
             machines: int = 1, two_phase: bool = False):
        self.inst       = inst
        self.time_limit = time_limit
        self.verbose    = verbose
        self.threads    = threads
        self.machines   = machines
        """
        if two_phase and machines > 1:
            warnings.warn(
                "two_phase=True is only valid for machines=1. "
                "Falling back to single-phase for parallel machines.",
                UserWarning, stacklevel=2)
            two_phase = False"""
        self.two_phase  = two_phase
        self.graph      = ReflectGraph(inst)
        self.model_name = "arcflow_reflect"
        if two_phase:
            self.model_name +="_two_phase"
        self._xi_s      = {}; self._xi_r = {}; self._y = {}
        self._u         = None; self._ub = None


    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _add_arc_flow_vars(self, m: gp.Model):
        """Create xi_s / xi_r arc variables and return them."""
        graph = self.graph
        xi_s = {(d, e, t): m.addVar(vtype=GRB.INTEGER, lb=0, name=f"xs_{d}_{e}_{t}")
                for (d, e, t) in graph.S_arcs}
        xi_r = {(d, e, t): m.addVar(vtype=GRB.INTEGER, lb=0, name=f"xr_{d}_{e}_{t}")
                for (d, e, t) in graph.R_arcs}
        return xi_s, xi_r

    def _add_flow_conservation(self, m: gp.Model, xi_s, xi_r):
        """Constraints (17): flow conservation at every non-source node."""
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
        """Constraint (18): source outflow = 2*z."""
        graph = self.graph
        z_expr = quicksum(xi_r[d, e, t] for (d, e, t) in graph.R_arcs)
        out_0  = (quicksum(xi_s[0, e, t] for (d, e, t) in graph.S_arcs if d == 0) +
                  quicksum(xi_r[0, e, t] for (d, e, t) in graph.R_arcs if d == 0))
        m.addConstr(out_0 == 2 * z_expr, name="source_outflow")
        return z_expr

    def _add_demand_constraints(self, m: gp.Model, xi_s, xi_r, y, z_expr,
                                per_machine: bool = False):
        """Demand-split constraints (D): flow_t + sum_k y[t,k] = d[t].

        per_machine=True  → y is indexed (t, k); use for phase-2 parallel model.
        per_machine=False → y is indexed t only; use for single-machine or phase-1.
        """
        inst  = self.inst; graph = self.graph
        s_set = set(graph.S_arcs)
        M     = range(self.machines)
        for t, arcs in graph.Aj.items():
            flow_t = quicksum(
                xi_s[d, e, tt] if (d, e, tt) in s_set else xi_r[d, e, tt]
                for (d, e, tt) in arcs)
            if per_machine:
                m.addConstr(flow_t + quicksum(y[t, k] for k in M) == inst.item_types[t],
                            name=f"dem_{t}")
            else:
                if y is None:
                    m.addConstr(flow_t == inst.item_types[t], name=f"dem_{t}")
                else:
                    m.addConstr(flow_t + y[t] == inst.item_types[t], name=f"dem_{t}")


    def _add_z_lower_bound(self, m: gp.Model, z_expr):
        """Optional lower bound on z derived from total load."""
        inst  = self.inst
        total_load    = sum(inst.jobs)
        full_bin_load = inst.full_bin_count * inst.T
        partial_load  = total_load - full_bin_load
        T = inst.T
        z_lb_load = math.ceil(partial_load / T) -self.machines
        large = [p for p in inst.jobs if T / 2 < p < T]
        small = [p for p in inst.jobs if 0 < p <= T / 2]
        slack = len(large) * T - sum(large)             # residual room in large-item bins
        overflow = max(0, sum(small) - slack)           # small load that can't fit there
        z_lb_mt = len(large) + math.ceil(overflow / T) - self.machines
        z_lb = max(0,z_lb_load,z_lb_mt)
        m.addConstr(z_expr >= z_lb, name="z_lb")

    # ------------------------------------------------------------------
    # Public build_model
    # ------------------------------------------------------------------

    def build_model(self) -> gp.Model:
        if self.two_phase:
            return self._build_model_two_phase()
        else:
            return self._build_model_single_phase()

    # ------------------------------------------------------------------
    # Single-phase model  (supports machines >= 1)
    # ------------------------------------------------------------------

    def _build_model_single_phase(self) -> gp.Model:
        inst = self.inst; graph = self.graph; T = inst.T; tc = inst.t_charge
        M    = range(self.machines)

        m = gp.Model("SMSP_ArcFlowReflect")
        m.Params.TimeLimit  = self.time_limit
        m.Params.Threads    = self.threads
        m.Params.MIPGap     = 1e-6
        m.setParam("OutputFlag", 1 if self.verbose else 0)

        xi_s, xi_r = self._add_arc_flow_vars(m)

        if self.machines > 1:
            y  = {(t, k): m.addVar(vtype=GRB.INTEGER, lb=0, ub=d, name=f"y_{t}_{k}")
                  for t, d in inst.item_types.items() for k in M if k>0}
            for t, d in inst.item_types.items():
                y[t,0] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=d, name=f"y_{t}_0")
            u  = {(t, k): m.addVar(vtype=GRB.INTEGER, lb=0, ub=d, name=f"u_{t}_{k}")
                  for t, d in inst.item_types.items() for k in M}
            for k in M:
                u[-1, k] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"u_{-1}_{k}")
            ub = {k: m.addVar(vtype=GRB.INTEGER, lb=0, ub=inst.full_bin_count, name=f"ub_{k}")
                  for k in M}
            Cmax = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Cmax")
        else:
            y  = {t: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=d, name=f"y_{t}")
                  for t, d in inst.item_types.items()}
            u = ub = None
        m.update()

        self._add_flow_conservation(m, xi_s, xi_r)
        z_expr = self._add_source_outflow(m, xi_s, xi_r)
        self._add_demand_constraints(m, xi_s, xi_r, y, z_expr,
                                     per_machine=self.machines > 1)
        self._add_z_lower_bound(m, z_expr)

        if self.machines > 1:
            # u linkage: sum_k u[t,k] = reflect flow for item type t
            for t in inst.item_types:
                z_k = quicksum(xi_r[d, e, t1] for (d, e, t1) in graph.R_arcs if t1 == t)
                m.addConstr(quicksum(u[t, k] for k in M) == z_k, name=f"u_link_{t}")

            z_loss = quicksum(xi_r[d, e, t1] for (d, e, t1) in graph.R_arcs if t1 is None)
            m.addConstr(quicksum(u[-1, k] for k in M) == z_loss, name="u_link_loss")

            m.addConstr(quicksum(ub[k] for k in M) == inst.full_bin_count, name="ub_split")

            for k in M:
                m.addConstr(quicksum(t * y[t, k] for t in inst.item_types) <= T,
                            name=f"last_cap_{k}")
                total_bins_k = (quicksum(u[t, k] for t in inst.item_types)
                                + u[-1, k] + ub[k])
                m.addConstr(
                    total_bins_k * (T + tc)
                    + quicksum(t * y[t, k] for t in inst.item_types) <= Cmax,
                    name=f"cmax_{k}")
                # Symmetry-breaking
                
                if k > 1:
                    for k in M[1:]:
                        total_bins_k = (quicksum(u[t, k] for t in inst.item_types)
                                        + u[-1, k] + ub[k]+ quicksum(t * y[t, k] for t in inst.item_types))
                        total_bins_k_min_1 = (quicksum(u[t,k-1] for t in inst.item_types)
                                        + u[-1, k-1] + ub[k-1]+ quicksum(t * y[t, k-1] for t in inst.item_types))
                        m.addConstr(
                        total_bins_k  <= total_bins_k_min_1) 
                        """
                        total_bins_k = (quicksum(u[t, k] for t in inst.item_types)
                                        + u[-1, k] + ub[k])
                        total_bins_k_min_1 = (quicksum(u[t,k-1] for t in inst.item_types)
                                        + u[-1, k-1] + ub[k-1])
                        m.addConstr(
                        total_bins_k * (T + tc)
                        + quicksum(t * y[t, k] for t in inst.item_types) <= total_bins_k_min_1 * (T + tc)
                        + quicksum(t * y[t, k-1] for t in inst.item_types)
                        )"""
                """
                if k > 0:
                    for k in M[1:]:
                        m.addConstr(
                            quicksum(y[t, k] for t in inst.item_types)<= quicksum(y[t1, k - 1] for t1 in inst.item_types),
                                name=f"sym_{t}_{k}")#"""

            cmax_lb = math.ceil(sum(inst.jobs) / self.machines)
            m.addConstr(Cmax >= cmax_lb, name="cmax_lb_load")
            m.setObjective(Cmax, GRB.MINIMIZE)
        else:
            m.addConstr(quicksum(t * y[t] for t in inst.item_types) <= T, name="last_cap")
            m.setObjective(
                (z_expr + inst.full_bin_count) * (T + tc)
                + quicksum(t * y[t] for t in inst.item_types),
                GRB.MINIMIZE)

        # Log file for root-gap extraction
        m.Params.LogFile       = "gurobi_run3.log"
        m.Params.LogToConsole  = 1
        m._root_obj            = None

        self._model = m; self._xi_s = xi_s; self._xi_r = xi_r; self._y = y
        self._u = u; self._ub = ub
        return m

    # ------------------------------------------------------------------
    # Two-phase model  (single machine only)
    # ------------------------------------------------------------------

    def  _build_model_two_phase(self) -> gp.Model:
        """
        Build the model and immediately run phase 1 (minimise total z).
        Phase 2 (fix z, minimise Cmax with full machine assignment freedom)
        is run in solve().

        Phase 1 is machine-agnostic: the arc-flow graph has no notion of
        machines, so z* is the global minimum bin count.  Phase 2 adds all
        parallel-machine variables and constraints and optimises Cmax directly.
        """
        inst = self.inst; graph = self.graph; T = inst.T; tc = inst.t_charge
        M    = range(self.machines)

        m = gp.Model("SMSP_ArcFlowReflect_2phase")
        m.Params.TimeLimit = self.time_limit
        m.Params.Threads   = self.threads
        m.Params.MIPGap    = 1e-6
        m.setParam("OutputFlag", 1 if self.verbose else 0)

        xi_s, xi_r = self._add_arc_flow_vars(m)

        # Phase 1 uses scalar y[t] (machine-agnostic) just to enforce demand
        # and last-bin capacity — they will be replaced in phase 2.
        self._add_flow_conservation(m, xi_s, xi_r)
        z_expr = self._add_source_outflow(m, xi_s, xi_r)
        if self.machines==1:
            y    = {t: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=d, name=f"y_{t}")
                    for t, d in inst.item_types.items()}
            m.addConstr(quicksum(t * y[t] for t in inst.item_types) <= T*self.machines,
                        name="last_cap_p1")
            self._add_demand_constraints(m, xi_s, xi_r, y, z_expr)
        else:
            self._add_demand_constraints(m, xi_s, xi_r, None, z_expr)
        m.update()

        
       

        # Phase 1: minimise total z
        m.setObjective(z_expr+ inst.full_bin_count, GRB.MINIMIZE)
        m._root_bound = None; m._root_obj = None
        self._add_z_lower_bound(m, z_expr)
        m.optimize(_phase_callback)

        phase1_time = m.Runtime
        if m.SolCount == 0:
            m._phase1_time     = phase1_time
            m._phase1_root_gap = None
            m._phase2_ready    = False
            self._model = m; self._xi_s = xi_s; self._xi_r = xi_r
            self._u = None; self._ub = None
            return m

        z_opt = round(m.ObjVal)
        phase1_root_gap = (abs(m._root_bound - z_opt) / abs(z_opt)
                           if m._root_bound is not None and z_opt != 0 else 0.0)

        # Fix z at its optimal value
        z_arc = z_opt - inst.full_bin_count
        for t in inst.item_types:
            m.remove(m.getConstrByName(f"dem_{t}"))
        if self.machines>1:
            z_arc =z_opt - inst.full_bin_count-self.machines  # the part z_expr must cover
            m.addConstr(z_expr >= z_arc, name="fix_z")
        else:
            m.addConstr(z_expr == z_arc, name="fix_z")
        if self.machines > 1:
            y  = {(t, k): m.addVar(vtype=GRB.INTEGER, lb=0, ub=d, name=f"y_{t}_{k}")
                  for t, d in inst.item_types.items() for k in M if k >0}
            for t, d in inst.item_types.items():
                y[t, 0] =  m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=d, name=f"y_{t}_0")
            u  = {(t, k): m.addVar(vtype=GRB.INTEGER, lb=0, ub=d, name=f"u_{t}_{k}")
                  for t, d in inst.item_types.items() for k in M}
            for k in M:
                u[-1, k] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"u_{-1}_{k}")
            ub = {k: m.addVar(vtype=GRB.INTEGER, lb=0, ub=inst.full_bin_count,
                              name=f"ub_{k}") for k in M}
            Cmax = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Cmax")
        else:
            y    = {t: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=d, name=f"y_{t}")
                    for t, d in inst.item_types.items()}
            u = ub = None
            
        # Re-add demand constraints in per-machine form
        self._add_demand_constraints(m, xi_s, xi_r, y, z_expr,
                                     per_machine=self.machines > 1)


        # Remove phase-1-only constraints and scalar y variables so we can
        # re-add them in per-machine form for phase 2


        m.update()
        if self.machines > 1:
            for t in inst.item_types:
                z_k = quicksum(xi_r[d, e, t1] for (d, e, t1) in graph.R_arcs if t1 == t)
                m.addConstr(quicksum(u[t, k] for k in M) == z_k, name=f"u_link_{t}")
            z_loss = quicksum(xi_r[d, e, t1] for (d, e, t1) in graph.R_arcs if t1 is None)
            m.addConstr(quicksum(u[-1, k] for k in M) == z_loss, name="u_link_loss")
            m.addConstr(quicksum(ub[k] for k in M) == inst.full_bin_count, name="ub_split")

            for k in M:
                m.addConstr(quicksum(t * y[t, k] for t in inst.item_types) <= T,
                            name=f"last_cap_{k}")
                total_bins_k = (quicksum(u[t, k] for t in inst.item_types)
                                + u[-1, k] + ub[k])
                m.addConstr(
                    total_bins_k * (T + tc)
                    + quicksum(t * y[t, k] for t in inst.item_types) <= Cmax,
                    name=f"cmax_{k}")
                if k > 1:
                    for k in M[1:]:
                        total_bins_k = (quicksum(u[t, k] for t in inst.item_types)
                                       + u[-1, k] + ub[k]+ quicksum(t * y[t, k] for t in inst.item_types))
                        total_bins_k_min_1 = (quicksum(u[t,k-1] for t in inst.item_types)
                                       + u[-1, k-1] + ub[k-1]+ quicksum(t * y[t, k-1] for t in inst.item_types))
                        m.addConstr(
                        total_bins_k  <= total_bins_k_min_1) 

            cmax_lb = math.ceil(sum(inst.jobs) / self.machines)
            m.addConstr(Cmax >= cmax_lb, name="cmax_lb_load")
            m.setObjective(Cmax, GRB.MINIMIZE)
            self._Cmax_var = Cmax
        else:
            m.addConstr(quicksum(t * y[t] for t in inst.item_types) <= T, name="last_cap")
            m.setObjective(quicksum(t * y[t] for t in inst.item_types), GRB.MINIMIZE)
            self._Cmax_var = None



        m._root_bound      = None; m._root_obj = None
        m._phase1_time     = phase1_time
        m._phase1_root_gap = phase1_root_gap
        m._phase2_ready    = True

        self._model = m; self._xi_s = xi_s; self._xi_r = xi_r; self._y = y
        self._u = u; self._ub = ub
        return m
    
    # ------------------------------------------------------------------
    # solve()
    # ------------------------------------------------------------------

    def solve(self) -> SolverResult:
        if self._model is None:
            self.build_model()
        if self.two_phase:
            return self._solve_two_phase()
        else:
            return self._solve_single_phase()

    def _solve_single_phase(self) -> SolverResult:
        m = self._model; t0 = time.time()
        m.optimize()
        m.Params.LogFile = ""          # release file handle
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

        T = self.inst.T; tc = self.inst.t_charge

        if self.machines > 1:
            z_val = (round(sum(v.X for v in self._xi_r.values()))
                     + self.inst.full_bin_count)
            last_load = sum(t * self._y[t, k].X
                            for t in self.inst.item_types
                            for k in range(self.machines))
            cmax   = m.ObjVal
            status = "optimal" if st == GRB.OPTIMAL else "feasible"
        else:
            z_val     = (round(sum(v.X for v in self._xi_r.values()))
                         + self.inst.full_bin_count)
            last_load = sum(t * self._y[t].X for t in self.inst.item_types)
            cmax      = m.ObjVal
            status    = "optimal" if st == GRB.OPTIMAL else "feasible"

        res = self._build_result(
            status=status, cmax=cmax, z_val=z_val, last_load=last_load,
            lb=m.ObjBound, gap=m.MIPGap, runtime=rt,
            root_gap=root_gap,
            runtime_phase1=None, phase1_root_gap=None, phase2_root_gap=None)
        return res

    def _solve_two_phase(self) -> SolverResult:
        m  = self._model
        t0 = time.time()

        # Check whether phase 1 already failed
        if not getattr(m, "_phase2_ready", False):
            rt = time.time() - t0 + m._phase1_time
            st = m.Status
            return SolverResult(
                status="infeasible" if st == GRB.INFEASIBLE else "timeout",
                cmax=None, z_nonlast=None, last_load=None,
                lb=None, gap=None, runtime=rt,
                root_gap=None,
                runtime_phase1=m._phase1_time,
                phase1_root_gap=m._phase1_root_gap,
                phase2_root_gap=None,
                instance=self.inst)

        # Phase 2
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
                lb=None, gap=None, runtime=rt,
                root_gap=None,
                runtime_phase1=m._phase1_time,
                phase1_root_gap=m._phase1_root_gap,
                phase2_root_gap=phase2_root_gap,
                instance=self.inst)

        T  = self.inst.T; tc = self.inst.t_charge
        z_val  = (round(sum(v.X for v in self._xi_r.values()))
                  + self.inst.full_bin_count)
        status = "optimal" if st == GRB.OPTIMAL else "feasible"

        if self.machines > 1:
            # Phase 2 objective is Cmax directly
            cmax      = m.ObjVal
            last_load = sum(t * self._y[t, k].X
                            for t in self.inst.item_types
                            for k in range(self.machines))
            lb        = m.ObjBound
            gap       = m.MIPGap
        else:
            # Phase 2 objective is last-bin load; reconstruct Cmax
            last_load = m.ObjVal
            cmax      = z_val * (T + tc) + last_load
            lb        = z_val * (T + tc) + m.ObjBound
            gap       = (cmax - lb) / cmax if cmax > 0 else 0.0

        res = self._build_result(
            status=status, cmax=cmax, z_val=z_val, last_load=last_load,
            lb=lb, gap=gap, runtime=rt,
            root_gap=None,
            runtime_phase1=m._phase1_time,
            phase1_root_gap=m._phase1_root_gap,
            phase2_root_gap=phase2_root_gap)
        return res
    
    def _build_result(self, *, status, cmax, z_val, last_load,
                      lb, gap, runtime,
                      root_gap, runtime_phase1, phase1_root_gap, phase2_root_gap
                      ) -> SolverResult:
        inst = self.inst

        # Debug: print non-zero variables
        try:
            for v in self._model.getVars():
                if v.X > 1e-5:
                    print(f"{v.varName} = {v.X:.4g}")
        except Exception:
            pass

        # Reconstruct bins
        xs_val      = {k: v.X for k, v in self._xi_s.items() if v.X > 0.5}
        xr_val      = {k: v.X for k, v in self._xi_r.items() if v.X > 0.5}
        bins_ptimes = _reconstruct_bins(xs_val, xr_val, self.graph)
        for t, count in inst.full_bin_items.items():
            for _ in range(count):
                bins_ptimes.append([t])

        # Sort bins by smallest job index (cosmetic)
        full_pool: Dict[int, List[int]] = defaultdict(list)
        for idx, p in enumerate(inst.jobs):
            full_pool[p].append(idx)

        bins_idx = [sorted(full_pool[p].pop(0) for p in pt) for pt in bins_ptimes]
        if bins_idx:
            bins_idx, bins_ptimes = zip(
                *sorted(zip(bins_idx, bins_ptimes), key=lambda x: x[0][0]))
            bins_idx    = list(bins_idx)
            bins_ptimes = list(bins_ptimes)

        # Last bin / per-machine reconstruction
        if self.machines > 1:
            # Determine how many non-last bins each machine gets from u[t,k] values.
            # bins_ptimes is already a flat list ordered by arc-flow reconstruction;
            # we consume from it sequentially per machine.
            bin_iter = iter(range(len(bins_ptimes)))
            machine_assignments = {}
            for k in range(self.machines):
                n_bins_k = (sum(round(self._u[t, k].X) for t in inst.item_types)
                            + round(self._u[-1, k].X)
                            + round(self._ub[k].X))
                # Take the next n_bins_k bins from the flat list
                k_bin_indices = [next(bin_iter) for _ in range(n_bins_k)]
                k_bins_ptimes  = [bins_ptimes[i]  for i in k_bin_indices]
                k_bins_idx     = [bins_idx[i]      for i in k_bin_indices]
                last_k_ptimes = []
                for t in sorted(inst.item_types):
                    last_k_ptimes.extend([t] * round(self._y[t, k].X))
                last_k_ptimes.sort()
                last_k_indices = sorted(full_pool[p].pop(0) for p in last_k_ptimes)
                machine_assignments[k] = {
                    "bins_ptimes":  k_bins_ptimes,
                    "bins_indices": k_bins_idx,
                    "last_ptimes":  last_k_ptimes,
                    "last_indices": last_k_indices,
                }
            last_ptimes = sorted(p for k in range(self.machines)
                                 for p in machine_assignments[k]["last_ptimes"])
            last_idx    = sorted(i for k in range(self.machines)
                                 for i in machine_assignments[k]["last_indices"])
        else:
            machine_assignments = None
            last_ptimes: List[int] = []
            for t, var in self._y.items():
                last_ptimes.extend([t] * round(var.X))
            last_ptimes.sort()
            last_idx = sorted(full_pool[p].pop(0) for p in last_ptimes)

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
    """Return the solution as a formatted string (shared by file writer and console)."""
    if res.cmax is None:
        return f"Status: {res.status}\nNo feasible solution found.\n"

    lines = []
    lines.append(f"The num of Batches: {res.z_nonlast}")
    lines.append(f"makespan: {res.cmax}")

    if res.machine_assignments is None:
        # Single-machine: flat batch list + single last bin
        lines.append("The jobs in each Batch")
        for b, (idx, pt) in enumerate(zip(res.bins_indices, res.bins_ptimes)):
            lines.append(f"Batch{b}:index:{idx} Processing Time: {sum(pt)}")
        lines.append("The last Batch")
        lines.append(f"index:{res.last_indices} "
                     f"Processing Time: {sum(res.last_ptimes or [])}")
    else:
        # Parallel-machine: shared batch list, then per-machine breakdown
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

        # root_gap column: single-phase fills root_gap; 2-phase fills phase1_root_gap
        root_gap_col = res.root_gap if res.root_gap is not None else res.phase1_root_gap

        w.writerow({
            "instance":       name,
            "n":              res.instance.n,
            "T":              res.instance.T,
            "status":         res.status,
            "cmax":           f"{res.cmax:.1f}"      if res.cmax      is not None else "",
            "lb":             f"{res.lb:.2f}"        if res.lb        is not None else "",
            "gap_pct":        f"{res.gap * 100:.4f}" if res.gap       is not None else "",
            "z_nonlast":      res.z_nonlast          if res.z_nonlast is not None else "",
            "last_load":      f"{res.last_load:.1f}" if res.last_load is not None else "",
            "runtime_s":      f"{res.runtime:.3f}",
            "runtime_phase1": f"{res.runtime_phase1:.3f}" if res.runtime_phase1 is not None else "",
            "root_gap":       f"{root_gap_col:.8f}"  if root_gap_col  is not None else "",
            "root_gap_phase2": (f"{res.phase2_root_gap:.8f}"
                                if res.phase2_root_gap is not None else ""),
            "model":          model,
            "numOfThreads":   num_threads,
            "set":            set_name,
            "comment":        "",
            "machines":       machines,
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
        #if i-1>317 and i-1 not in [369,372,527,546,598,614,658]:
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
                print(f"ERROR: {exc}")
                dummy = SMInstance(n=0, jobs=[], T=0, t_charge=t_charge)
                res = SolverResult(status="error", cmax=None, z_nonlast=None,
                                   last_load=None, lb=None, gap=None,
                                   runtime=0.0, root_gap=None,
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
    fname = os.path.basename(fpath)
    inst  = SMInstance.from_file(fpath, t_charge=t_charge)
    print(f"Instance : {fname}\n  {inst.summary()}")
    print(f"Graph    : {ReflectGraph(inst).summary()}")
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
# --- Folder run, single-phase, 2 machines ---
folder = "MOD"
run_folder(
    folder     = f"Benchmark Instances/Instances/{folder}/",
    csv_path   = f"results/{folder}_results_5M_symmCont_strengthened_2p.csv",
    sol_dir    = f"results/{folder}_solutions_5M_symmCont_strengthened_2p",
    t_charge   = 0, 
    time_limit = 3600.0,
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
    csv_path   = f"results/{folder}_results_2phase2.csv",
    sol_dir    = f"results/{folder}_solutions_2phase2",
    t_charge   = 0,
    time_limit = 3600.0,
    threads    = 1,
    verbose    = True,
    machines   = 1,
    two_phase  = True,
)
"""

"""
# --- Single instance run ---
setstr = "MOD"
file   = "L_00000224"
result = run_single(
    f"Benchmark Instances/Instances/{setstr}/{file}",
    t_charge   = 0,
    time_limit = 3600,
    machines   = 2,
    two_phase  = False,
)
print(result.cmax, result.z_nonlast, result.last_load)
#"""


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
"""
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Arc-Flow Reflect solver for P_m|pm|Cmax")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--instance", help="Single instance file")
    grp.add_argument("--folder",   help="Folder of instances (batch)")
    p.add_argument("--out_csv",    default="results.csv")
    p.add_argument("--out_sol",    default=None, help="Dir for .sol files")
    p.add_argument("--t_charge",   type=int,   default=0)
    p.add_argument("--time_limit", type=float, default=3600.0)
    p.add_argument("--threads",    type=int,   default=1)
    p.add_argument("--machines",   type=int,   default=1)
    p.add_argument("--two_phase",  action="store_true",
                   help="Use 2-phase approach (single machine only)")
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
                   sol_dir=args.out_sol, **kw)"""