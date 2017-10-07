import random
import nltk
import numpy as np
from nltk.sem.logic import *
import copy

class RelationalModel:
    def __init__(self, domain, properties, binary_rels, valuation):
        self._domain = domain
        self._properties = properties
        self._binary_rels = binary_rels
        self._model = nltk.Model(set(domain), nltk.Valuation(valuation))
        
        self._preds = dict()
        self._pred_sats = dict()
        for (pred, sats) in valuation:
            self._preds[pred] = sats
            
            if pred in binary_rels:
                sat_map_0 = dict()
                sat_map_1 = dict()
                for sat in sats:
                    if sat[0] not in sat_map_0:
                        sat_map_0[sat[0]] = set([])
                    if sat[1] not in sat_map_1:
                        sat_map_1[sat[1]] = set([])
                    sat_map_1[sat[1]].add(sat[0])
                    sat_map_0[sat[0]].add(sat[1])
                self._pred_sats[pred] = (sat_map_0, sat_map_1)
            elif pred in properties:
                sat_set = set([])
                for sat in sats:
                    sat_set.add(sat)
                self._pred_sats[pred] = sat_set

    def evaluate(self, form, g):
        return self._model.evaluate_exp(nltk.sem.Expression.fromstring(form), g)

    def evaluate_exp(self, exp, g):
        return self._model.satisfy(exp, g)

    def satisfiers(self, form, var, g):
        return self.satisfiers_exp(nltk.sem.Expression.fromstring(form), var, g)

    def satisfiers_exp(self, exp, var, g):
        return self._model.satisfiers(exp, var, g)

    # NOTE: This assumes there are no constants
    def satisfying_assignments_exp_app_fast(self, exp, g):
        if not isinstance(exp, nltk.sem.logic.ApplicationExpression):
            return None

        pred_vars = []
        for arg in exp.args:
            if not isinstance(arg, nltk.sem.logic.IndividualVariableExpression):
                return None
            else:
                pred_vars.append(str(arg))

        assgns = []
        pred_str = str(exp.pred)
        if not pred_str in self._preds:
            return assgns

        pred = self._preds[pred_str]
        for sat in pred:
            assgn = []
            consistent = True
            for i in range(len(sat)):
                var_i = pred_vars[i]
                val_i = sat[i]
                
                if var_i in g:
                    if g[var_i] != val_i:
                        consistent = False
                        break
                else:
                    assgn.append((var_i, val_i))
            
            if consistent:
                for var in g:
                    assgn.append((var, g[var]))
                assgns.append(nltk.Assignment(self._domain, assgn))
                
        return assgns

    # NOTE: This assumes there are no constants
    def satisfying_assignments_app_fast_nonltk(self, pred_str, pred_vars, g):
        assgns = []
        if not pred_str in self._preds:
            return assgns

        pred = self._preds[pred_str]
        
        if len(pred_vars) == 1:
            if pred_vars[0] not in g:
                for sat in self._pred_sats[pred_str]:
                    assgn = copy.copy(g)
                    assgn[pred_vars[0]] = sat
                    assgns.append(assgn)
            elif g[pred_vars[0]] in self._pred_sats[pred_str]:
                assgns.append(copy.copy(g))
         
            return assgns
        elif len(pred_vars) == 2:

            if pred_vars[0] not in g and pred_vars[1] not in g:
                for sat in pred:
                    assgn = copy.copy(g)
                    assgn[pred_vars[0]] = sat[0]
                    assgn[pred_vars[1]] = sat[1]
                    assgns.append(assgn)
            elif pred_vars[0] in g and pred_vars[1] not in g:
                if g[pred_vars[0]] in self._pred_sats[pred_str][0]:
                    for sat in self._pred_sats[pred_str][0][g[pred_vars[0]]]:
                        assgn = copy.copy(g)
                        assgn[pred_vars[1]] = sat
                        assgns.append(assgn)
            elif pred_vars[1] in g and pred_vars[0] not in g:
                if g[pred_vars[1]] in self._pred_sats[pred_str][1]:
                    for sat in self._pred_sats[pred_str][1][g[pred_vars[1]]]:
                        assgn = copy.copy(g)
                        assgn[pred_vars[0]] = sat
                        assgns.append(assgn)
            elif g[pred_vars[0]] in self._pred_sats[pred_str][0] and g[pred_vars[1]] in self._pred_sats[pred_str][0][g[pred_vars[0]]]:
                assgns.append(copy.copy(g))
            return assgns
        
        for sat in pred:
            assgn = dict()
            consistent = True
            for i in range(len(sat)):
                var_i = pred_vars[i]
                val_i = sat[i]

                if var_i in g:
                    if g[var_i] != val_i:
                        consistent = False
                        break
                else:
                    assgn[var_i] = val_i

            if consistent:
                for var in g:
                    assgn[var] = g[var]
                assgns.append(assgn)

        return assgns


    @staticmethod
    def make_random(domain, properties, binary_rels):
        property_sets = [set([]) for i in range(len(properties))]
        binary_rel_sets = [set([]) for i in range(len(binary_rels))]

        v = []
        for i in range(len(domain)):
            v.append((domain[i], domain[i])) 
            
            for j in range(len(properties)):
                if np.random.randint(2) == 1:
                    property_sets[j].add(domain[i])
            
            for j in range(len(binary_rels)):
                for k in range(len(domain)):
                    if np.random.randint(2) == 1:
                        binary_rel_sets[j].add((domain[i], domain[k]))

        for i in range(len(property_sets)):
            v.append((properties[i], property_sets[i]))

        for i in range(len(binary_rel_sets)):
            v.append((binary_rels[i], binary_rel_sets[i]))

        return RelationalModel(domain, properties, binary_rels, v)


class ClosedFormula:
    def __init__(self, form, g):
        self._exp = None
        self._form = form
        self._g = g

    def get_g(self):
        return self._g

    def get_form(self):
        return self._form

    def get_exp(self, close=False, exclude=[]):
        if self._exp is None:
            self._exp = nltk.sem.Expression.fromstring(self._form)
        exp = self._exp

        if not close:
            return exp
        for key in self._g:
            if key not in exclude:
                key_exp = nltk.sem.Expression.fromstring(key)
                value_exp = nltk.sem.Expression.fromstring(value)
                exp = exp.replace(key.variable, value, False)
        return exp


    def exp_matches(self, closed_form):
        return self.get_exp().equiv(closed_form.get_exp())

    def is_sub_g(self, closed_form):
        for v in self._g:
            f_g = closed_form.get_g()
            if v not in f_g or f_g[v] != self._g[v]:
                return False
        return True
    
    def matches(self, closed_form):
        return self.exp_matches(closed_form) and self.is_sub_g(closed_form)

    def orthogonize(self, closed_form):
        new_exp = self.get_exp()
        new_g = []
        for v in self._g:
            if v in closed_form.get_g():
                v_new = v
                while v_new in closed_form.get_g():
                    v_new = v_new + '1'
                v_exp = nltk.sem.Expression.fromstring(v)
                v_new_exp = nltk.sem.Expression.fromstring(v_new)
                new_exp = new_exp.replace(v_exp.variable, v_new_exp, False)
                new_g.append((v_new, self._g[v]))
            else:
                new_g.append((v, self._g[v]))
        return ClosedFormula(str(new_exp), nltk.Assignment(self._g.domain, new_g))

    def __str__(self):
        new_exp = self.get_exp()
        for v in self._g:
            v_exp = nltk.sem.Expression.fromstring(v)
            value_exp = nltk.sem.Expression.fromstring(self._g[v])
            new_exp = new_exp.replace(v_exp.variable, value_exp, False)
        return str(new_exp)

    def conjoin(self, cf2, orthogonize=False):
        if orthogonize:
            cf2 = cf2.orthogonize(self)
        new_g = of2_o.get_g().copy()
        new_g.update(self.get_g())
        return ClosedFormula(str(self.get_exp() & of2.get_exp()), new_g)

    def disjoin(self, cf2, orthogonize=False):
        if orthogonize:
            cf2 = cf2.orthogonize(self)
        new_g = of2_o.get_g().copy()
        new_g.update(self.get_g())
        return ClosedFormula(str(self.get_exp() | of2.get_exp()), new_g)

    def negate(self):
        return ClosedFormula(str(- self.get_exp()), self.get_g())

    def is_unary_predicate(self):
        return isinstance(self.get_exp(), nltk.sem.logic.ApplicationExpression) and (len(self.get_exp().free()) + len(self.get_exp().constants())) == 1

    def get_predicates(self):
        return [str(pred) for pred in self.get_exp().predicates()]


class OpenFormula:
    def __init__(self, domain, form, variables, init_g=None):
        self._exp = None
        self._domain = domain
        self._form = form
        self._variables = variables
        if init_g is not None:
            self._init_g = init_g
        else:
            self._init_g = nltk.Assignment(self._domain, [])
        self._closed_forms = None
        self._predicates = None

    def _make_closed_forms(self, domain, form, variables, init_g):
        assgn_lists = self._make_assignments_helper(domain, variables, init_g, 0, [[]])
        closed = []
        for i in range(len(assgn_lists)):
            g = nltk.Assignment(domain, assgn_lists[i])
            closed.append(ClosedFormula(form, g))
        return closed

    def _make_assignments_helper(self, domain, variables, init_g, i, assgn_lists):
        if (i == len(variables)):
            return assgn_lists
        next_lists = []

        if variables[i] not in init_g:
            for j in range(len(domain)):
                for k in range(len(assgn_lists)):
                    assgn_list = copy.deepcopy(assgn_lists[k])
                    assgn_list.append((variables[i], domain[j]))        
                    next_lists.append(assgn_list)
        else:
            for k in range(len(assgn_lists)):
                assgn_list = copy.deepcopy(assgn_lists[k])
                assgn_list.append((variables[i], init_g[variables[i]]))
                next_lists.append(assgn_list)

        return self._make_assignments_helper(domain, variables, init_g, i+1, next_lists)

    def get_variables(self):
        return self._variables

    def get_init_g(self):
        return self._init_g

    def exp_matches(self, open_form):
        return self.get_exp().equiv(open_form.get_exp())

    def get_domain(self):
        return self._domain

    def get_form(self):
        return self._form

    def get_exp(self):
        if self._exp is None:
            self._exp = nltk.sem.Expression.fromstring(self._form)
        return self._exp

    def get_closed_forms(self):
        if self._closed_forms is None:
            self._closed_forms = self._make_closed_forms(self._domain, self._form, self._variables, init_g)
        return self._closed_forms

    def orthogonize(self, open_form):
        new_exp = self.get_exp()
        new_g = []
        new_variables = []
        for v in self._variables:
            if v in open_form._variables:
                v_new = v
                while v_new in open_form._variables:
                    v_new = v_new + '1'
                v_exp = nltk.sem.Expression.fromstring(v)
                v_new_exp = nltk.sem.Expression.fromstring(v_new)
                new_exp = new_exp.replace(v_exp.variable, v_new_exp, False)
                if v in self._init_g:
                    new_g.append((v_new, self._init_g[v]))

                new_variables.append(v_new)
            elif v in self._init_g:
                new_g.append((v, self._init_g[v]))
                new_variables.append(v)
            else:
                new_variables.append(v)
        return OpenFormula(self._domain, str(new_exp), new_variables, nltk.Assignment(self._init_g.domain, new_g))

    # FIXME Note that for now this assumes that self and of2 have the same domain
    def conjoin(self, of2, orthogonize=False):
        if orthogonize:
            of2 = of2.orthogonize(self)
        new_init_g = of2.get_init_g().copy()
        new_init_g.update(self.get_init_g())
        new_variables = list(set(self._variables + of2._variables))
 
        return OpenFormula(self._domain, str(self.get_exp() & of2.get_exp()), new_variables, init_g=new_init_g)


    def disjoin(self, of2, orthogonize=False):
        if orthogonize:
            of2 = of2.orthogonize(self)
        new_init_g = of2_o.get_init_g().copy()
        new_init_g.update(self.get_init_g())
        new_variables = list(set(self._variables + of2._variables))

        return OpenFormula(self._domain, str(self.get_exp() | of2.get_exp()), new_variables, init_g=new_init_g)

    def negate(self):
        return OpenFormula(self._domain, str(- self._get_exp()), self._variables, self._init_g)

    # FIXME This only works if the formula is a predicate
    # Also assumes init_g is empty
    def satisfiers_broken_fast(self, model, g):
        return model.satisfying_assignments_app_fast_nonltk(self.get_predicates()[0], self.get_variables(), g)

    def _satisfiers_helper(self, model, variables, var_exprs, var_index, partial_g):
        if var_index == len(self._variables):
             return [nltk.Assignment(self._domain, partial_g)]

        sats = []
        var = variables[var_index]
        expr = var_exprs[var_index]
        var_sats = model.satisfiers_exp(expr, var, nltk.Assignment(self._domain, partial_g))

        for var_sat in var_sats:
            partial_g_next = copy.deepcopy(partial_g)
            partial_g_next.append((var, var_sat))
            sats.extend(self._satisfiers_helper(model, variables, var_exprs, var_index + 1, partial_g_next))

        return sats

    def satisfiers(self, model, g=None):
        # First try the fast way if this is just a predicate expression
        full_init_g_obj = None
        if g is None:
            full_init_g_obj = self._init_g
        elif len(self._init_g) == 0:
            full_init_g_obj = g
        else:
            full_init_g = []
            for var in g:
                full_init_g.append((var, g[var]))
            for var in self._init_g:
                if var not in g:
                    full_init_g.append((var, self._init_g[var]))
            full_init_g_obj = nltk.Assignment(self._domain, full_init_g)

        maybe_sats = model.satisfying_assignments_exp_app_fast(self.get_exp(), full_init_g_obj)
        if maybe_sats is not None:
            return maybe_sats

        # Otherwise, do the slow way
        variables = []
        partial_g = []
        var_exprs = []
        for i in range(len(self._variables)):
            var = self._variables[i]
            if var not in self._init_g and (g is None or var not in g):
                exist_str_i = ""
                for j in range(i+1, len(self._variables)):
                    exist_str_i += "exists " + self._variables[j] + "."
                if len(exist_str_i) == 0:
                    var_exprs.append(self.get_exp())
                else:
                    var_exprs.append(nltk.sem.Expression.fromstring(exist_str_i + self._form))
                variables.append(var)
            else:
                if var in self._init_g:
                    partial_g.append((var, self._init_g[var]))
                else:
                    partial_g.append((var, g[var]))
        
        if len(variables) == 0:
            partial_g_obj = nltk.Assignment(self._domain, partial_g)
            if model.evaluate_exp(self.get_exp(), partial_g_obj):
                return [partial_g_obj]
            else:
                return []

        return self._satisfiers_helper(model, variables, var_exprs, 0, partial_g)

    def is_unary_predicate(self):
        return isinstance(self.get_exp(), nltk.sem.logic.ApplicationExpression) and (len(self.get_exp().free()) + len(self.get_exp().constants())) == 1

    def get_predicates(self):
        if self._predicates is None:
            self._predicates = [str(pred) for pred in self.get_exp().predicates()]
        return self._predicates

    def clone(self, var_map=dict(), symmetric=False):
        new_exp = self.get_exp()
        new_g = []
        new_variables = []
        for v in self._variables:
            if v in var_map:
                v_new = var_map[v]
                v_exp = nltk.sem.Expression.fromstring(v)
                v_new_exp = nltk.sem.Expression.fromstring(v_new)
                new_exp = new_exp.replace(v_exp.variable, v_new_exp, False)
                if v in self._init_g:
                    new_g.append((v_new, self._init_g[v]))
                new_variables.append(v_new)
            elif v in self._init_g:
                new_g.append((v, self._init_g[v]))
                new_variables.append(v)
            else:
                new_variables.append(v)

        new_form = OpenFormula(self._domain, str(new_exp), new_variables, nltk.Assignment(self._init_g.domain, new_g))
        if not symmetric:
            return new_form
        else:
            svars = sorted(new_variables)
            
            smap = dict()
            remp_map = dict()
            for i in range(len(new_variables)):
                smap[new_variables[i]] = svars[i] + "_p"
                remp_map[svars[i] + "_p"] = svars[i]             

            return new_form.clone(var_map=smap).clone(var_map=remp_map)
