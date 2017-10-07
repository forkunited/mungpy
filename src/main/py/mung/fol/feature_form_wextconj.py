""" Weighted extistential conjunction features """

from mung import feature
import copy
import nltk
import numpy as np
import dill as pickle 

def _form_cmp(x,y):
    if x.is_unary_predicate() and not y.is_unary_predicate():
        return -1
    elif not x.is_unary_predicate() and y.is_unary_predicate():
        return 1
    else:
        return (y.get_form() > x.get_form()) - (y.get_form() < x.get_form())

class FeatureFormWextconjToken(feature.FeatureToken):
    def __init__(self, name, conjuncts, weight_fn, symmetric_rels=True, distinct_bindings=True, normalize=True):
        feature.FeatureToken.__init__(self)

        cs = list(conjuncts)        

        # FIXME Want this to sort by forms with variables removed (so that order is independent of 
        # variables.  This currently doesn't work correctly, but will cover the cases for which
        # it's currently used
        #
        # This also currently works untder the assumption that variables will appear in at least
        # one unary predicate
        cs.sort(cmp=_form_cmp)
        vmap = dict()
        var_index = 0
        for c in cs:
            vars_c = c.get_variables()        
            for var in vars_c:
                if var in vmap:
                    continue
                vmap[var] = "x" + str(var_index)
                var_index += 1

        for i in range(len(cs)):
            cs[i] = cs[i].clone(var_map=vmap, symmetric=symmetric_rels)

        cs.sort(cmp=_form_cmp) # Sort a second time to take new variable names into account

        self._conjuncts = cs

        self._conjuncts_opt_ordered = self._make_conjuncts_opt_ordered()

        self._open_conj = self._make_open_conj(cs)
        self._weight_fn = weight_fn
        self._distinct_bindings = distinct_bindings

        self._normalize = normalize
        self._mean = None
        self._sd = None
 
        self._name = name

    def _make_conjuncts_opt_ordered(self):
        vars_to_unary = dict()
        vars_to_nonunary = dict()
        variables = set([])
        conjs_ord = []

        for i in range(len(self._conjuncts)):
            conj = self._conjuncts[i]
            if conj.is_unary_predicate():
                for var in conj.get_variables():
                    if var not in vars_to_unary:
                        vars_to_unary[var] = []
                    vars_to_unary[var].append(i)
                    variables.add(var)
            else:
                for var in conj.get_variables():
                    if var not in vars_to_nonunary:
                        vars_to_nonunary[var] = []
                    vars_to_nonunary[var].append(i)
                    variables.add(var)
        
        added_conjs = set([])
        for var in variables:
            if var in vars_to_unary:
                unary_indices = vars_to_unary[var]
                for unary_i in unary_indices:
                    if unary_i not in added_conjs:
                        conjs_ord.append(self._conjuncts[unary_i])
                        added_conjs.add(unary_i)

            if var not in vars_to_nonunary:
                continue
 
            nonunary_indices = vars_to_nonunary[var]
            for nonunary_i in nonunary_indices:
                if nonunary_i not in added_conjs:
                    nonunary_conj = self._conjuncts[nonunary_i]
                    conjs_ord.append(nonunary_conj)
                    added_conjs.add(nonunary_i)
                    cur_vars = nonunary_conj.get_variables()
                    for cur_var in cur_vars:
                        cur_unary_indices = vars_to_unary[cur_var]
                        for cur_unary_i in cur_unary_indices:
                            if cur_unary_i not in added_conjs:
                                conjs_ord.append(self._conjuncts[cur_unary_i])
                                added_conjs.add(cur_unary_i)
        return conjs_ord

    def _make_open_conj(self, conjuncts):
        open_conj = conjuncts[0]
        for i in range(1, len(conjuncts)):
            open_conj = open_conj.conjoin(conjuncts[i])
        return open_conj

    def get_open_conj(self):
        return self._open_conj

    def get_weight_fn(self):
        return self._weight_fn

    def get_conjunct_count(self):
        return len(self._conjuncts)

    def get_conjunct(self, i):
        return self._conjuncts[i]

    def __str__(self):
        return str(self._open_conj.get_form()) + "_" + self._weight_fn.__name__ # FIXME Hack

    def _compute_helper(self, model, conj_index, partial_g_internal, partial_values):
        partial_g = partial_g_internal #nltk.Assignment(self._conjuncts[0].get_domain(), partial_g_internal)
        if conj_index < 0:
            return [partial_g]

        conj_i = self._conjuncts_opt_ordered[conj_index]
        sats_i = conj_i.satisfiers_broken_fast(model, partial_g) #conj_i.satisfiers(model, g=partial_g)
        sats = []
        for sat in sats_i:
            partial_g_next = dict() # []
            partial_values_next = copy.copy(partial_values)
            #for assn in partial_g_internal:
            #    partial_g_next.append(assn)
            
            for key in partial_g:
                partial_g_next[key] = partial_g[key]

            failed_distinct = False
            for var in sat:
                if var not in partial_g:
                    #partial_g_next.append((var, sat[var]))
                    partial_g_next[var] = sat[var]
                    if self._distinct_bindings and sat[var] in partial_values_next:
                        failed_distinct = True
                        break
                    partial_values_next.add(sat[var])
            
            if not failed_distinct:
                sats.extend(self._compute_helper(model, conj_index-1, partial_g_next, partial_values_next))
        return sats

    def compute(self, datum):
        # NOTE Old slow way
        #sats = self._open_conj.satisfiers(datum.get_model()) 
        
        # NOTE Faster?
        sats = self._compute_helper(datum.get_model(), len(self._conjuncts)-1, dict(), set([])) # Change dict to [] if want to use nltk assignments
        # End faster?

        weight = self._weight_fn(datum, sats)
        if self._normalize and self._mean is not None:
            return (weight - self._mean)/self._sd
        else:
            return weight

    def equals(self, feature_token):
        if not isinstance(feature_token, FeatureFormWextconjToken):
            return False

        if self._weight_fn.__name__ != feature_token._weight_fn.__name__:  # FIXME Hack
            return False

        for i in range(len(self._conjuncts)):
            if self._conjuncts[i].get_form() != feature_token._conjuncts[i].get_form():
                return False

        return True

    def init_start(self):
        self._init_normalize = self._normalize
        self._normalize = False
        self._mean = 0.0
        self._sd = 0.0
        self._init_values = []

    def init_datum(self, datum):
        self._init_values.append(self.compute(datum))

    def init_end(self):
        self._mean = np.mean(self._init_values)
        self._sd = np.std(self._init_values)
        if self._sd == 0.0:
            self._sd = 1.0

        self._normalize = self._init_normalize
        self._init_values = None

    def get_name(self):
        return self._name

class FeatureFormWextconjType(feature.FeatureType):
    def __init__(self, name, conjunct_lists, weight_fn, symmetric_rels=True, distinct_bindings=True):
        feature.FeatureType.__init__(self)

        self._symmetric_rels = symmetric_rels
        self._distinct_bindings = distinct_bindings

        self._conjunct_lists = []
        for conjunct_list in conjunct_lists:
            clist = list(conjunct_list)
            clist.sort(cmp=lambda x, y : (y.get_form() > x.get_form()) - (y.get_form() < x.get_form()))
            self._conjunct_lists.append(clist)
        
        self._weight_fn = weight_fn
        self._tokens = self._make_feature_tokens()

        self._name = name

    def _make_feature_tokens(self):
        tokens = self._make_feature_tokens_helper(0, [])
        unique_tokens = []
        for token in tokens:
            dup = False
            for unique_token in unique_tokens:
                if unique_token.equals(token):
                    dup = True
            if not dup:
                unique_tokens.append(token)
        return unique_tokens 

    def _make_feature_tokens_helper(self, conj_index, partial_conjs):
        if conj_index == len(self._conjunct_lists):
            return [FeatureFormWextconjToken(self._name + "_" + str(conj_index), partial_conjs, self._weight_fn, symmetric_rels=self._symmetric_rels, distinct_bindings=self._distinct_bindings)]

        tokens = []
        for conj in self._conjunct_lists[conj_index]:
            next_conjs = copy.deepcopy(partial_conjs)
            next_conjs.append(conj)
            tokens.extend(self._make_feature_tokens_helper(conj_index + 1, next_conjs))
        return tokens

    def compute(self, datum, vec, start_index):
        for i in range(len(self._tokens)):
            vec[start_index + i] = tokens[i].compute(datum)
        return vec

    def get_size(self):
        return len(self._tokens)

    def get_token(self, index):
        return self._tokens[index]

    def __eq__(self, feature_type):
        if not isinstance(feature_type, FeatureFormWextconjType):
            return False
        
        if self._weight_fn.__name__ != feature_type._weight_fn.__name__:  # FIXME Hack
            return False

        if len(self._conjunct_lists) != len(feature_type._conjunct_lists):
            return False

        for i in range(len(self._tokens)):
            if not self._tokens[i].equals(feature_type._tokens[i]):
                return False

        return True

    def init_start(self):
        for token in self._tokens:
            token.init_start()

    def init_datum(self, datum):
        for token in self._tokens:
            token.init_datum(datum)

    def init_end(self):
        for token in self._tokens:
            token.init_end()

    def get_name(self):
        return self._name

    def save(self, file_path):
        obj = dict()
        obj["type"] = "FeatureFormWextconjType"
        obj["name"] = self._name

        obj["symmetric_rels"] = self._symmetric_rels
        obj["distinct_bindings"] = self._distinct_bindings
        obj["conjunct_lists"] = pickle.dumps(self._conjunct_lists)

        obj["weight_fn"] = pickle.dumps(self._weight_fn)

        with open(file_path, 'w') as fp:
            json.dump(obj, fp)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as fp:
            obj = json.load(fp)
            return FeatureFormWextconjType.from_dict(obj)

    @staticmethod
    def from_dict(obj):
        name = obj["name"]
        symmetric_rels = obj["symmetric_rels"]
        distinct_bindings = obj["distinct_bindings"]
        conjunct_lists = pickle.loads(obj["conjunct_lists"])
        weight_fn = pickle.loads(obj["weight_fn"])

        return FeatureFormWextconjType(name, conjunct_lists, weight_fn, symmetric_rels=symmetric_rels, distinct_bindings=distinct_bindings)

feature.register_feature_type(FeatureFormWextconjType)
