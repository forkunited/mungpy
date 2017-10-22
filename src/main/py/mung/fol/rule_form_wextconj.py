from mung import data
from mung.fol import feature_form_wextconj
from mung import rule

class BinaryJoinPropertyRule(rule.BinaryRule):
    def __init__(self, name):
        rule.BinaryRule.__init__(self, name)

    def matches(self, feature_token1, feature_token2):
        if not isinstance(feature_token1, feature_form_wextconj.FeatureFormWextconjToken) or not isinstance(feature_token2, feature_form_wextconj.FeatureFormWextconjToken):
            return False
        if feature_token1.get_conjunct_count() <= 1 or feature_token2.get_conjunct_count() <= 1:
            return False
        return True
        #f_form2 = feature_token2.get_closed_form()
        #if self._lhs_closed_form1.matches(f_form1) and self._lhs_closed_form2.matches(f_form2):
        #    return True
        #if not self._ordered and self._lhs_closed_form1.matches(f_form2) and self._lhs_closed_form2.matches(f_form1):
        #    return True
        #return False

    def apply(self, feature_token1, feature_token2):
        if not self.matches(feature_token1, feature_token2):
            return None
        
        output_feats = []
        pred_indices = dict()
        for i in range(feature_token1.get_conjunct_count()):
            conj = feature_token1.get_conjunct(i)
            if not conj.is_unary_predicate():
                continue
            pred = conj.get_predicates()[0]
            if pred not in pred_indices:
                pred_indices[pred] = []
            pred_indices[pred].append(i)

        for i in range(feature_token2.get_conjunct_count()):
            conj = feature_token2.get_conjunct(i)
            if not conj.is_unary_predicate():
                continue
            
            pred = conj.get_predicates()[0]             
            if pred not in pred_indices:
                continue

            indices1 = pred_indices[pred]
            for j in indices1:
                output_feats.append(self._make_feature_type(feature_token1, feature_token2, j, i))
        return output_feats

    def _make_feature_type(self, ftoken1, ftoken2, join_index1, join_index2):
        conjs = []
        for i in range(ftoken1.get_conjunct_count()):
            conj = ftoken1.get_conjunct(i)
            vmap = dict()
            for v in conj.get_variables():
                vmap[v] = "f" + v
            conjs.append(conj.clone(var_map=vmap))

        join_var_target = "f" + ftoken1.get_conjunct(join_index1).get_variables()[0]
        join_var_source = ftoken2.get_conjunct(join_index2).get_variables()[0]
        
        for i in range(ftoken2.get_conjunct_count()):
            if i == join_index2:
                continue
            conj = ftoken2.get_conjunct(i)
            vmap = dict()
            for v in conj.get_variables():
                if v == join_var_source:
                    vmap[v] = join_var_target
                else: 
                    vmap[v] = "s" + v
            conjs.append(conj.clone(var_map=vmap))
        
        conj_lists = [[conj] for conj in conjs]
        weight_fn = ftoken1.get_weight_fn()

        return feature_form_wextconj.FeatureFormWextconjType(ftoken1.get_name() + "_" + ftoken2.get_name() + "_" + self.get_name(), conj_lists, weight_fn)          



