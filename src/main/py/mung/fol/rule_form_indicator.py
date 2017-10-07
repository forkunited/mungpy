from mung import data
from mung.fol import feature_form_indicator
from mung import rule

class UnaryRule(rule.UnaryRule):
    def __init__(self, lhs_closed_form, rhs_open_form_fn):
        rule.UnaryRule.__init__(self)
        self._lhs_closed_form = lhs_closed_form
        self._rhs_open_form_fn = rhs_open_form_fn

    def matches(self, feature_token):
        if not isinstance(feature_token, feature_form_indicator.FeatureFormIndicatorToken):
            return False

        if self._lhs_closed_form is None:
            return True

        f_form = feature_token.get_closed_form()

        if not self._lhs_closed_form.matches(f_form):
            return False

        return True

    def apply(self, feature_token):
        if not self.matches(feature_token):
            return None
        open_forms = self._rhs_open_form_fn(feature_token.get_closed_form())
        return [feature_form_indicator.FeatureFormIndicatorType(open_form) for open_form in open_forms]


class BinaryRule(rule.BinaryRule):
    def __init__(self, lhs_closed_form1, lhs_closed_form2, rhs_open_form_fn, ordered=False):
        rule.BinaryRule.__init__(self)
        self._lhs_closed_form1 = lhs_closed_form1
        self._lhs_closed_form2 = lhs_closed_form2
        self._rhs_open_form_fn = rhs_open_form_fn
        self._ordered = ordered

    def matches(self, feature_token1, feature_token2):
        if not isinstance(feature_token1, feature_form_indicator.FeatureFormIndicatorToken) or not isinstance(feature_token2, feature_form_indicator.FeatureFormIndicatorToken):
            return False
        if self._lhs_closed_form1 is None and self._lhs_closed_form2 is None:
            return True


        f_form1 = feature_token1.get_closed_form()
        f_form2 = feature_token2.get_closed_form()
        if self._lhs_closed_form1.matches(f_form1) and self._lhs_closed_form2.matches(f_form2):
            return True
        if not self._ordered and self._lhs_closed_form1.matches(f_form2) and self._lhs_closed_form2.matches(f_form1):
            return True
        return False

    def apply(self, feature_token1, feature_token2):
        if not self.matches(feature_token1, feature_token2):
            return None
        open_forms = self._rhs_open_form_fn(feature_token1.get_closed_form(), feature_token2.get_closed_form())
        return [feature_form_indicator.FeatureFormIndicatorType(open_form) for open_form in open_forms]
