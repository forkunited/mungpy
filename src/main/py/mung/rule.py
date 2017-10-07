class UnaryRule:
    def __init__(self):
        pass


class BinaryRule:
    def __init__(self):
        pass


class RuleSet:
    def __init__(self):
        self._unary_rules = []
        self._binary_rules = []

    def add_unary_rule(self, rule):
        self._unary_rules.append(rule)

    def add_binary_rule(self, rule):
        self._binary_rules.append(rule)

    def _apply_unary_rules(self, feature_token, results):
        for rule in self._unary_rules:
            if rule.matches(feature_token):
                results.extend(rule.apply(feature_token))
        return results

    def _apply_binary_rules(self, feature_token1, feature_token2, results):
        for rule in self._binary_rules:
            if rule.matches(feature_token1, feature_token2):
                results.extend(rule.apply(feature_token1, feature_token2))
        return results

    def apply(self, feature_tokens):
        results = []
        for i in range(len(feature_tokens)):
            self._apply_unary_rules(feature_tokens[i], results)
            for j in range(0, len(feature_tokens)):
                self._apply_binary_rules(feature_tokens[i], feature_tokens[j], results)
        return results

    def apply_unary_binary(self, unary, binary):
        results = []
        for i in range(len(unary)):
            self._apply_unary_rules(unary[i], results)

        for i in range(len(binary)):
            self._apply_binary_rules(binary[i][0], binary[i][1], results)
        
        return results

