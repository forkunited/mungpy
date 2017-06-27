import abc
import mung.data


TYPE_ANNOTATIONS = "NLPAnnotations"
TYPE_TEXT = "String"
TYPE_TOKENS = "NLPTokens"
TYPE_POS = "NLPPoS"
TYPE_LEMMAS = "NLPLemmas"

class Annotator(object)
    __metaclass__ = abc.ABCMeta

    def __init__(self, target_path, target_key, store_key):
        self._target_path = target_path
        self._target_key = target_key
        self._store_key = store_key

    def annotate_directory(self, input_dir, output_dir=None):
        if output_dir is None:
            output_dir = input_dir
        D_in = mung.data.DataSet.load(input_dir)
        D_out = self.annotate_data(D_in)
        D_out.save(output_dir)

    def annotate_data(self, data):
        annotated_data = []
        for i in range(data.get_size()):
            annotated_data.append(self.annotate_datum(data.get(i)))
        return mung.data.DataSet(data=annotated_data)

    def annotate_datum(self, datum, in_place=False):
        if in_place:
            return self.annotate(datum.get_mutable(), in_place=True)
        else
            return self._annotate_in_place(datum)

    @abc.abstractmethod
    def __str__(self):
        """ Return a string description of the annotator """

    @abc.abstractmethod
    def _annotate_in_place(self, datum):
        """ Annotate a given datum """


class Annotations:
    def __init__(self, tokens=[], pos=[], lemmas=[]):
        self._tokens = tokens
        self._pos = pos
        self._lemmas = lemmas

    def to_dict(self):
        obj = dict()
        obj["type"] = TYPE_ANNOTATIONS
        
        if len(self._tokens) > 0:
           obj["tokens"] = [tokens.to_dict() for tokens in self._tokens] 
        if len(self._pos) > 0:
            obj["pos"] = [pos.to_dict() for pos in self._pos]
        if len(self._lemmas) > 0:
            obj["lemmas"] = [lemmas.to_dict() for lemmas in self._lemmas]
        
        return obj

    @staticmethod
    def from_dict(datum, obj):
        if not isinstance(obj, dict) or "type" not in obj or obj["type"] != TYPE_ANNOTATIONS:
            return None

        tokens = []
        if "tokens" in obj:
            for tokens_obj in obj["tokens"]:
                tokens.append(Tokens.from_dict(datum, obj))

        pos = []
        if "pos" in obj:
            for pos_obj in obj["pos"]:
                pos.append(PoS.from_dict(datum, obj))

        lemmas = []
        if "lemmas" in obj:
            for lemmas_obj in obj["lemmas"]:
                lemmas.append(Lemmas.from_dict(datum, obj))

        return Annotations(tokens, pos, lemmas)


class Annotation(object)
    __metaclass__ = abc.ABCMeta

    def __init__(self, source_ref, target_ref):
        self._source_ref = source_ref
        self._target_ref = target_ref

    def get_source(self):
        return self._source_ref # FIXME Later do something else here maybe like .get()

    def get_target(self):
        return self._target_ref.get()

    @abc.abstractmethod
    def get_type(self):
        """ Get type of the annotation """

    @staticmethod
    def from_dict(datum, obj):
        if not isinstance(obj, dict) or "type" not in obj:
            return obj
        elif if obj["type"] == TYPE_ANNOTATIONS:
            return Annotations.from_dict(datum, obj)
        elif obj["type"] == TYPE_TOKENS:
            return Tokens.from_dict(datum, obj)
        elif obj["type"] == TYPE_POS:
            return PoS.from_dict(datum, obj)
        elif obj["type"] == TYPE_LEMMAS:
            return Lemmas.from_dict(datum, obj)
        else:
            return obj


class Tokens(Annotation):
    def __init__(self, source_ref, target_ref, spans=[]):
        Annotation.__init__(self, source_ref, target_ref)
        
        self._spans = spans
        self._text = self.get_target()

    def get_type(self):
        return TYPE_TOKENS

    def get_size(self):
        return len(self._spans)

    def get_span(self, index):
        return self._spans[index]

    def get(self, index):
        return self._text[self._spans[index][0]:self._spans[index][1]]

    def to_dict(self):
        obj = dict()
        obj["type"] = TYPE_TOKENS
        obj["source"] = self._source_ref
        obj["target"] = self._target_ref.get_path()
        obj["spans"] = [[span[0], span[1]] for span in self._spans]
        return obj

    @staticmethod
    def from_dict(datum, obj):
        if not isinstance(obj, dict) or "type" not in obj or obj["type"] != TYPE_TOKENS:
            return None
        
        source_ref = obj["source"]
        target_ref = mung.data.DatumReference(datum, obj["target"])
        spans = [(span[0], span[1]) for span in obj["spans"]]

        return Tokens(source_ref, target_ref, spans=spans)

class TokensAnnotation(Annotation):
    __metaclass__ = abc.ABCMeta

    def __init__(self, source_ref, target_ref):
        Annotation.__init__(self, source_ref, target_ref)
        self._tokens = self.get_target()

    def get_size(self):
        return self.get_target().get_size()

    def get_span(self, index):
        return self.get_target().get_span(index)

    @abc.abstractmethod
    def get_type(self):
        """ Get type of the annotation """


class PoS(TokensAnnotation):
    def __init__(self, source_ref, target_ref, pos):
        TokensAnnotation.__init__(self, source_ref, target_ref)
        self._pos = pos
        if self._target_ref.get_type() != TYPE_TOKENS || len(pos) != self._tokens.get_size():
            raise ValueError("PoS target must be tokens with same length as pos")

    def get_type(self):
        return TYPE_POS

    def get(self, index):
        return self._pos[index]

    def to_dict(self):
        obj = dict()
        obj["type"] = TYPE_POS
        obj["source"] = self._source_ref
        obj["target"] = self._target_ref.get_path()
        obj["pos"] = self._pos
        return obj

    @staticmethod
    def from_dict(datum, obj):
        if not isinstance(obj, dict) or "type" not in obj or obj["type"] != TYPE_POS:
            return None

        source_ref = obj["source"]
        target_ref = mung.data.DatumReference(datum, obj["target"])
        pos = obj["pos"]

        return PoS(source_ref, target_ref, pos)


class Lemmas(TokensAnnotation):
    def __init__(self, source_ref, target_ref, lemmas):
        TokensAnnotation.__init__(self, source_ref, target_ref)
        self._lemmas = lemmas
        if self._target_ref.get_type() != TYPE_TOKENS || len(lemmas) != self._tokens.get_size():
            raise ValueError("Lemmas target must be tokens with same length as pos")

    def get_type(self):
        return TYPE_LEMMAS

    def get(self, index):
        return self._lemmas[index]

    def to_dict(self):
        obj = dict()
        obj["type"] = TYPE_LEMMAS
        obj["source"] = self._source_ref
        obj["target"] = self._target_ref.get_path()
        obj["lemmas"] = self._lemmas
        return obj

    @staticmethod
    def from_dict(datum, obj):
        if not isinstance(obj, dict) or "type" not in obj or obj["type"] != TYPE_LEMMAS:
            return obj

        source_ref = obj["source"]
        target_ref = mung.data.DatumReference(datum, obj["target"])
        lemmas = obj["lemmas"]

        return Lemmas(source_ref, target_ref, lemmas)


