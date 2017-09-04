import abc
import mung.data

TYPE_TOKENS = "NLPTokens"
TYPE_POS = "NLPPoS"
TYPE_LEMMAS = "NLPLemmas"
TYPE_STRINGS = "NLPStrings"
TYPE_SENTENCES = "NLPSentences"

class Annotator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, target_path, target_key, store_key):
        self._target_path = target_path
        self._target_key = target_key
        self._store_key = store_key

    def annotate_directory(self, input_dir, output_dir=None, id_key="id", batch=1000):
        if output_dir is None:
            output_dir = input_dir
        D_in = mung.data.DataSet.load(input_dir, id_key=id_key)
        D_out = self.annotate_data(D_in)
        D_out.save(output_dir, batch=batch)

    def annotate_data(self, data):
        annotated_data = []
        for i in range(data.get_size()):
            print "Annotating " + data.get(i).get_id() + " (" + str(i+1) +  "/" + str(data.get_size()) + ")"
            annotated_data.append(self.annotate_datum(data.get(i)))
        return mung.data.DataSet(data=annotated_data)

    def annotate_datum(self, datum, in_place=False):
        if not in_place:
            return self.annotate_datum(datum.get_mutable(), in_place=True)
        else:
            return self._annotate_in_place(datum)

    @abc.abstractmethod
    def __str__(self):
        """ Return a string description of the annotator """

    @abc.abstractmethod
    def _annotate_in_place(self, datum):
        """ Annotate a given datum """


class Annotation(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, target_ref):
        self._target_ref = target_ref

    def get_target(self):
        return self._target_ref.get()

    @abc.abstractmethod
    def get_type(self):
        """ Get type of the annotation """


class Tokens(Annotation):
    def __init__(self, target_ref, spans=[]):
        Annotation.__init__(self, target_ref)

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
        obj["target"] = self._target_ref.get_path()
        obj["spans"] = [[span[0], span[1]] for span in self._spans]
        return obj

    @staticmethod
    def from_dict(datum, obj):
        if not isinstance(obj, dict) or "type" not in obj or obj["type"] != TYPE_TOKENS:
            return None

        target_ref = mung.data.DatumReference(datum, obj["target"])
        spans = [(span[0], span[1]) for span in obj["spans"]]

        return Tokens(target_ref, spans=spans)


class TokensAnnotation(Annotation):
    __metaclass__ = abc.ABCMeta

    def __init__(self, target_ref):
        Annotation.__init__(self, target_ref)
        self._tokens = self.get_target()

    def get_size(self):
        return self.get_target().get_size()

    def get_span(self, index):
        return self.get_target().get_span(index)

    @abc.abstractmethod
    def get_type(self):
        """ Get type of the annotation """


class PoS(TokensAnnotation):
    def __init__(self, target_ref, pos):
        TokensAnnotation.__init__(self, target_ref)
        self._pos = pos

    def get_type(self):
        return TYPE_POS

    def get(self, index):
        return self._pos[index]

    def to_dict(self):
        obj = dict()
        obj["type"] = TYPE_POS
        obj["target"] = self._target_ref.get_path()
        obj["pos"] = self._pos
        return obj

    @staticmethod
    def from_dict(datum, obj):
        if not isinstance(obj, dict) or "type" not in obj or obj["type"] != TYPE_POS:
            return None

        target_ref = mung.data.DatumReference(datum, obj["target"])
        pos = obj["pos"]

        return PoS(target_ref, pos)


class Lemmas(TokensAnnotation):
    def __init__(self, target_ref, lemmas):
        TokensAnnotation.__init__(self, target_ref)
        self._lemmas = lemmas

    def get_type(self):
        return TYPE_LEMMAS

    def get(self, index):
        return self._lemmas[index]

    def to_dict(self):
        obj = dict()
        obj["type"] = TYPE_LEMMAS
        obj["target"] = self._target_ref.get_path()
        obj["lemmas"] = self._lemmas
        return obj

    @staticmethod
    def from_dict(datum, obj):
        if not isinstance(obj, dict) or "type" not in obj or obj["type"] != TYPE_LEMMAS:
            return obj

        target_ref = mung.data.DatumReference(datum, obj["target"])
        lemmas = obj["lemmas"]

        return Lemmas(target_ref, lemmas)

class Strings(TokensAnnotation):
    def __init__(self, target_ref, strs):
        TokensAnnotation.__init__(self, target_ref)
        self._strs = strs

    def get_type(self):
        return TYPE_STRINGS

    def get(self, index):
        return self._strs[index]

    def to_dict(self):
        obj = dict()
        obj["type"] = TYPE_STRINGS
        obj["target"] = self._target_ref.get_path()
        obj["strs"] = self._strs
        return obj

    @staticmethod
    def from_dict(datum, obj):
        if not isinstance(obj, dict) or "type" not in obj or obj["type"] != TYPE_STRINGS:
            return obj

        target_ref = mung.data.DatumReference(datum, obj["target"])
        strs = obj["strs"]

        return Strings(target_ref, strs)

class Sentences(Annotation):
    def __init__(self, target_ref, token_spans):
        Annotation.__init__(self, target_ref)
        self._token_spans = token_spans

    def get_type(self):
        return TYPE_SENTENCES

    def get_size(self):
        return len(self._token_spans)

    def get(self, index):
        return self._token_spans[index]

    def to_dict(self):
        obj = dict()
        obj["type"] = TYPE_SENTENCES
        obj["target"] = self._target_ref.get_path()
        obj["token_spans"] = [[span[0], span[1]] for span in self._token_spans]
        return obj

    @staticmethod
    def from_dict(datum, obj):
        if not isinstance(obj, dict) or "type" not in obj or obj["type"] != TYPE_SENTENCES:
            return None

        target_ref = mung.data.DatumReference(datum, obj["target"])
        spans = [(span[0], span[1]) for span in obj["token_spans"]]

        return Sentences(target_ref, token_spans)
