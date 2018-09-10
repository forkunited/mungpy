import abc
import mung.data
from mung.data import DatumReference, DataSet, Datum
from mung.feature import DataFeatureMatrix, MultiviewDataSet

TYPE_TOKENS = "NLPTokens"
TYPE_POS = "NLPPoS"
TYPE_LEMMAS = "NLPLemmas"
TYPE_STRINGS = "NLPStrings"
TYPE_SENTENCES = "NLPSentences"

class Annotator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        return

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

class ModelAnnotator(Annotator):
    def __init__(self, annotator_name, model, features, transform_datum_fn, store_key, target_path=None, label_fn=None, model_input_name="input"):
        Annotator.__init__(self)
        self._annotator_name = annotator_name
        self._model = model
        self._features = features

        self._transform_datum_fn = transform_datum_fn
        self._store_key = store_key
        self._target_path = target_path

        self._label_fn = label_fn
        self._model_input_name = model_input_name
        self._data_parameter = { model_input_name : model_input_name } # FIXME Refactor this later
        
    def __str__(self):
        return self._annotator_name

    def _annotate_in_place(self, datum):
        targets = None
        if self._target_path is not None:
            targets = datum.get(self._target_path, first=False, include_paths=True)
        else:
            targets = [(".", datum.to_dict())]
        
        for (target_path, target) in targets:
            transformed_target = self._transform_datum_fn(target)
            transformed_target["id"] = datum.get_id() + " _" + target_path
            transformed_datums = [Datum(properties=transformed_target)]

            annos = self._annotate_for_datums(datum, transformed_datums, target_path)
            obj = annos.to_dict()
            datum.set(self._store_key, obj, path=target_path)
        return datum

    def _annotate_for_datums(self, source_datum, datums, target_path):
        annotated_ref = DatumReference(source_datum, target_path)

        # FIXME This is grotesque.  Batch data later
        D = DataSet(data=datums) 
        DF = DataFeatureMatrix(D, self._features, init_features=False)
        M = MultiviewDataSet(data=D, dfmats={ self._model_input_name : DF })

        anno_obj = dict()
        anno_obj["prediction"] = self._model.predict_data(M, self._data_parameter).item()
        anno_obj["score"] = self._model.score_data(M, self._data_parameter).item()
        anno_obj["p"] = self._model.p_data(M, self._data_parameter)[0].tolist()

        if self._label_fn is not None:
            anno_obj["label"] = self._label_fn(anno_obj["prediction"])

        return AnnotationGeneric(annotated_ref, self._annotator_name, anno_obj)


class Annotation(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, target_ref):
        self._target_ref = target_ref

    def get_target(self):
        return self._target_ref.get()

    @abc.abstractmethod
    def get_type(self):
        """ Get type of the annotation """

class AnnotationGeneric(Annotation):
    def __init__(self, target_ref, anno_type, anno_dict):
        Annotation.__init__(self, target_ref)
        self._anno_dict = anno_dict
        self._anno_type = anno_type

    def get_type(self):
        return self._anno_type

    def to_dict(self):
        obj = dict()
        obj["type"] = self._anno_type
        obj["target"] = self._target_ref.get_path()
        obj["anno"] = self._anno_dict
        return obj

    @staticmethod
    def from_dict(datum, obj):
        if not isinstance(obj, dict) or "type" not in obj:
            return None
        
        target_ref = mung.data.DatumReference(datum, obj["target"])
        anno_type = obj["type"]
        anno_dict = obj["anno"]
        return AnnotationGeneric(target_ref, anno_type, anno_dict)

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
