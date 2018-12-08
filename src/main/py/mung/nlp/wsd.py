from nltk.corpus import wordnet as wn

from mung.data import DatumReference
from mung.nlp.annotation import TokensAnnotation, Tokens, PoS, Annotator

class Synsets(TokensAnnotation):
    def __init__(self, target_ref, synset_lists):
        TokensAnnotation.__init__(self, target_ref)
        self._synset_lists = synset_lists

    def get_type(self):
        return "NLPSynsets"

    def get(self, index):
        return self._synset_lists[index]

    def to_dict(self):
        obj = dict()
        obj["type"] = "NLPSynsets"
        obj["target"] = self._target_ref.get_path()
        obj["synset_lists"] = self._synset_lists
        return obj

    @staticmethod
    def from_dict(datum, obj):
        if not isinstance(obj, dict) or "type" not in obj or obj["type"] != "NLPSynsets":
            return obj

        target_ref = mung.data.DatumReference(datum, obj["target"])
        synset_lists = obj["synset_lists"]

        return Synsets(target_ref, synset_lists)

class WordNetAnnotator(Annotator):
    def __init__(self, target_key, store_key, language_key=None, pos_key=None, target_path=None, store_path=None):
        Annotator.__init__(self)
        self._target_path = target_path
        self._store_path = store_path
        self._target_key = target_key
        self._store_key = store_key
        self._language_key = language_key
        self._pos_key = pos_key
        self._available_langs = set(wn.langs())

    def __str__(self):
        return "wordnet"

    def _annotate_in_place(self, datum):
        targets = None
        if self._target_path is not None:
            targets = datum.get(self._target_path, first=False, include_paths=True)
        else:
            targets = [(".", datum.to_dict())]
        for (target_path, target) in targets:
            tokens = Tokens.from_dict(datum, target[self._target_key])
            pos_tags = None
            if self._pos_key is not None:
                pos_tags = PoS.from_dict(datum, target[self._pos_key])
            annos = self._annotate_tokens(datum, tokens, pos_tags, target_path)
            obj = annos.to_dict()
            datum.set(self._store_key, obj, path=self._store_path)
        return datum

    def _annotate_tokens(self, datum, tokens, pos_tags, target_path):
        tokens_ref = DatumReference(datum, target_path + "." + self._target_key)
        if tokens.get_size() == 0:
            return Synsets(tokens_ref, [])

        lang = 'eng'
        if self._language_key is not None:
            lang = datum.get(self._language_key)
            if len(lang) == 2:
                if lang == 'en':
                    lang = 'eng'
                elif lang == 'es':
                    lang = 'spa'
                elif lang == 'fr':
                    lang = 'fra'
                elif lang == 'it':
                    lang = 'ita'

        synsets = []
        for i in range(tokens.get_size()):
            token = tokens.get(i)
            
            pos = None
            if pos_tags is not None:
                pos_str = pos_tags.get(i)
                if pos_str == 'VERB':
                    pos = wn.VERB
                elif pos_str == 'NOUN':
                    pos = wn.NOUN
                elif pos_str == 'ADJ':
                    pos = wn.ADJ
                elif pos_str == 'ADV':
                    pos = wn.ADV

            if lang not in self._available_langs:
                synsets.append([])
            else:
                synsets.append([synset.name() for synset in wn.synsets(token, pos=pos, lang=lang)])

        return Synsets(tokens_ref, synsets)
