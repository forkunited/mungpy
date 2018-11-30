from nltk.corpus import wordnet as wn

from mung.data import DatumReference
from mung.nlp.annotation import TokensAnnotation, Annotator

class Synsets(TokensAnnotation):
    def __init__(self, target_ref, synset_lists):
        TokensAnnotation.__init__(self, target_ref)
        self._synset_lists = synset_lists

    def get_type(self):
        return "synsets"

    def get(self, index):
        return self._synset_lists[index]

    def to_dict(self):
        obj = dict()
        obj["type"] = "synsets"
        obj["target"] = self._target_ref.get_path()
        obj["synset_lists"] = self._synset_lists
        return obj

    @staticmethod
    def from_dict(datum, obj):
        if not isinstance(obj, dict) or "type" not in obj or obj["type"] != "synsets":
            return obj

        target_ref = mung.data.DatumReference(datum, obj["target"])
        synset_lists = obj["synset_lists"]

        return Synsets(target_ref, synset_lists)

class WordNetAnnotator(Annotator):
    def __init__(self, target_key, store_key, language_key=None, target_path=None):
        Annotator.__init__(self)
        self._target_path = target_path
        self._target_key = target_key
        self._store_key = store_key
        self._language_key = language_key
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
            tokens = Tokens.from_dict(target[self._target_key])
            annos = self._annotate_text(datum, tokens, target_path)
            obj = annos.to_dict()
            datum.set(self._store_path, obj, path=target_path)
        return datum

    def _annotate_tokens(self, datum, tokens, target_path):
        tokens_ref = DatumReference(datum, target_path + "." + self._target_key)
         if text.strip() == '':
            return Synsets(tokens_ref, [])

        synsets = []
        for i in range(tokens.get_size()):
            token = tokens.get(i)
            if language_key is not None:
                lang = datum.get(language_key)
                if len(lang) == 2:
                    if lang == 'en':
                        lang = 'eng'
                    elif lang == 'es':
                        lang = 'spa'
                    elif lang == 'fr':
                        lang = 'fra'
                    elif lang == 'it':
                        lang = 'ita'
            else:
                lang = 'eng'
            if lang not in self._available_langs:
                synsets.append([])
            else:
                synsets.append([synset.name() for synset in wn.synsets(token, lang=lang)])

        return Synsets(tokens_ref, synsets)
