from pycorenlp import StanfordCoreNLP
from mung.data import DatumReference
from mung.nlp.annotation import Tokens, PoS, Lemmas, Sentences, Strings, Annotator

TYPE_ANNOTATIONS = "CoreNLPAnnotations"

STANFORD_NLP_PORT = 9000

class CoreNLPAnnotations:
    KEY_TYPE = "type"
    KEY_TOKENS = "tokens"
    KEY_POS = "pos"
    KEY_LEMMAS = "lemmas"
    KEY_SENTENCES = "sents"
    KEY_CLEAN_STRINGS = "clean_strs"
    KEY_TOKEN_STRINGS = "token_strs"

    def __init__(self, tokens=None, pos=None, lemmas=None, sentences=None, clean_strs=None, token_strs=None):
        self._tokens = tokens
        self._pos = pos
        self._lemmas = lemmas
        self._sentences = sentences
        self._clean_strs = clean_strs
        self._token_strs = token_strs

    def to_dict(self):
        obj = dict()
        obj[self.KEY_TYPE] = TYPE_ANNOTATIONS

        if self._tokens is not None:
           obj[self.KEY_TOKENS] = self._tokens.to_dict()
        if self._pos is not None:
            obj[self.KEY_POS] = self._pos.to_dict()
        if self._lemmas is not None:
            obj[self.KEY_LEMMAS] = self._lemmas.to_dict()
        if self._sentences is not None:
            obj[self.KEY_SENTENCES] = self._sentences.to_dict()
        if self._clean_strs is not None:
            obj[self.KEY_CLEAN_STRINGS] = self._clean_strs.to_dict()
        if self._token_strs is not None:
            obj[self.KEY_TOKEN_STRINGS] = self._token_strs.to_dict()

        return obj

    @staticmethod
    def from_dict(datum, obj):
        if not isinstance(obj, dict) or self.KEY_TYPE not in obj or obj[self.KEY_TYPE] != TYPE_ANNOTATIONS:
            return None

        tokens = None
        if self.KEY_TOKENS in obj:
            tokens = Tokens.from_dict(datum, obj[self.KEY_TOKENS])

        pos = None
        if self.KEY_POS in obj:
            pos = PoS.from_dict(datum, obj[self.KEY_POS])

        lemmas = None
        if self.KEY_LEMMAS in obj:
            lemmas = Lemmas.from_dict(datum, obj[self.KEY_LEMMAS])

        sentences = None
        if self.KEY_SENTENCES in obj:
            sentences = Sentences.from_dict(datum, obj[self.KEY_SENTENCES])

        clean_strs = None
        if self.KEY_CLEAN_STRINGS in obj:
            clean_strs = Strings.from_dict(datum, obj[self.KEY_CLEAN_STRINGS])

        token_strs = None
        if self.KEY_TOKEN_STRINGS in obj:
            token_strs = Strings.from_dict(datum, obj[self.KEY_TOKEN_STRINGS])

        return CoreNLPAnnotations(tokens=tokens, pos=pos, lemmas=lemmas, sentences=sentences, clean_strs=clean_strs, token_strs=token_strs)


class CoreNLPAnnotator(Annotator):
    def __init__(self, target_path, target_key, store_key):
        Annotator.__init__(self, target_path, target_key, store_key)
        self._nlp = StanfordCoreNLP('http://localhost:{}'.format(STANFORD_NLP_PORT))

    def __str__(self):
        return "corenlp-3.6+"

    def _annotate_in_place(self, datum):
        targets = datum.get(self._target_path, first=False, include_paths=True)
        for (target_path, target) in targets:
            text = str(target[self._target_key])
            annos = self._annotate_text(datum, text, target_path)
            obj = annos.to_dict()
            datum.set(self._store_key, obj, path=target_path)
        return datum

    # Borrowed from https://github.com/futurulus/coop-nets/blob/master/behavioralAnalysis/tagPOS.ipynb
    def _annotate_text(self, datum, text, target_path):
        try:
            text_ref = DatumReference(datum, target_path + "." + self._target_key)
            tokens_ref = DatumReference(datum, target_path + "." + self._store_key + "." + CoreNLPAnnotations.KEY_TOKENS)
            if text.strip() == '':
                tokens = Tokens(text_ref, spans=[])
                pos = PoS(tokens_ref, [])
                lemmas = Lemmas(tokens_ref, [])
                sentences = Sentences(tokens_ref, [])
                clean_strs = Strings(tokens_ref, [])
                return CoreNLPAnnotations(tokens=tokens,pos=pos,lemmas=lemmas,sentences=sentences,clean_strs=clean_strs)

            ann = self._nlp.annotate(
                text,
                properties={'annotators': 'pos,lemma',
                            'outputFormat': 'json'})
            sent_spans = []
            spans = []
            lemma_strs = []
            pos_strs = []
            clean_strs = []
            token_strs = []
            if isinstance(ann, basestring):
                ann = json.loads(ann.replace('\x00', '?').encode('latin-1'), encoding='utf-8', strict=True)

            token_index = 0
            for sentence in ann['sentences']:
                for token in sentence['tokens']:
                    spans.append((token["characterOffsetBegin"], token["characterOffsetEnd"]))
                    lemma_strs.append(token['lemma'])
                    pos_strs.append(token['pos'])

                    token_text = text[token["characterOffsetBegin"]:token["characterOffsetEnd"]]
                    token_strs.append(token_text)

                    token_text = token_text.lower()
                    if token_text.endswith("er"):
                        clean_strs.append(token_text[:-2])
                        clean_strs.append("-er")
                    elif token_text.endswith("est"):
                        clean_strs.append(token_text[:-3])
                        clean_strs.append("-est")
                    elif token_text.endswith("ish"):
                        clean_strs.append(token_text[:-3])
                        clean_strs.append("-ish")
                    else:
                        clean_strs.append(token_text)

                sent_spans.append((token_index, token_index + len(sentence['tokens'])))
                token_index += len(sentence['tokens'])

            tokens = Tokens(text_ref, spans=spans)
            pos = PoS(tokens_ref, pos_strs)
            lemmas = Lemmas(tokens_ref, lemma_strs)
            sentences = Sentences(tokens_ref, sent_spans)
            clean_strs = Strings(tokens_ref, clean_strs)
            token_strs = Strings(tokens_ref, token_strs)

            return CoreNLPAnnotations(tokens=tokens, pos=pos, lemmas=lemmas, sentences=sentences, clean_strs=clean_strs, token_strs=token_strs)
        except Exception as e:
            raise
