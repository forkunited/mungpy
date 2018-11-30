from mung.data import DatumReference
from mung.nlp.annotation import Tokens, Strings, PoS, Lemmas, Sentences, \
                                Paragraphs, \
                                MorphologicalProperties, Morphology, \
                                DependencyTree, Dependencies, Annotator
from ufal.udpipe import Model, Pipeline, ProcessingError

import time

class UDPipeAnnotations:
    KEY_TOKENS = "tokens"
    KEY_POS = "pos"
    KEY_LEMMAS = "lemmas"
    KEY_SENTENCES = "sents"
    KEY_PARAGRAPHS = "paragraphs"
    KEY_DEPENDENCIES = "deps"
    KEY_MORPHOLOGY = "morph"
    KEY_TOKEN_STRINGS = "token_strs"

    def __init__(self, tokens=None, pos=None, lemmas=None, sentences=None, paragraphs=None, dependencies=None, morphology=None, token_strs=None):
        self._tokens = tokens
        self._pos = pos
        self._lemmas = lemmas
        self._sentences = sentences
        self._paragraphs = paragraphs
        self._dependencies = dependencies
        self._morphology = morphology
        self._token_strs = token_strs

    def to_dict(self):
        obj = dict()

        if self._tokens is not None:
           obj[self.KEY_TOKENS] = self._tokens.to_dict()
        if self._pos is not None:
            obj[self.KEY_POS] = self._pos.to_dict()
        if self._lemmas is not None:
            obj[self.KEY_LEMMAS] = self._lemmas.to_dict()
        if self._sentences is not None:
            obj[self.KEY_SENTENCES] = self._sentences.to_dict()
        if self._paragraphs is not None:
            obj[self.KEY_PARAGRAPHS] = self._paragraphs.to_dict()
        if self._dependencies is not None:
            obj[self.KEY_DEPENDENCIES] = self._dependencies.to_dict()
        if self._morphology is not None:
            obj[self.KEY_MORPHOLOGY] = self._morphology.to_dict()
        if self._token_strs is not None:
            obj[self.KEY_TOKEN_STRINGS] = self._token_strs.to_dict()

        return obj

    @staticmethod
    def from_dict(datum, obj):
        if not isinstance(obj, dict):
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

        paragraphs = None
        if self.KEY_PARAGRAPHS in obj:
            paragraphs = Paragraphs.from_dict(datum, obj[self.KEY_PARAGRAPHS])

        dependencies = None
        if self.KEY_DEPENDENCIES in obj:
            dependencies = Dependencies.from_dict(datum, obj[self.KEY_DEPENDENCIES])

        morphology = None
        if self.KEY_MORPHOLOGY in obj:
            morphology = Morphology.from_dict(datum, obj[self.KEY_MORPHOLOGY])

        token_strs = None
        if self.KEY_TOKEN_STRINGS in obj:
            token_strs = Strings.from_dict(datum, obj[self.KEY_TOKEN_STRINGS])

        return CoreNLPAnnotations(tokens=tokens, pos=pos, lemmas=lemmas, sentences=sentences, paragraphs=paragraphs, \
                                  dependencies=dependencies, morphology=morphology, token_strs=token_strs)


class UDPipeAnnotator(Annotator):
    def __init__(self, model_path, target_key, store_key, target_path=None, store_path=None):
        Annotator.__init__(self)
        
        # Note that UDPipe segfaults if this is not kept in memory
        self._model = Model.load(model_path)

        self._pipeline = Pipeline(self._model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        self._target_path = target_path
        self._store_path = store_path
        self._target_key = target_key
        self._store_key = store_key

    def __str__(self):
        return "udpipe"

    def _annotate_in_place(self, datum):
        targets = None
        if self._target_path is not None:
            targets = datum.get(self._target_path, first=False, include_paths=True)
        else:
            targets = [(".", datum.to_dict())]
        for (target_path, target) in targets:
            text = target[self._target_key]
            annos = self._annotate_text(datum, text, target_path)
            obj = annos.to_dict()
            datum.set(self._store_key, obj, path=self._store_path)
        return datum

    # Borrowed from https://github.com/ufal/udpipe/blob/master/bindings/python/examples/run_udpipe.py
    def _annotate_text(self, datum, text, target_path):
        text_ref = DatumReference(datum, target_path + "." + self._target_key)
        tokens_ref = DatumReference(datum, target_path + "." + self._store_key + "." + UDPipeAnnotations.KEY_TOKENS)
        sentences_ref = DatumReference(datum, target_path + "." + self._store_key + "." + UDPipeAnnotations.KEY_SENTENCES)
        if text.strip() == '':
            tokens = Tokens(text_ref, spans=[])
            pos = PoS(tokens_ref, [])
            lemmas = Lemmas(tokens_ref, [])
            sentences = Sentences(tokens_ref, [])
            paragraphs = Paragraphs(tokens_ref, [])
            dependencies = Dependencies(sentences_ref, [])
            morphology = Morphology(tokens_ref, [])
            token_strs = Strings(tokens_ref, [])
            return UDPipeAnnotations(tokens=tokens,pos=pos,lemmas=lemmas,sentences=sentences,\
                                     paragraphs=paragraphs, dependencies=dependencies, morphology=morphology, \
                                     token_strs=token_strs)

        error = ProcessingError()
        processed = self._pipeline.process(text, error)
        if error.occurred():
            raise ValueError("An error occurred when running run_udpipe: " + error.message + "\n")

        processed_dict = self._parse_conllu(processed, text_ref, tokens_ref, sentences_ref)
        return UDPipeAnnotations(**processed_dict)

    def _parse_conllu(self, conllu, text_ref, tokens_ref, sentences_ref):
        token_char_spans = []
        pos_tags = []
        lemmas = []
        sentence_token_spans = []
        paragraph_token_spans = []
        dep_trees = []
        morphs = []
        token_strs = []

        conllu_lines = conllu.split("\n")

        cur_deps = []
        cur_sentence_start_token = 0
        cur_paragraph_start_token = 0
        cur_token = 0
        cur_char = 0
        cur_contracted_token_skips = 0 # Offset for dependencies due to contractions
        cur_contracted_token = None
        cur_contracted_token_indices = set()
        for line in conllu_lines:
            line_parts = line.split("\t")
            if len(line_parts) >= 10: # token line
                if not line_parts[0].isdigit():
                    token_range = line_parts[0].split('-')
                    cur_contracted_token_indices.add(token_range[0])
                    cur_contracted_token_indices.add(token_range[1])
                    cur_contracted_token_skips += int(token_range[1]) - int(token_range[0])
                    cur_contracted_token = line_parts[1]
                    continue
                
                token_str = line_parts[1]
                if cur_contracted_token is not None:
                    token_str = cur_contracted_token
                    cur_contracted_token = None
                elif line_parts[0] in cur_contracted_token_indices:
                    continue

                lemma = line_parts[2]
                pos = line_parts[3]
                morph_str = line_parts[5]
                dep_parent = int(line_parts[6])
                dep_type = line_parts[7]
                spaces_after = line_parts[9]

                token_char_spans.append((cur_char, cur_char + len(token_str)))
                pos_tags.append(pos)
                lemmas.append(lemma)
                cur_deps.append({ 'type' : dep_type, 'parent' : dep_parent - 1})
                morphs.append(MorphologicalProperties.from_conllu(morph_str))
                token_strs.append(token_str)

                num_spaces_after = 1
                if spaces_after.startswith('SpacesAfter='):
                    spaces_after = spaces_after.split('=')[1]
                    if spaces_after == 'No':
                        num_spaces_after = 0
                    else:
                        num_spaces_after = len(spaces_after.replace('\\', ''))
                
                cur_char += len(token_str) + num_spaces_after
                cur_token += 1
            elif line.startswith('# newpar') and cur_token > 0:
                paragraph_token_spans.append((cur_paragraph_start_token, cur_token))
                cur_paragraph_start_token = cur_token
            elif line.startswith('# sent_id') and cur_token > 0:
                sentence_token_spans.append((cur_sentence_start_token, cur_token))
                cur_sentence_start_token = cur_token

                # FIXME Subtract off token skips (the offset is not sufficient for this...)
                dep_trees.append(DependencyTree(cur_deps))
                cur_contracted_token_skips = 0
                cur_contracted_token_indices = set()
                cur_deps = []
        
        return {
          "tokens": Tokens(text_ref, spans=token_char_spans),
          "pos": PoS(tokens_ref, pos_tags),
          "lemmas": Lemmas(tokens_ref, lemmas),
          "sentences": Sentences(tokens_ref, sentence_token_spans),
          "paragraphs": Paragraphs(tokens_ref, paragraph_token_spans),
          "dependencies": Dependencies(sentences_ref, dep_trees),
          "morphology": Morphology(tokens_ref, morphs),
          "token_strs": Strings(tokens_ref, token_strs)
        }
