from pycorenlp import StanfordCoreNLP

STANFORD_NLP_PORT = 9000

class CoreNLPAnnotator(Annotator)
    def __init__(self, target_path, target_key, store_key):
        Annotator.__init__(self, target_path, target_key, store_key)
        self._nlp = StanfordCoreNLP('http://localhost:{}'.format(STANFORD_NLP_PORT))

    def __str__(self):
        return "corenlp-3.6+"

    def _annotate_in_place(self, datum):
        targets = datum.get(self._target_path, first=False, include_paths=True)
        for (target_path, target) in targets:
            text = target[self._target_key]
            annos = self._annotate_text(text)        
            obj = Annotations(tokens=annos["words"], pos=annos["pos"], lemmas=annos["lemmas"]).to_dict()
            datum.set(self._store_key, obj, path=target_path)
            return datum

    # Borrowed from https://github.com/futurulus/coop-nets/blob/master/behavioralAnalysis/tagPOS.ipynb
    def _annotate_text(self, text):
        try:
            if text.strip() == '':
                anno_obj = dict()
                anno_obj["words"] = []
                anno_obj["lemmas"] = []
                anno_obj["pos"] = []
                return anno_obj

            #text = str(text)
            ann = nlp.annotate(
                text,
                properties={'annotators': 'pos,lemma',
                            'outputFormat': 'json'})
            words = []
            lemmas = []
            pos = []
            if isinstance(ann, basestring):
                ann = json.loads(ann.replace('\x00', '?').encode('latin-1'), encoding='utf-8', strict=True)
            for sentence in ann['sentences']:
                s_words = []
                s_lemmas = []
                s_pos = []
                for token in sentence['tokens']:
                    s_words.append(token['word'])
                    s_lemmas.append(token['lemma'])
                    s_pos.append(token['pos'])
                words.append(s_words)
                lemmas.append(s_lemmas)
                pos.append(s_pos)

            anno_obj = dict()
            anno_obj["words"] = words
            anno_obj["lemmas"] = lemmas
            anno_obj["pos"] = pos

            return anno_obj
        except Exception as e:
            raise
