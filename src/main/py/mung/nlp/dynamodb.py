from mung.data import DatumReference, Datum
from mung.nlp.annotation import GenericTokensAnnotation, Tokens, Annotator
from boto3.dynamodb.types import TypeSerializer, TypeDeserializer
from decimal import Decimal

QUERY_BATCH_SIZE = 50

class DynamoDBAnnotator(Annotator):
    def __init__(self, annotation_type, \
                 dynamodb_client, dynamodb_table, dynamodb_key, dynamodb_token_field, \
                 target_key, store_key, language_key, token_clean_fn, db_annotation_path, \
                 target_path=None, store_path=None):
        Annotator.__init__(self)
        self._annotation_type = annotation_type

        self._dynamodb_client = dynamodb_client
        self._dynamodb_table = dynamodb_table
        self._dynamodb_key = dynamodb_key
        self._dynamodb_token_field = dynamodb_token_field
        
        self._target_path = target_path
        self._store_path = store_path
        self._target_key = target_key
        self._store_key = store_key
        self._language_key = language_key
        self._token_clean_fn = token_clean_fn
        self._db_annotation_path = db_annotation_path

        self._type_serializer = TypeSerializer()
        self._type_deserializer = TypeDeserializer()

    def __str__(self):
        return 'dynamodb'

    def _annotate_in_place(self, datum):
        targets = None
        if self._target_path is not None:
            targets = datum.get(self._target_path, first=False, include_paths=True)
        else:
            targets = [(".", datum.to_dict())]
        for (target_path, target) in targets:
            tokens = Tokens.from_dict(datum, target[self._target_key])
            annos = self._annotate_tokens(datum, tokens, target_path)
            obj = annos.to_dict()
            datum.set(self._store_key, obj, path=self._store_path)
        return datum

    def _annotate_tokens(self, datum, tokens, target_path):
        tokens_ref = DatumReference(datum, target_path + "." + self._target_key)
        if tokens.get_size() == 0:
            return GenericTokensAnnotation(self._annotation_type, tokens_ref, [])

        language = datum.get(self._language_key)
        token_keys = []
        for i in range(tokens.get_size()):
            token = self._token_clean_fn(tokens.get(i))
            token_lang = token + '_' + language
            token_keys.append(token_lang)

        objs = self._query_token_annotations(token_keys, language)
        return GenericTokensAnnotation(self._annotation_type, tokens_ref, objs)

    def _query_token_annotations(self, token_keys, language):
        objs = []
        for i in range(0, len(token_keys), QUERY_BATCH_SIZE):
            token_key_batch = token_keys[i:(i+QUERY_BATCH_SIZE)]
            objs.extend(self._query_token_annotations_batch(token_key_batch, language))
        return objs

    # FIXME At some point, may need to make this faster.
    def _query_token_annotations_batch(self, token_key_batch, language):
        token_key_batch_reverse = dict()
        for i, token_key in enumerate(token_key_batch):
            if token_key not in token_key_batch_reverse:
                token_key_batch_reverse[token_key] = []
            token_key_batch_reverse[token_key].append(i)
        token_key_batch_distinct = [{ self._dynamodb_key : self._type_serializer.serialize(token_key) } \
                                    for token_key in token_key_batch_reverse.keys()]

        result = self._dynamodb_client.batch_get_item(RequestItems={ self._dynamodb_table: { 'Keys': token_key_batch_distinct }})
        items = result['Responses'][self._dynamodb_table]
        
        objs_batch = [{} for i in range(len(token_key_batch))]
        for item in items:
            item = self._item_to_json(item)
            item_datum = Datum(properties=item)
            annotation_result = item_datum.get(self._db_annotation_path)
            item_indices = token_key_batch_reverse[item[self._dynamodb_token_field] + '_' + language]
            for index in item_indices:
                objs_batch[index] = annotation_result

        return objs_batch

    def _item_to_json(self, dynamo_item):
        return self._item_to_json_helper(
            self._type_deserializer.deserialize({ "M" : dynamo_item })
        )

    def _item_to_json_helper(self, dynamo_item):
        if isinstance(dynamo_item, list):
            return [self._item_to_json_helper(dynamo_item[i]) for i in range(len(dynamo_item))]
        elif isinstance(dynamo_item, dict):
            json_item = dict()
            for k, v in dynamo_item.iteritems():
                json_item[k] = self._item_to_json_helper(v)
            return json_item
        elif isinstance(dynamo_item, Decimal):
            if dynamo_item % 1 > 0:
                return float(dynamo_item)
            else:
                return int(dynamo_item)
        else:
            return dynamo_item
