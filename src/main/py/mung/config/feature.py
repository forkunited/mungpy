from mung.feature import MultiviewDataSet, SubsetType, PairedMultiviewDataSet
from mung.data import Partition

# Expects config of the form:
# {
#   data_path : [PATH TO DATA SET],
#   (Optional) paired_data_path : [PATH TO PAIRED DATA],
#   mats : {
#     dfmat_paths : [DICTIONARY OF DFMAT PATHS]
#     dfmatseq_paths : [DICTIONARY OF DFMATSEQ PATHS]
#     ordering_seq : [FEATURESEQ NAME TO ORDER BATCHES BY]
#   },
#   (optional) subsets : [
#       name : [NAME]
#       type : [RANDOM|FIRST|FILTER|PARTITION]
#       size : [SUBSET SIZE] (if RANDOM or FIRST)
#       file : [PARTITION FILE] (if PARTITION)
#       key : [PARTITION ID KEY] (if PARTITION)
#       parts : [DICTIONARY OF PART NAMES] (if PARTITION)
#       filter : [FILTER DICTIONARY] (if FILTER)
#       (optional) filter_type : Type of filter (EQUAL|NOT_EQUAL|GREATER|LESS) (Default: EQUAL)
#       (optional) superset : Name of the set to take a subset of
#     ]
#   }
# }
def load_mvdata(config, ignore_paired_data=True):
    data_path = config["data_path"]
    mats = config["mats"]
    D = MultiviewDataSet.load(data_path, **mats)

    D_paired = None
    if (not ignore_paired_data) and "paired_data_path" in config:
        D_paired = PairedMultiviewDataSet.load(config["paired_data_path"], D)

    S = dict()
    if "subsets" in config:
        for item in config["subsets"]:
            D_cur = D
            if "superset" in item:
                D_cur = S[item["superset"]]

            if item["type"] == SubsetType.RANDOM:
                S[item["name"]] = D_cur.get_random_subset(int(item["size"]))
            elif item["type"] == SubsetType.FIRST:
                S[item["name"]] = D_cur.get_subset(0, int(item["size"]))
            elif item["type"] == SubsetType.PARTITION:
                P = Partition.load(item["file"])
                id_key = "id"
                if "key" in item:
                    id_key = item["key"]
                
                merged_parts = dict()
                for k,v in item["parts"].iteritems():
                    if v not in merged_parts:
                        merged_parts[v] = []
                    merged_parts[v].append(k)

                for new_part_key, merged_keys in merged_parts.iteritems():
                    P.merge_parts(merged_keys, new_part_key)
                
                D_parts = D.partition(P, lambda d : d.get(id_key))
                for part_key, part in D_parts.iteritems():
                    if part_key in item["parts"]:
                        S[item["parts"][part_key]] = part
            else: # FILTER
                subset_filter = item["filter"]

                if "filter_type" not in item or item["filter_type"] == "EQUAL":
                    def f(d):
                        d_match = True
                        for d_key, d_value in subset_filter.iteritems():
                            d_match = (d_match and d.get(d_key) == d_value)
                        return d_match
                    S[item["name"]] = D_cur.filter(f)
                elif "filter_type" in item and item["filter_type"] == "NOT_EQUAL":
                    def f(d):
                        d_match = True
                        for d_key, d_value in subset_filter.iteritems():
                            d_match = (d_match and d.get(d_key) != d_value)
                        return d_match
                    S[item["name"]] = D_cur.filter(f)
                elif "filter_type" in item and item["filter_type"] == "GREATER":
                    def f(d):
                        d_match = True
                        for d_key, d_value in subset_filter.iteritems():
                            d_match = (d_match and d.get(d_key) > d_value)
                        return d_match
                    S[item["name"]] = D_cur.filter(f)
                elif "filter_type" in item and item["filter_type"] == "LESS":
                    def f(d):
                        d_match = True
                        for d_key, d_value in subset_filter.iteritems():
                            d_match = (d_match and d.get(d_key) < d_value)
                        return d_match
                    S[item["name"]] = D_cur.filter(f)
    return D, S, D_paired
