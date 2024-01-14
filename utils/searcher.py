import difflib
import Levenshtein
import numpy as np

class LexiconSearcher:
    def __init__(self, lexicon_path, matching="lcs"):
        try :
            assert matching in ["lcs", "edit_distance"]
        except:
            raise NotImplementedError("Matching method should be \"lcs\" or \"edit_distance\"")
        self.poi_names = []
        with open(lexicon_path, 'r') as f:
            pois = f.readlines()
            for poi in pois:
                self.poi_names.append(poi.strip())
        self.slots = ["poi名称", "poi修饰", "poi目标", "起点名称","起点修饰", "起点目标", "终点名称", "终点修饰", "终点目标", "途经点名称"]
        self.matching = matching

    def search(self, slot, value):
        if self.matching == "lcs":
            if slot not in self.slots:
                return value
            else:
                return difflib.get_close_matches(value, self.poi_names, n=1, cutoff=0)[0]
        if self.matching == "edit_distance":
            if slot not in self.slots:
                return value
            else:
                distances = [Levenshtein.distance(value, poi) for poi in self.poi_names]
                return self.poi_names[np.argmin(distances)]
if __name__ == "__main__":
    # For Debugging
    lexicon_searcher = LexiconSearcher("/mnt/workspace/Zizheng/Project/data/lexicon/poi_name.txt")
    print(lexicon_searcher.search("poi名称", "哈尔滨医科大学附属"))

