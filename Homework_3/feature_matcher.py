import cv2 as cv
from utils import imshow

class FeatureMatcher:

    def __init__(self):
        self.k_neighbours = 2
        self.FLANN_INDEX_KDTREE = 0
        self.index_trees = 5
        self.index_params = dict(algorithm = self.FLANN_INDEX_KDTREE, trees=self.index_trees)
        self.search_checks = 50
        self.search_params = dict(checks=self.search_checks)   # or pass empty dictionary
        self.matcher = cv.FlannBasedMatcher(self.index_params, self.search_params)

        self.lowe = 0.75
        self.min_match_count = 10

    def set_FLANN_INDEX_KDTREE(self, val):
        val = int(val) if val > 0 else 1
        self.FLANN_INDEX_KDTREE = val
        self.index_params = dict(algorithm = self.FLANN_INDEX_KDTREE, trees = val)
        self.matcher = cv.FlannBasedMatcher(self.index_params, self.search_params)

    def set_index_trees(self, trees):
        trees = int(trees) if trees > 0 else 1
        self.index_trees = trees
        self.index_params = dict(algorithm = self.FLANN_INDEX_KDTREE, trees = trees)
        self.matcher = cv.FlannBasedMatcher(self.index_params, self.search_params)

    def set_search_checks(self, checks):
        checks = int(checks) if checks > 0 else 1
        self.search_checks = checks
        self.search_params = dict(checks=checks)   # or pass empty dictionary
        self.matcher = cv.FlannBasedMatcher(self.index_params, self.search_params)

    def match(self, descr1, descr2, debug=False, img=None, kp_img=None, query=None, kp_query=None ):
        # print(descr1.shape)
        # print(descr2.shape)
        # print(self.index_params, self.search_params)
        matches = self.matcher.knnMatch(descr1, descr2, self.k_neighbours)
        if not debug:
            return matches
        else:
            assert (img is not None) and (kp_img is not None) and (query is not None) and (kp_query is not None)
            print(f'len(kp) = {len(kp_img)}, len(matches) = {len(matches)}')
            matches_mask = [[1,0] for i in range(len(matches))]
            draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matches_mask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
            img_matches = cv.drawMatchesKnn(query, kp_query, img, kp_img, matches, None, **draw_params)
            imshow('img_matches', img_matches)
            return matches

    def filter_matches(self, matches, debug=False, img=None, kp_img=None, query=None, kp_query=None):
        filtered_matches = []
        matches_mask = [[0,0] for i in range(len(matches))]
        for i,(m, n) in enumerate(matches):
            if m.distance < self.lowe*n.distance:
                matches_mask[i]=[1,0]
                filtered_matches.append(m)
        if not debug:
            return filtered_matches, matches_mask
        else:
            assert (img is not None) and (kp_img is not None) and (query is not None) and (kp_query is not None)
            draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matches_mask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
            img_matches = cv.drawMatchesKnn(query, kp_query, img, kp_img, matches, None, **draw_params)
            imshow('img_filtered_matches', img_matches)
            return filtered_matches, matches_mask