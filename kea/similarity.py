
import logging
import os
import cv2

THREGHOLD = 0.85

import xml.etree.ElementTree as ET


class SimpleNode:
    def __init__(self, tag):
        self.tag = tag
        self.children = []


def build_simple_tree(elem):
    node = SimpleNode(elem.tag)
    for child in elem:
        node.children.append(build_simple_tree(child))
    return node


def compare_simple_trees(n1, n2):
    if n1 is None and n2 is None:
        return (0, 0)
    if n1 is None or n2 is None:
        return (0, 1)

    score = 0
    total = 1

    if n1.tag == n2.tag:
        score += 1

    children1 = n1.children
    children2 = n2.children
    for c1, c2 in zip(children1, children2):
        s, t = compare_simple_trees(c1, c2)
        score += s
        total += t

    total += abs(len(children1) - len(children2))
    return (score, total)


def compare_xml_strings(xml_str1, xml_str2):
    try:
        root1 = ET.fromstring(xml_str1)
        root2 = ET.fromstring(xml_str2)
    except ET.ParseError as e:
        raise ValueError(f"XML Parse Error: {e}")

    tree1 = build_simple_tree(root1)
    tree2 = build_simple_tree(root2)
    score, total = compare_simple_trees(tree1, tree2)

    if total == 0:
        return 100.0
    similarity = (score / total) * 100
    return round(similarity, 2)

class Similarity(object):
    def __init__(self, sim_k) -> None:
        self.sim_k: int = sim_k
        self.sim_count = 0
        self.logger = logging.getLogger('SimilarityCalculator')
        self.cache: list[str] = []
        

    def detected_ui_tarpit(self,input_manager):
        """
        start calculate similarity between last state screen and current screen
        """
        last_state = input_manager.policy.get_last_state()
        last_state_screen = last_state.get_state_screen()
        current_state = input_manager.device.get_current_state()
        current_state_screen = current_state.get_state_screen()
        sim_score = self.calculate_similarity(last_state_screen,current_state_screen)
        self.logger.info(f'similarity score:{sim_score}')
        if sim_score < THREGHOLD :
            self.logger.info(f'different page!')
            self.sim_count = 0
            input_manager.policy.clear_action_history()
        else:
            self.sim_count += 1   
        if self.sim_count >= self.sim_k :
            return True
        return False

    def detected_ui_tarpit_modified(self, input_manager):
        """
        start calculate similarity between last state screen and current screen
        """
        last_state = input_manager.policy.get_last_state()
        last_state_screen = last_state.get_state_screen()
        current_state = input_manager.device.get_current_state()
        current_state_screen = current_state.get_state_screen()
        sim_score = self.calculate_similarity(last_state_screen,current_state_screen)
        self.logger.info(f'similarity score:{sim_score}')
        if sim_score < 0.9:
            self.logger.info(f'different page!')
            self.sim_count = 0
            input_manager.policy.clear_action_history()
        else:
            self.sim_count += 1
        if self.sim_count >= 6:
            return True
        return False

    def detect(self, xml1: str, xml2: str):
        for i in range(-len(self.cache), 0):
            xml = self.cache[i]
            res = compare_xml_strings(xml1, xml)
            self.logger.info(f"Similarity score({i}): {res}; sim_count: {self.sim_count} ")
            if res > 90:
                self.sim_count += 1
                if self.sim_count >= 3:
                    self.sim_count = 0
                    if len(self.cache) >= 3:
                        self.cache.pop(0)
                    self.cache.append(xml1)
                    return True
                return False
        self.sim_count = 0
        if len(self.cache) >= 3:
            self.cache.pop(0)
        self.cache.append(xml1)
        return False
    
    @staticmethod
    def dhash(image, hash_size=8):
        resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        diff = gray[:, 1:] > gray[:, :-1]

        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

    @staticmethod
    def hamming_distance(hash1, hash2):
        return bin(hash1 ^ hash2).count("1")

    @staticmethod
    def calculate_similarity(fileA, fileB):
        imgA = cv2.imread(fileA)
        imgB = cv2.imread(fileB)
        hashA = Similarity.dhash(imgA)
        hashB = Similarity.dhash(imgB)
        similarity_score = 1 - Similarity.hamming_distance(hashA, hashB) / 64.0 
        return similarity_score     