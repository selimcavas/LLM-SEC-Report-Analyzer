import re
import unicodedata
from bs4 import BeautifulSoup as bs
import requests
import sys

count = 0

def parse_10Q_filing(text, section):
    
    if section not in [0, 1, 2, 3, 4]:
        print("Not a valid section")
        sys.exit()

    def extract_text(text, item_start, item_end):
        item_start = item_start
        item_end = item_end
        starts = [i.start() for i in item_start.finditer(text)]
        ends = [i.start() for i in item_end.finditer(text)]
        positions = list()
        for s in starts:
            control = 0
            for e in ends:
                if control == 0:
                    if s < e:
                        control = 1
                        positions.append([s,e])
        item_length = 0
        item_position = list()
        for p in positions:
            if (p[1]-p[0]) > item_length:
                item_length = p[1]-p[0]
                item_position = p

        item_text = text[item_position[0]:item_position[1]]

        return(item_text)

        
    if section == 1 or section == 0:
        try:
            item1_start = re.compile("item\s*[1][\.\;\:\-\_]*\s*\\b", re.IGNORECASE)
            item1_end = re.compile("item\s*1a[\.\;\:\-\_]\s*Risk|item\s*2[\.\,\;\:\-\_]\s*Prop", re.IGNORECASE)
            businessText = extract_text(text, item1_start, item1_end)
        except Exception as e:
            raise Exception("Failed to extract text: " + str(e))
        
    if section == 2 or section == 0:
        try:
            item1a_start = re.compile("(?<!,\s)item\s*1a[\.\;\:\-\_]\s*Risk", re.IGNORECASE)
            item1a_end = re.compile("item\s*2[\.\;\:\-\_]\s*Prop|item\s*[1][\.\;\:\-\_]*\s*\\b", re.IGNORECASE)
            riskText = extract_text(text, item1a_start, item1a_end)
        except Exception as e:
            raise Exception("Failed to extract text: " + str(e))


            
    if section == 3 or section == 0:
        try:
            item7_start = re.compile("item\s*[7][\.\;\:\-\_]*\s*\\bM", re.IGNORECASE)
            item7_end = re.compile("item\s*7a[\.\;\:\-\_]\sQuanti|item\s*8[\.\,\;\:\-\_]\s*", re.IGNORECASE)
            mdaText = extract_text(text, item7_start, item7_end)
        except Exception as e:
            mdaText = "something went wrong"

            
    if section == 4 or section == 0:
        try:
            item2_start = re.compile("item\s*[2][\.\;\:\-\_]*\s*\\bM", re.IGNORECASE)
            item3_end = re.compile("item\s*3[\.\;\:\-\_]\sQuanti|item\s*3[\.\,\;\:\-\_]\s*", re.IGNORECASE)
            mdaText = extract_text(text, item2_start, item3_end)
        except Exception as e:
            mdaText = "something went wrong"
    
    if section == 0:
        data = [businessText, riskText, mdaText]
    elif section == 1:
        data = [businessText]
    elif section == 2:
        data = [riskText]
    elif section == 3:
        data = [mdaText]
    elif section == 4:
        data = [mdaText]
    return(data)