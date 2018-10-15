# This script annotates the books with the story tags
# It does so by reading the books and the corresponding stories from xlsx file
from xlrd import open_workbook
import re
wb = open_workbook('dataset.xlsx')
DIR1 = '/home/shirish/BTECHSEM2/project/books/'
blist = [
    'failing_forward.txt',
    'awaken_the_giant_within_en.txt',
    'leading_en.txt',
    'buffet_en.txt',
    'wait_en.txt',
    'how_to_win_friends_and_influence_people_en.txt',
    'thinking_fast_and_slow_en.txt',
    'the_4_hour_workweek_en.txt',
    'the_art_of_the_start_en.txt',
    'the_48_laws_of_power_en.txt',
    'the_7_habits_of_highly_effective_people_en.txt',
    'the_lean_startup_en.txt',
    'winning_en.txt',
    'drive_en.txt',
    'option_b_en.txt'
]
for s in wb.sheets():
    if s.name != 'Blinkist acces':
        values = []
        rowrange = [1, 10, 21, 32, 46, 52, 69, 77, 86, 95, 103, 111, 124, 139,
                    151, 161]
        for i in range(15):
            f1 = open(DIR1 + blist[i], "r")
            print blist[i],
            f1t = f1.read()
            f1.close()
            col = []
            f2 = open("new_" + blist[i], "w")
            for row in range(rowrange[i], rowrange[i + 1]):
                value = (s.cell(row, 3).value)
                value = value.encode('utf-8')
                col.append(value)
            print len(col)
            for i in range(len(col)):
                p1 = re.compile(col[i][:30], re.IGNORECASE)
                p2 = re.compile(col[i][len(col[i]) - 30:len(col[i])],
                                re.IGNORECASE)
                f1t = p1.sub("<story>" + col[i][:30], f1t)
                f1t = p2.sub(col[i][len(col[i]) - 30:len(col[i])] + "</story>",
                             f1t)
            f2.write(f1t)
            f2.close()
