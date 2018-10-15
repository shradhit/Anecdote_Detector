# This script extracts stories from the html files from
# blinkist and converts them to text files
from bs4 import BeautifulSoup


def totxt1(fname1, fname2):
    f = open(fname1, "r")
    html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    f2 = open(fname2, "w")
    for i in soup.find_all("div", "chapter chapter"):
        h1 = i.find_all("h1")
        d1 = i.find_all("div")
        for h2, d2 in zip(h1, d1):
            f2.write(h2.string.encode('utf8'))
            f2.write(d2.text.encode('utf8'))
    f2.close()


def newfunc(name):
    name2 = name.replace("html", "txt")
    totxt1(name, name2)


f = open("failing_forward.html", "r")
html = f.read()
soup = BeautifulSoup(html, 'html.parser')
html = f.read()
newfunc("no_logo_en.html")
list_of_files = [
    "awaken_the_giant_within_en.html",
    "buffet_en.html",
    "drive_en.html",
    "failing_forward.html",
    "how_to_win_friends_and_influence_people_en.html",
    "option_b_en.html",
    "the_48_laws_of_power_en.html",
    "the_4_hour_workweek_en.html",
    "the_7_habits_of_highly_effective_people_en.html",
    "the_art_of_the_start_en.html",
    "the_lean_startup_en.html",
    "thinking_fast_and_slow_en.html",
    "wait_en.html",
    "winning_en.html"
]
for i in list_of_files:
    newfunc(i)
