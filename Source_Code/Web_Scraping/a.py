# This script will add spaces around the story tags (ie <story> and </story>)
import re

fname = "a1"
f = open(fname, "r")
s = f.read()
p = re.compile("<story>")
q = re.compile("</story>")
s = p.sub(" <story> ", s)
s = q.sub(" </story> ", s)
fname2 = "a2"
f = open(fname2, "w")
f.write(s)
