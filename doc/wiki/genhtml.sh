#!/bin/bash
# Based on http://linguisticmystic.com/2015/03/02/how-to-make-a-website-using-pandoc/
# Requires pandoc http://pandoc.org/
# "Publish" github flavored markdown wiki tree into html
# Run from top folder ("wiki")

find . -name '*.md.tmp' -type f -exec rm {} \;
find . -name '*.md' -type f -exec perl -pe 'open (my $fh, ">>", "{}.tmp"); select $fh; s|([^\!])\[([^\]]+)\]\(([^)]+)\)|\1<a href="\3">\2</a>|g' {} \;
find . -name '*.md.tmp' -type f -exec perl -i -pe 's|\<a href="(.+).md"\>|<a href="\1.html">|g' {} \;
find . -name '*.md.tmp' -type f -exec perl -i -pe 's|\<a href="(.+).md\#(.+)"\>|<a href="\1.html#\2">|g' {} \;
find . -name '*.md.tmp' -type f -exec pandoc -s -f markdown_github -o {}.html {} \;
find . -depth -name '*.md.tmp.html' -execdir bash -c 'mv -f "$1" "${1//md.tmp.html/html}"' bash {} \;
find . -name '*.md.tmp' -type f -exec rm {} \;

# to purge all html
# xxx - if we ever hand create any html files in the wiki, this would be bad
#find . -name '*.html' -type f -exec rm {} \;
