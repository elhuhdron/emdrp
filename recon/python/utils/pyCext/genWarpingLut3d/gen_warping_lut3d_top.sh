
TMPDIR=tmp
#CONNFG=6
#CONNBG=26
CONNFG=26
CONNBG=6

mkdir -p $TMPDIR
rm -rf $TMPDIR/in-$CONNFG-$CONNBG-*.txt
rm -rf $TMPDIR/out-$CONNFG-$CONNBG-*.txt

for range in `seq 1 8`;
do
    # for debug
    #echo "gen_warping_lut3d_create($CONNFG, $CONNBG, 2^13, $range, 10000)" > $TMPDIR/in-$CONNFG-$CONNBG-$range.txt
    echo "gen_warping_lut3d_create($CONNFG, $CONNBG, 8, $range, 10000)" > $TMPDIR/in-$CONNFG-$CONNBG-$range.txt
    # mymatlabc is to whatever command line is needed to invoke matlab without java gui
    mymatlabc -nodisplay -nodesktop < $TMPDIR/in-$CONNFG-$CONNBG-$range.txt &> $TMPDIR/out-$CONNFG-$CONNBG-$range.txt &
done

