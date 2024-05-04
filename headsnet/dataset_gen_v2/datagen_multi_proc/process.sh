rm -r xp
rm -r yp
rm train_x.dat
rm train_y.dat
mkdir xp
mkdir yp
for file in ../ply/*.ply; do
	nohup ./gen "$( basename $file )"  > /dev/null 2>&1 &
done;
