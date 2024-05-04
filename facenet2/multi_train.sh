p=0
for i in {0..32768}; do
	nohup xterm -e 'python3 fit.py train_y/train_y_'$i'.dat' > /dev/null 2>&1 &
    echo 'python3 fit.py train_y/train_y_'$i'.dat'
    p=$((p+1))
    if ((p >= 24)); then
        p=0
        read -p "Any key to continue... " > /dev/null
    fi
done;
