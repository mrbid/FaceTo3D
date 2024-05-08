p=0
for i in {0..32767}; do
    if [ ! -f 'models/train_y_'$i'.keras' ]; then
        python3 fit.py 'train_y/train_y_'$i'.dat' > /dev/null 2>&1 &
        echo 'python3 fit.py train_y/train_y_'$i'.dat'
        p=$((p+1))
        if ((p >= 42)); then
            p=0
            date
            while wait -f %% 2>/dev/null; do :; done
        fi
    fi
done;
