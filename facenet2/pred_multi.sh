p=0
for i in {0..32768}; do
    if [ -f 'models/train_y_'$i'.keras' ]; then
        if [ ! -f 'pred_multi/train_y_'$i'.dat' ]; then
            python3 pred_multi.py 'models/train_y_'$i'.keras' > /dev/null 2>&1 &
            echo 'python3 pred_multi.py models/train_y_'$i'.keras'
            p=$((p+1))
            if ((p >= 42)); then
                p=0
                date
                while wait -f %% 2>/dev/null; do :; done
            fi
        fi
    fi
done;
