mkdir -p pred_multi_final
rm -rf pred_multi_final/*
for i in {0..32768}; do
    if [ -f 'pred_multi/train_y_'$i'.dat' ]; then
        cat pred_multi/train_y_$i.dat >> pred_multi_final/pred.dat
        cat pred_multi/train_y_$i.csv >> pred_multi_final/pred.csv
    else
        echo 'File not found: pred_multi/train_y_'$i'.dat'
        echo 'Sequence broken, exiting. (incomplete prediction may likely exist)'
        exit 0
    fi
done
