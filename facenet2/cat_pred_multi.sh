mkdir pred_multi_final
for i in {0..32768}; do
    cat pred_multi/train_y_$i.dat >> pred_multi_final/pred.dat
    cat pred_multi/train_y_$i.csv >> pred_multi_final/pred.csv
done