rm -f model.keras
rm -f predicted.csv
python3 fit.py
python3 pred.py
python3 view.py
