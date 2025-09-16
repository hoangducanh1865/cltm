## CLTM

# Setup environment:
```
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Train:
```
bash bash/train_20NSshort_sigmoid.sh
bash bash/train_R21578title_sigmoid.sh
bash bash/train_TMNtitle_sigmoid.sh
```

# Evaluate:
```
bash bash/evaluate_20NSshort_sigmoid.sh
bash bash/evaluate_R21578title_sigmoid.sh
bash bash/evaluate_TMNtitle_sigmoid.sh
```