import pandas as pd
import random

dat = read_csv("src/data/pushback_survey_prs_apr_8.csv")

# set reported label as True
reported = dat.loc[dat["reported"] == True]
print(set(reported["thread_label"]), set(reported["label"]))
#reported["thread_label"] = True
#reported["label"] = True

negative = dat.loc[dat["thread_label"] == False]
neg_ids = list(set(negative["thread_id"]))
neg_ids_sample = random.sample(neg_ids, 3*len(set(reported["thread_id"])))

neg_sample = negative.loc[negative["thread_id"].isin(neg_ids_sample)]
print(len(neg_sample), len(reported),
len(set(reported["thread_id"])),len(set(neg_sample["thread_id"])))

new_dat = pd.concat([reported, neg_sample])

new_dat = new_dat[need_cols]
new_dat["training"] = True
new_dat.to_csv("src/data/pushback_reported_prs_w_neg.csv", index=False)
