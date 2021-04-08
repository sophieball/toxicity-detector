text_based = ["perspective_score", "identity_attack",
                  "Please", "Please_start", "HASHEDGE", 
                  "Indirect_(btw)", 
                  "Hedges", 
                  "Factuality", "Deference", "Gratitude", "Apologizing", 
                  "1st_person_pl.", "1st_person", "1st_person_start", 
                  "2nd_person", "2nd_person_start",
         					"Indirect_(greeting)", "Direct_question", "Direct_start", 
                  "HASPOSITIVE", "HASNEGATIVE", "SUBJUNCTIVE", "INDICATIVE",
                ]

G_logs_based = ["rounds", "shepherd_time", "review_time"]
OSS_logs_based = ["rounds", "shepherd_time"]

drop_cols = ["Indirect_(btw)", "Indirect_(greeting)",
  "Apologizing", "Deference",
  "SUBJUNCTIVE", "INDICATIVE"]
text_based = list(set(text_based) - set(drop_cols))

length = ["length"]

def get_feature_set(dat):
  if dat == "G":
    logs_based = G_logs_based
  else:
    logs_based = OSS_logs_based

  if dat == "issues":
    feature_set = [ 
              text_based
    ]
  else: # code review comments in OSS and G share the same set of features
    feature_set = [
              logs_based,
              logs_based + length,
              text_based,
              text_based + logs_based + length,
              text_based + length,
              text_based + logs_based, 
     ]
  return feature_set
