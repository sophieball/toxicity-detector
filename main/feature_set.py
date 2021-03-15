def get_feature_set(dat):
  if dat == "G":
     feature_set = [
                 ["rounds", "shepherd_time", "review_time"], # logs
                 ["rounds", "shepherd_time", "review_time", "length"], # logs+length
                ["perspective_score", "identity_attack",
                  "Please", "Please_start", "HASHEDGE", 
                  "Indirect_(btw)", 
                  "Hedges", 
                  "Factuality", "Deference", "Gratitude", "Apologizing", 
                  "1st_person_pl.", "1st_person", "1st_person_start", 
                  "2nd_person", "2nd_person_start",
         					"Indirect_(greeting)", "Direct_question", "Direct_start", 
                  "HASPOSITIVE", "HASNEGATIVE", "SUBJUNCTIVE", "INDICATIVE",
                  "length"
                ],
                ["perspective_score", "identity_attack",
                  "Please", "Please_start", "HASHEDGE", 
                  "Indirect_(btw)", 
                  "Hedges", 
                  "Factuality", "Deference", "Gratitude", "Apologizing", 
                  "1st_person_pl.", "1st_person", "1st_person_start", 
                  "2nd_person", "2nd_person_start",
         					"Indirect_(greeting)", "Direct_question", "Direct_start", 
                  "HASPOSITIVE", "HASNEGATIVE", "SUBJUNCTIVE", "INDICATIVE",
                ],
                ["perspective_score", "identity_attack", # perspective
                  "Please", "Please_start", "HASHEDGE", 
                  "Indirect_(btw)", 
                  "Hedges", 
                  "Factuality", "Deference", "Gratitude", "Apologizing", 
                  "1st_person_pl.", "1st_person", "1st_person_start", 
                  "2nd_person", "2nd_person_start",
         					"Indirect_(greeting)", "Direct_question", "Direct_start", 
                  "HASPOSITIVE", "HASNEGATIVE", "SUBJUNCTIVE", "INDICATIVE",
                  "rounds", "shepherd_time", "review_time" # logs-based
                ],
                ["perspective_score", "identity_attack",
                  "Please", "Please_start", "HASHEDGE", 
                  "Indirect_(btw)", 
                  "Hedges", 
                  "Factuality", "Deference", "Gratitude", "Apologizing", 
                  "1st_person_pl.", "1st_person", "1st_person_start", 
                  "2nd_person", "2nd_person_start",
         					"Indirect_(greeting)", "Direct_question", "Direct_start", 
                  "HASPOSITIVE", "HASNEGATIVE", "SUBJUNCTIVE", "INDICATIVE",
                  "length",
                  "rounds", "shepherd_time", "review_time"
                ],
            ]

  elif dat == "issues":
    feature_set = [ 
                ["perspective_score", "identity_attack",
                  "Please", "Please_start", "HASHEDGE", 
                  "Indirect_(btw)", 
                  "Hedges", 
                  "Factuality", "Deference", "Gratitude", "Apologizing", 
                  "1st_person_pl.", "1st_person", "1st_person_start", 
                  "2nd_person", "2nd_person_start",
         					"Indirect_(greeting)", "Direct_question", "Direct_start", 
                  "HASPOSITIVE", "HASNEGATIVE", "SUBJUNCTIVE", "INDICATIVE",
                  "length"
                ],
                ["perspective_score", "identity_attack",
                  "Please", "Please_start", "HASHEDGE", 
                  "Indirect_(btw)", 
                  "Hedges", 
                  "Factuality", "Deference", "Gratitude", "Apologizing", 
                  "1st_person_pl.", "1st_person", "1st_person_start", 
                  "2nd_person", "2nd_person_start",
         					"Indirect_(greeting)", "Direct_question", "Direct_start", 
                  "HASPOSITIVE", "HASNEGATIVE", "SUBJUNCTIVE", "INDICATIVE",
                ],
              ]

  elif dat == "pr":
    feature_set = [
                 ["rounds", "shepherd_time"], # logs
                 ["rounds", "shepherd_time", "length"], # logs+length
                ["perspective_score", "identity_attack",
                  "Please", "Please_start", "HASHEDGE", 
                  "Indirect_(btw)", 
                  "Hedges", 
                  "Factuality", "Deference", "Gratitude", "Apologizing", 
                  "1st_person_pl.", "1st_person", "1st_person_start", 
                  "2nd_person", "2nd_person_start",
         					"Indirect_(greeting)", "Direct_question", "Direct_start", 
                  "HASPOSITIVE", "HASNEGATIVE", "SUBJUNCTIVE", "INDICATIVE",
                  "length"
                ],
                ["perspective_score", "identity_attack",
                  "Please", "Please_start", "HASHEDGE", 
                  "Indirect_(btw)", 
                  "Hedges", 
                  "Factuality", "Deference", "Gratitude", "Apologizing", 
                  "1st_person_pl.", "1st_person", "1st_person_start", 
                  "2nd_person", "2nd_person_start",
         					"Indirect_(greeting)", "Direct_question", "Direct_start", 
                  "HASPOSITIVE", "HASNEGATIVE", "SUBJUNCTIVE", "INDICATIVE",
                ],
                ["perspective_score", "identity_attack", # perspective
                  "Please", "Please_start", "HASHEDGE", 
                  "Indirect_(btw)", 
                  "Hedges", 
                  "Factuality", "Deference", "Gratitude", "Apologizing", 
                  "1st_person_pl.", "1st_person", "1st_person_start", 
                  "2nd_person", "2nd_person_start",
         					"Indirect_(greeting)", "Direct_question", "Direct_start", 
                  "HASPOSITIVE", "HASNEGATIVE", "SUBJUNCTIVE", "INDICATIVE",
                  "rounds", "shepherd_time" # logs-based
                ],
                ["perspective_score", "identity_attack",
                  "Please", "Please_start", "HASHEDGE", 
                  "Indirect_(btw)", 
                  "Hedges", 
                  "Factuality", "Deference", "Gratitude", "Apologizing", 
                  "1st_person_pl.", "1st_person", "1st_person_start", 
                  "2nd_person", "2nd_person_start",
         					"Indirect_(greeting)", "Direct_question", "Direct_start", 
                  "HASPOSITIVE", "HASNEGATIVE", "SUBJUNCTIVE", "INDICATIVE",
                  "length",
                  "rounds", "shepherd_time"
                ],
             ]
  return feature_set
