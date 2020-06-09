import toxicity_report


def test_nontoxic():
  [report, time] = toxicity_report.compute_prediction_report("test")
  assert report["score"] == 0
  assert report["en"] == 1
  assert report["orig"]["persp_raw"]["detectedLanguages"] == ["en"]


def test_empty():
  [report, time] = toxicity_report.compute_prediction_report("")
  assert report["score"] == 0
  assert report["reason"] == "empty"


def test_code():
  # should behave like empty
  [report, time] = toxicity_report.compute_prediction_report("`hate`")
  assert report["score"] == 0
  assert report["reason"] == "empty"


def test_toxic():
  [report, time] = toxicity_report.compute_prediction_report("I hate you")
  assert report["score"] == 1
  assert report["en"] == 1
  assert report["orig"]["persp_raw"]["detectedLanguages"] == ["en"]


def test_toxic():
  [report,
   time] = toxicity_report.compute_prediction_report("Kill this process!")
  assert report["score"] == 1
  assert report["alt_tried"] == 2


# def test_toxic():
#   [report, time] = toxicity_report.compute_prediction_report("Terminate me!")
#   print(report)
#   assert report["orig"]["score"] == 1
#   assert report["score"] == 0
#   assert report["alt_tried"] == 2
#   assert False


def test_alternative_text():
  assert set(toxicity_report.clean_text("this is stupid asfjasdfe")) == \
   set([ "potato is stupid asfjasdfe", "this potato stupid asfjasdfe", "this is stupid "])
  assert toxicity_report.clean_text("asfjasdfe") == \
   [ ""]
