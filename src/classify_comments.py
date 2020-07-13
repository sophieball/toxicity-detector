# this will classify all issues and comments as toxic
# in mongodb, starting with the newest
import logging
# logging.basicConfig(filename='toxicity_issues.log',level=logging.INFO)
# logging.basicConfig(level=logging.INFO)

import config

VERSION = "v2"
TABLE_PREFIX = "christian_toxic_"

logging.info("loading")
import pickle
import pymongo
import time
import toxicity_report
import statsd
from dateutil.parser import parse

monitor = statsd.StatsClient('localhost', 8125, prefix='toxic')


logging.info("connecting to database")
def connect_to_database():
	mongo_name = config.mongo["user"]
	mongo_password = config.mongo["passwd"]

	# Connect to the Mongo Database
	client = pymongo.MongoClient()
	db = client[config.mongo["db"]]
	db.authenticate(name=mongo_name, password=mongo_password)
	return db
db = connect_to_database()

logging.info("starting")





def get_next_date(table): # updated_at date as string
	r = db[TABLE_PREFIX+table].find_one(
			filter= {"toxicity."+VERSION: {"$exists": 0}}, 
			sort= [("updated_at", -1)]
		 )
	if r == None:
		return ""
	else:
		return r["updated_at"]

def claim_next(table): # [id, updated_at, time]
	start = time.time()
	r = db[TABLE_PREFIX+table].find_one_and_update(
			filter= {"toxicity."+VERSION: {"$exists": 0}}, 
			sort= [("updated_at", -1)], 
			update= {"$set":{"toxicity."+VERSION+".in_progress": 1 }} 
		 )
	if r == None:
		return None
	return [r["_id"], r["updated_at"], time.time() - start]

def get_text(table, id): # [text, time]
	start = time.time()
	i = db[table].find_one({"_id": id}, {"title":1, "body":1})
	text = ""
	if "title" in i:
		text += str(i["title"]) + ": "
	if "body" in i:
		text += str(i["body"])
	# print(text)
	return [text, time.time() - start]


def update_db(table, id, new_data):
	start = time.time()
	db[TABLE_PREFIX+table].update({"_id": id},{ "$set": new_data })
	return time.time() - start


def process_one_item(table):
	monitor.incr("process."+table)

	# grab the most recent issue to process
	[issue_id, d, t1] = claim_next(table)
	print(table, issue_id, d)

	# get the text
	[text, t2] = get_text(table, issue_id)

	# score the text
	[score_report, t3] = toxicity_report.compute_prediction_report(text)
	result = {"toxicity."+VERSION: score_report}

	# write results to db
	t4=update_db(table,issue_id,result)

	monitor.timing("db.next."+table,int(t1*1000))
	monitor.timing("db.gettext."+table,int(t2*1000))
	monitor.timing("db.writeresult."+table,int(t4*1000))
	monitor.timing("scoring."+table,int(t3*1000))
	monitor.gauge("lastprocessed."+table,parse(d).timestamp())
	if score_report["score"]==1:
		monitor.incr("foundtoxic."+table)
	#print("db time", t1, t2, t4, "scoring time", t3)

def process_100_items(table):
	for x in range(0, 99):
		process_one_item(table)


tables = ["issues", "issue_comments", "pull_requests", "pull_request_comments","commit_comments" ]

if __name__ == '__main__': 
	while True:
		nexts = list(map(get_next_date, tables))
		next_ = max(nexts)
		idx = nexts.index(next_)
		process_100_items(tables[idx])
	
